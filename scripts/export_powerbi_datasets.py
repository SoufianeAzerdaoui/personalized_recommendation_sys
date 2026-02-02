from pathlib import Path
from pyspark.sql import SparkSession, functions as F

SERVE_LOGS = "data_lake/serving/reco_served_logs_delta"
OUT_BASE = "exports/powerbi"

# checkpoint simple: fichier texte contenant le dernier served_at exporté (ISO)
CHECKPOINT_PATH = f"{OUT_BASE}/_checkpoint_last_served_at.txt"


def read_checkpoint_iso() -> str | None:
    p = Path(CHECKPOINT_PATH)
    if not p.exists():
        return None
    s = p.read_text(encoding="utf-8").strip()
    return s if s else None


def write_checkpoint_iso(ts_iso: str) -> None:
    p = Path(CHECKPOINT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ts_iso, encoding="utf-8")


def main():
    spark = (
        SparkSession.builder
        .appName("ExportPowerBI_Incremental")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logs = spark.read.format("delta").load(SERVE_LOGS)

    # 1) Lire checkpoint
    last_iso = read_checkpoint_iso()
    if last_iso:
        # filtre strict pour éviter de ré-exporter les mêmes lignes
        logs_new = logs.where(F.col("served_at") > F.to_timestamp(F.lit(last_iso)))
        print(f"🔎 Incremental export: served_at > {last_iso}")
    else:
        logs_new = logs
        print("🔎 First export: no checkpoint found -> exporting ALL logs")

    # Si rien de nouveau, stop proprement
    if logs_new.rdd.isEmpty():
        print("✅ No new logs since last export. Nothing to do.")
        return

    # 2) Préparer colonnes temps
    logs_new = (
        logs_new
        .withColumn("served_date", F.to_date("served_at"))
        .withColumn("served_hour", F.date_format("served_at", "yyyy-MM-dd HH:00:00"))
    )

    # ---------------------------
    # A) Logs flat (1 ligne = 1 requête)
    # ---------------------------
    logs_flat = (
        logs_new.select(
            "request_id", "served_at", "served_date", "served_hour",
            "user_id", "k", "mode", "candidate_count",
            "message", "user_activity", "latency_ms",
            F.size("product_ids").alias("returned_items")
        )
    )

    # ---------------------------
    # B) Items flat (1 ligne = 1 item recommandé)
    # ---------------------------
    items_flat = (
        logs_new
        .withColumn("pairs", F.arrays_zip("product_ids", "scores"))
        .withColumn("pair", F.explode_outer("pairs"))
        .select(
            "request_id", "served_at", "served_date",
            "user_id", "mode", "k",
            F.col("pair.product_ids").alias("product_id"),
            F.col("pair.scores").alias("score"),
            "latency_ms"
        )
    )

    # Dates impactées par l’export incremental (important pour kpis_daily)
    impacted_dates = [r["served_date"] for r in logs_new.select("served_date").distinct().collect()]
    impacted_dates_str = [str(d) for d in impacted_dates]
    print(f"🗓️ impacted_dates={impacted_dates_str}")

    # ---------------------------
    # C) KPIs journaliers (recalcul uniquement des jours impactés)
    # ---------------------------
    # Pour recalculer correctement un jour, il faut TOUTES les lignes de ce jour
    logs_impacted_all = (
        logs
        .withColumn("served_date", F.to_date("served_at"))
        .where(F.col("served_date").isin(impacted_dates))
    )

    kpis_daily = (
        logs_impacted_all
        .groupBy("served_date")
        .agg(
            F.count("*").alias("requests"),
            F.sum(F.when(F.col("mode") == "fallback_trending", 1).otherwise(0)).alias("fallback_requests"),
            F.avg("latency_ms").alias("latency_avg_ms"),
            F.expr("percentile_approx(latency_ms, 0.95)").alias("latency_p95_ms"),
            F.avg(F.when(F.col("mode") == "hybrid_rerank", F.col("user_activity")).otherwise(None)).alias("user_activity_avg"),
        )
        .withColumn("fallback_rate", F.col("fallback_requests") / F.col("requests"))
    )

    # ---------------------------
    # 3) Write Parquet partitionné (append) + overwrite ciblé pour kpis partitions
    # ---------------------------
    out_logs = f"{OUT_BASE}/serving_logs_flat"
    out_items = f"{OUT_BASE}/serving_items_flat"
    out_kpis = f"{OUT_BASE}/kpis_daily"

    # Logs / Items: append partitionné => pas d’overwrite global
    (logs_flat
        .repartition("served_date")           # 1+ fichiers par partition (ok pour Power BI)
        .write
        .mode("append")
        .partitionBy("served_date")
        .parquet(out_logs)
    )

    (items_flat
        .repartition("served_date")
        .write
        .mode("append")
        .partitionBy("served_date")
        .parquet(out_items)
    )

    # KPIs: overwrite uniquement les partitions impacted_dates
    # -> nécessite spark.sql.sources.partitionOverwriteMode=dynamic
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    (kpis_daily
        .repartition("served_date")
        .write
        .mode("overwrite")                    # overwrite dynamique = seulement partitions présentes dans df
        .partitionBy("served_date")
        .parquet(out_kpis)
    )

    # 4) Update checkpoint avec le max served_at exporté
    max_ts = logs_new.agg(F.max("served_at").alias("mx")).collect()[0]["mx"]
    max_iso = max_ts.isoformat(sep=" ")
    write_checkpoint_iso(max_iso)

    # Résumé
    print("✅ Export done (incremental):")
    print(f" - {out_logs} (append by served_date)")
    print(f" - {out_items} (append by served_date)")
    print(f" - {out_kpis} (overwrite impacted served_date partitions)")
    print(f"✅ checkpoint updated: {CHECKPOINT_PATH} = {max_iso}")


if __name__ == "__main__":
    main()
