from pyspark.sql import SparkSession, functions as F

SERVE_LOGS = "data_lake/serving/reco_served_logs_delta"
OUT_BASE = "exports/powerbi"

def main():
    spark = (
        SparkSession.builder
        .appName("ExportPowerBI")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logs = spark.read.format("delta").load(SERVE_LOGS)

    # ---------------------------
    # 1) Logs flat (1 ligne = 1 requête)
    # ---------------------------
    logs_flat = (
        logs
        .withColumn("served_date", F.to_date("served_at"))
        .withColumn("served_hour", F.date_format("served_at", "yyyy-MM-dd HH:00:00"))
        .select(
            "request_id", "served_at", "served_date", "served_hour",
            "user_id", "k", "mode", "candidate_count",
            "message", "user_activity", "latency_ms",
            F.size("product_ids").alias("returned_items")
        )
    )

    # ---------------------------
    # 2) Items flat (1 ligne = 1 item recommandé)
    # ---------------------------
    items_flat = (
        logs
        .withColumn("served_date", F.to_date("served_at"))
        # zip(product_ids, scores) puis explode => (product_id, score)
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

    # ---------------------------
    # 3) KPIs journaliers
    # ---------------------------
    kpis_daily = (
        logs
        .withColumn("served_date", F.to_date("served_at"))
        .groupBy("served_date")
        .agg(
            F.count("*").alias("requests"),
            F.sum(F.when(F.col("mode") == "fallback_trending", 1).otherwise(0)).alias("fallback_requests"),
            F.avg("latency_ms").alias("latency_avg_ms"),
            F.expr("percentile_approx(latency_ms, 0.95)").alias("latency_p95_ms"),
            F.avg(F.when(F.col("mode") == "hybrid_rerank", F.col("user_activity")).otherwise(None)).alias("user_activity_avg"),
        )
        .withColumn("fallback_rate", F.col("fallback_requests") / F.col("requests"))
        .orderBy(F.col("served_date").desc())
    )

    # ---------------------------
    # Write Parquet (Power BI friendly)
    # ---------------------------
    (logs_flat.coalesce(1).write.mode("overwrite").parquet(f"{OUT_BASE}/serving_logs_flat"))
    (items_flat.coalesce(1).write.mode("overwrite").parquet(f"{OUT_BASE}/serving_items_flat"))
    (kpis_daily.coalesce(1).write.mode("overwrite").parquet(f"{OUT_BASE}/kpis_daily"))

    print("✅ Export done:")
    print(f" - {OUT_BASE}/serving_logs_flat")
    print(f" - {OUT_BASE}/serving_items_flat")
    print(f" - {OUT_BASE}/kpis_daily")

if __name__ == "__main__":
    main()
