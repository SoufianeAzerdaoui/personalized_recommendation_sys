import sys
from pyspark.sql import SparkSession, functions as F, Window

def main(silver_path, als_recos_path, k, days, sample_frac):
    spark = (
        SparkSession.builder
        .appName("EvalALS_TopK_Long")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    k = int(k)
    days = int(days)
    sample_frac = float(sample_frac)

    # --- Load ---
    events = spark.read.format("delta").load(silver_path)
    als    = spark.read.format("delta").load(als_recos_path)

    # --- Detect event cols ---
    cols = set(events.columns)

    user_candidates = [c for c in ["user_id", "visitorid", "uid"] if c in cols]
    item_candidates = [c for c in ["product_id", "item_id", "productid"] if c in cols]
    ts_candidates   = [c for c in ["event_time", "timestamp", "ts", "datetime"] if c in cols]

    if not user_candidates or not item_candidates or not ts_candidates:
        raise ValueError(f"Events columns not compatible. columns={events.columns}")

    user_col = user_candidates[0]
    item_col = item_candidates[0]
    ts_col   = ts_candidates[0]

    # --- Filter last N days ---
    max_ts = events.select(F.max(F.col(ts_col)).alias("m")).collect()[0]["m"]
    ev = events.filter(F.col(ts_col) >= (F.lit(max_ts).cast("timestamp") - F.expr(f"INTERVAL {days} DAYS")))

    if sample_frac < 1.0:
        ev = ev.sample(False, sample_frac, seed=42)

    # --- Build target = last item per user ---
    w_last = Window.partitionBy(user_col).orderBy(F.col(ts_col).desc())
    targets = (
        ev.select(user_col, item_col, ts_col)
          .withColumn("rn", F.row_number().over(w_last))
          .filter(F.col("rn") == 1)
          .select(
              F.col(user_col).alias("user_id_eval"),
              F.col(item_col).alias("target_item")
          )
    )

    # --- Build ALS TopK per user (from long format) ---
    w_rank = Window.partitionBy("user_id").orderBy(F.col("score").desc())
    topk = (
        als.select("user_id", "product_id", "score")
           .withColumn("rk", F.row_number().over(w_rank))
           .filter(F.col("rk") <= k)
           .groupBy("user_id")
           .agg(F.collect_set("product_id").alias("topk_items"))
    )

    # --- Join & compute hitrate ---
    joined = (
        targets.join(topk, targets.user_id_eval == topk.user_id, "left")
               .select("user_id_eval", "target_item", "topk_items")
               .withColumn("hit", F.array_contains(F.col("topk_items"), F.col("target_item")).cast("int"))
    )

    res = joined.agg(
        F.count("*").alias("n_users_eval"),
        F.sum("hit").alias("n_hits"),
        F.mean("hit").alias(f"hitrate@{k}")
    )

    joined_has = joined.filter(F.col("topk_items").isNotNull())

    joined_has.agg(
        F.count("*").alias("n_users_with_reco"),
        F.sum("hit").alias("n_hits"),
        F.mean("hit").alias(f"hitrate@{k}_on_covered_users")
    ).show(truncate=False)

    coverage = joined.select(F.mean((F.col("topk_items").isNotNull()).cast("double")).alias("coverage_rate"))
    coverage.show(truncate=False)


    print("Joined columns:", joined.columns)

    print("="*80)
    print("ALS EVAL SETTINGS (LONG FORMAT)")
    print("="*80)
    print(f"silver_path    = {silver_path}")
    print(f"als_path       = {als_recos_path}")
    print(f"days           = {days}")
    print(f"sample_frac    = {sample_frac}")
    print(f"events user    = {user_col}")
    print(f"events item    = {item_col}")
    print(f"events ts      = {ts_col}")
    print("="*80)
    res.show(truncate=False)

    # Coverage (combien d'utilisateurs ont une reco topK)
    joined.select(F.when(F.col("topk_items").isNull(), F.lit(0)).otherwise(F.lit(1)).alias("has_reco")) \
          .groupBy("has_reco").count().show()

    spark.stop()

if __name__ == "__main__":
    # args: silver_path als_recos_path k days sample_frac
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
