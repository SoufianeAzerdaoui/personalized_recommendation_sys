import sys
from pyspark.sql import SparkSession, functions as F, Window

VALID_EVENTS = ["view", "cart", "purchase"]

def main(silver_path: str, recos_path: str, k: int = 5, days: int = 7, sample_frac: float = 1.0):
    spark = (
        SparkSession.builder
        .appName("EvalContentItem2Item")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1) Load interactions (SILVER)
    inter = (
        spark.read.format("delta").load(silver_path)
        .select("user_id", "product_id", "event_time", "event_type")
        .filter(
            F.col("user_id").isNotNull() &
            F.col("product_id").isNotNull() &
            F.col("event_time").isNotNull() &
            F.col("event_type").isin(VALID_EVENTS)
        )
    )

    # optional sampling to speed up on laptop
    if sample_frac < 1.0:
        inter = inter.sample(withReplacement=False, fraction=sample_frac, seed=42)

    # 2) Limit to last N days (keeps it fast + realistic)
    max_ts = inter.agg(F.max("event_time").alias("m")).collect()[0]["m"]
    if max_ts is None:
        print("❌ No events after filters.")
        spark.stop()
        return

    cutoff = F.lit(max_ts) - F.expr(f"INTERVAL {int(days)} DAYS")
    inter = inter.filter(F.col("event_time") >= cutoff)

    print("=" * 80)
    print("EVAL SETTINGS")
    print("=" * 80)
    print(f"silver_path = {silver_path}")
    print(f"recos_path  = {recos_path}")
    print(f"days        = {days}")
    print(f"sample_frac = {sample_frac}")
    print(f"max_ts      = {max_ts}")
    print("=" * 80)

    # 3) Build next-item transitions per user: (A -> B)
    w = Window.partitionBy("user_id").orderBy(F.col("event_time").asc())

    seq = (
        inter
        .withColumn("next_item", F.lead("product_id", 1).over(w))
        .select(F.col("product_id").alias("src_item"), F.col("next_item").alias("tgt_item"))
        .filter(F.col("tgt_item").isNotNull())
        .dropDuplicates()
    )

    n_pairs = seq.count()
    print(f"pairs (A->B) = {n_pairs}")

    # 4) Load content recos
    recos = spark.read.format("delta").load(recos_path).select("product_id", "similar_items")

    # 5) Evaluate: hit if tgt_item in similar_items(src_item)
    joined = (
        seq.join(recos, seq.src_item == recos.product_id, "inner")
           .select("src_item", "tgt_item", "similar_items")
    )

    hit = joined.withColumn("is_hit", F.array_contains(F.col("similar_items"), F.col("tgt_item")).cast("int"))

    metrics = hit.agg(
        F.count("*").alias("n_pairs_joined"),
        F.avg("is_hit").alias("hitrate")
    )

    print("=" * 80)
    print(f"RESULTS (HitRate@{k})")
    print("=" * 80)
    metrics.show(truncate=False)

    # 6) Extra debug: distribution of k (how many recos per item)
    print("=" * 80)
    print("Reco list size distribution")
    print("=" * 80)
    recos.select(F.size("similar_items").alias("k")).groupBy("k").count().orderBy("k").show(50, truncate=False)

    spark.stop()


if __name__ == "__main__":
    """
    Usage:
      spark-submit eval_content_item2item.py <silver_delta_path> <recos_delta_path> [k] [days] [sample_frac]

    Example:
      spark-submit eval_content_item2item.py data_lake/silver/events_clean_delta data_lake/serving/content_item2item_topk_delta 5 7 1.0
    """
    if len(sys.argv) < 3:
        print("Usage: spark-submit eval_content_item2item.py <silver_delta_path> <recos_delta_path> [k] [days] [sample_frac]")
        sys.exit(1)

    silver_path = sys.argv[1]
    recos_path = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    days = int(sys.argv[4]) if len(sys.argv) >= 5 else 7
    sample_frac = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0

    main(silver_path, recos_path, k=k, days=days, sample_frac=sample_frac)
