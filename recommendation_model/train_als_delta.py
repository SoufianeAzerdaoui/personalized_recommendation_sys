import sys
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


# ============================================================
# SparkSession (Delta)
# ============================================================
spark = (
    SparkSession.builder
    .appName("TrainALSImplicitDelta")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


def compute_ranking_metrics(pred_df, truth_df, k=50):
    """
    pred_df: columns [user_id, predictions] where predictions = array<product_id>
    truth_df: columns [user_id, ground_truth] where ground_truth = array<product_id>
    """
    joined = (
        truth_df.join(pred_df, on="user_id", how="inner")
                .select("predictions", "ground_truth")
    )

    # RankingMetrics wants RDD[(pred: List, labels: List)]
    rdd = joined.rdd.map(lambda r: (r["predictions"], r["ground_truth"]))
    metrics = RankingMetrics(rdd)

    # recallAtK: average over users
    recall_at_k = metrics.recallAt(k)

    # meanAveragePrecisionAtK exists in RankingMetrics (Spark)
    map_at_k = metrics.meanAveragePrecisionAt(k)

    return recall_at_k, map_at_k


def train_als(
    input_gold_als_delta: str,
    model_out: str,
    recos_out: str,
    metrics_out: str,
    k: int = 50,
    test_days: int = 7,
    rank: int = 64,
    reg: float = 0.08,
    iters: int = 15,
    alpha: float = 40.0,
    seed: int = 42,
):
    # ------------------------------------------------------------
    # 1) Read GOLD ALS (Delta)
    # ------------------------------------------------------------
    df = spark.read.format("delta").load(input_gold_als_delta)

    required = {"user_id", "product_id", "rating", "last_event_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {sorted(list(missing))}")

    # safety + cast to int for ALS
    df = (
        df.filter(
            F.col("user_id").isNotNull()
            & F.col("product_id").isNotNull()
            & F.col("rating").isNotNull()
            & (F.col("rating") > 0)
            & F.col("last_event_time").isNotNull()
        )
        .select(
            F.col("user_id").cast("int").alias("user_id"),
            F.col("product_id").cast("int").alias("product_id"),
            F.col("rating").cast("float").alias("rating"),
            F.col("last_event_time").alias("ts"),
        )
    )

    # ------------------------------------------------------------
    # 2) Temporal split: last N days = test
    # ------------------------------------------------------------
    max_ts = df.agg(F.max("ts").alias("m")).collect()[0]["m"]
    if max_ts is None:
        raise ValueError("No data in dataset after filters.")

    cutoff = max_ts - timedelta(days=test_days)

    train = df.filter(F.col("ts") < F.lit(cutoff)).select("user_id", "product_id", "rating")
    test = df.filter(F.col("ts") >= F.lit(cutoff)).select("user_id", "product_id", "rating")

    train.cache()
    test.cache()

    train_users = train.select("user_id").distinct()
    test_users = test.select("user_id").distinct()

    print("=" * 80)
    print("SPLIT INFO")
    print("=" * 80)
    print(f"Max timestamp: {max_ts}")
    print(f"Cutoff (test starts): {cutoff}  (test_days={test_days})")
    print(f"Train rows: {train.count()} | Test rows: {test.count()}")
    print(f"Train users: {train_users.count()} | Test users: {test_users.count()}")

    # ------------------------------------------------------------
    # 3) Train ALS implicit
    # ------------------------------------------------------------
    als = ALS(
        userCol="user_id",
        itemCol="product_id",
        ratingCol="rating",
        implicitPrefs=True,
        nonnegative=True,
        coldStartStrategy="drop",
        rank=rank,
        regParam=reg,
        maxIter=iters,
        alpha=alpha,
        seed=seed,
    )

    model = als.fit(train)

    # Save model
    model.write().overwrite().save(model_out)
    print(f"✅ Model saved to: {model_out}")

    # ------------------------------------------------------------
    # 4) Generate TopK recommendations for users in TEST
    # ------------------------------------------------------------
    recos = model.recommendForUserSubset(test_users, k)  # columns: user_id, recommendations(array<struct>)
    # flatten
    recos_flat = (
        recos
        .select(
            "user_id",
            F.explode("recommendations").alias("rec")
        )
        .select(
            "user_id",
            F.col("rec.product_id").alias("product_id"),
            F.col("rec.rating").alias("score")
        )
        .withColumn("generated_at", F.current_timestamp())
    )

    # also keep array form for metrics
    pred_for_metrics = (
        recos
        .select(
            "user_id",
            F.expr("transform(recommendations, x -> x.product_id)").alias("predictions")
        )
    )

    # ------------------------------------------------------------
    # 5) Build ground truth for RankingMetrics from TEST
    # ------------------------------------------------------------
    truth = (
        test.groupBy("user_id")
            .agg(F.collect_set("product_id").alias("ground_truth"))
    )

    # ------------------------------------------------------------
    # 6) Evaluate ranking metrics
    # ------------------------------------------------------------
    recall_at_k, map_at_k = compute_ranking_metrics(pred_for_metrics, truth, k=k)

    # simple coverage metrics
    n_users_eval = truth.count()
    distinct_reco_items = recos_flat.select("product_id").distinct().count()

    print("=" * 80)
    print(f"METRICS @K={k}")
    print("=" * 80)
    print(f"Users evaluated: {n_users_eval}")
    print(f"Recall@{k}: {recall_at_k}")
    print(f"MAP@{k}: {map_at_k}")
    print(f"Distinct recommended items (coverage): {distinct_reco_items}")

    # ------------------------------------------------------------
    # 7) Write recommendations + metrics (Delta)
    # ------------------------------------------------------------
    (recos_flat.write
        .format("delta")
        .mode("overwrite")
        .save(recos_out)
    )
    print(f"✅ Recommendations saved to: {recos_out}")

    metrics_df = spark.createDataFrame(
        [
            (
                datetime.utcnow().isoformat(),
                k,
                test_days,
                int(train.count()),
                int(test.count()),
                int(n_users_eval),
                float(recall_at_k),
                float(map_at_k),
                int(distinct_reco_items),
                rank,
                reg,
                iters,
                alpha,
            )
        ],
        schema="""
            run_utc string,
            k int,
            test_days int,
            train_rows long,
            test_rows long,
            users_evaluated long,
            recall_at_k double,
            map_at_k double,
            distinct_reco_items long,
            rank int,
            reg double,
            iters int,
            alpha double
        """
    )

    (metrics_df.write
        .format("delta")
        .mode("append")
        .save(metrics_out)
    )
    print(f"✅ Metrics appended to: {metrics_out}")

    train.unpersist()
    test.unpersist()


if __name__ == "__main__":
    """
    Usage:
      spark-submit --packages io.delta:delta-core_2.12:2.4.0 train_als_delta.py \
        <input_gold_als_delta> <model_out> <recos_out> <metrics_out> [k] [test_days]

    Example:
      spark-submit --packages io.delta:delta-core_2.12:2.4.0 train_als_delta.py \
        data_lake/gold/gold_als_interactions_delta \
        data_lake/models/als_implicit_delta \
        data_lake/serving/als_recos_topk_delta \
        data_lake/metrics/als_metrics_delta \
        50 7
    """
    if len(sys.argv) < 5:
        print("❌ Usage: train_als_delta.py <input_gold_als_delta> <model_out> <recos_out> <metrics_out> [k] [test_days]")
        sys.exit(1)

    input_path = sys.argv[1]
    model_out = sys.argv[2]
    recos_out = sys.argv[3]
    metrics_out = sys.argv[4]
    k = int(sys.argv[5]) if len(sys.argv) >= 6 else 50
    test_days = int(sys.argv[6]) if len(sys.argv) >= 7 else 7

    train_als(
        input_gold_als_delta=input_path,
        model_out=model_out,
        recos_out=recos_out,
        metrics_out=metrics_out,
        k=k,
        test_days=test_days,
    )

    spark.stop()
