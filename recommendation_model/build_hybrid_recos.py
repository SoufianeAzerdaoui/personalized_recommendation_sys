import sys
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.storagelevel import StorageLevel


def normalize_scores(df, score_col, group_col, out_col):
    """
    Normalize scores per group into [0,1] using min-max.
    If min==max => score becomes 1.0
    """
    w = Window.partitionBy(group_col)
    return (
        df.withColumn("s_min", F.min(F.col(score_col)).over(w))
          .withColumn("s_max", F.max(F.col(score_col)).over(w))
          .withColumn(
              out_col,
              F.when(F.col("s_max") == F.col("s_min"), F.lit(1.0))
               .otherwise((F.col(score_col) - F.col("s_min")) / (F.col("s_max") - F.col("s_min")))
          )
          .drop("s_min", "s_max")
    )


def main(
    silver_path,
    als_path,
    content_path,
    out_path,
    k,
    days,
    sample_frac
):
    spark = (
        SparkSession.builder
        .appName("BuildHybridRecos")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # safer defaults for local
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    k = int(k)
    days = int(days)
    sample_frac = float(sample_frac)

    # --- Weights (tunable) ---
    W_ALS = 0.65
    W_CONTENT = 0.30
    W_POP = 0.05

    # fallback pool size (IMPORTANT for laptop)
    POP_POOL = 300  # 200-500 is enough for fallback, avoid 5000+

    print("=" * 80)
    print("BUILD HYBRID SETTINGS")
    print("=" * 80)
    print("silver_path =", silver_path)
    print("als_path    =", als_path)
    print("content_path=", content_path)
    print("out_path    =", out_path)
    print("k =", k, "| days =", days, "| sample_frac =", sample_frac)
    print("weights: ALS=", W_ALS, "CONTENT=", W_CONTENT, "POP=", W_POP)
    print("POP_POOL =", POP_POOL)
    print("=" * 80)

    # --- Load events ---
    events = spark.read.format("delta").load(silver_path)
    cols = set(events.columns)

    user_col = [c for c in ["user_id", "visitorid", "uid"] if c in cols][0]
    item_col = [c for c in ["product_id", "item_id", "productid"] if c in cols][0]
    ts_col   = [c for c in ["event_time", "timestamp", "ts", "datetime"] if c in cols][0]

    # --- Last N days window ---
    max_ts = events.select(F.max(F.col(ts_col)).alias("m")).collect()[0]["m"]
    ev = events.filter(F.col(ts_col) >= (F.lit(max_ts).cast("timestamp") - F.expr(f"INTERVAL {days} DAYS")))

    if sample_frac < 1.0:
        ev = ev.sample(False, sample_frac, seed=42)

    # cache ev: used multiple times
    ev = ev.persist(StorageLevel.MEMORY_AND_DISK)

    # Users in the window
    users = ev.select(F.col(user_col).alias("user_id")).distinct()

    # --- History to filter "already seen" (same window) ---
    history = (
        ev.select(F.col(user_col).alias("user_id"), F.col(item_col).alias("seen_item"))
          .distinct()
          .repartition(8)
          .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # --- Target anchor: last item per user in the window ---
    w_last = Window.partitionBy(user_col).orderBy(F.col(ts_col).desc())
    last_item = (
        ev.select(F.col(user_col).alias("user_id"), F.col(item_col).alias("last_item"), F.col(ts_col))
          .withColumn("rn", F.row_number().over(w_last))
          .filter(F.col("rn") == 1)
          .select("user_id", "last_item")
          .repartition(8)
          .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # =========================================================
    # 1) ALS candidates (long format) => (user_id, item_id, raw_score, source)
    # =========================================================
    als = spark.read.format("delta").load(als_path)  # user_id, product_id, score, generated_at

    als_pre = als.select(
        F.col("user_id").cast("long").alias("user_id"),
        F.col("product_id").cast("long").alias("item_id"),
        F.col("score").cast("double").alias("raw_score")
    )

    w_rank_als = Window.partitionBy("user_id").orderBy(F.col("raw_score").desc())
    als_top = (
        als_pre.withColumn("rk", F.row_number().over(w_rank_als))
               .filter(F.col("rk") <= k * 2)  # keep a bit more then trim later
               .drop("rk")
               .withColumn("source", F.lit("als"))
    )
    als_top = normalize_scores(als_top, "raw_score", "user_id", "score_norm")

    # =========================================================
    # 2) Content candidates: last_item -> similar_items/scores
    # =========================================================
    content = spark.read.format("delta").load(content_path)  # product_id, similar_items, similar_scores, generated_at

    # join per user last_item = product_id in content
    cjoin = last_item.join(
        content.select(
            F.col("product_id").cast("long").alias("product_id"),
            "similar_items",
            "similar_scores"
        ),
        last_item.last_item.cast("long") == F.col("product_id"),
        "left"
    ).drop("product_id")

    # explode arrays into long candidates
    content_long = (
        cjoin.select(
            "user_id",
            "last_item",
            F.posexplode_outer("similar_items").alias("pos", "item_id"),
            F.col("similar_scores").alias("scores_arr")
        )
        .withColumn("raw_score", F.expr("scores_arr[pos]"))
        .drop("scores_arr", "pos")
        .filter(F.col("item_id").isNotNull())
        .withColumn("item_id", F.col("item_id").cast("long"))
        .withColumn("raw_score", F.col("raw_score").cast("double"))
        .withColumn("source", F.lit("content"))
    )
    content_long = normalize_scores(content_long, "raw_score", "user_id", "score_norm")

    # =========================================================
    # 3) Combine ALS + Content first, then filter seen/last_item
    # =========================================================
    cand_base = (
        als_top.select("user_id", "item_id", "score_norm", "source")
              .unionByName(content_long.select("user_id", "item_id", "score_norm", "source"))
    )

    # remove last_item itself
    cand_base = (
        cand_base.join(last_item, "user_id", "left")
                 .filter((F.col("last_item").isNull()) | (F.col("item_id") != F.col("last_item").cast("long")))
                 .drop("last_item")
    )

    # remove already seen
    cand_base = cand_base.join(
        history,
        (cand_base.user_id == history.user_id) & (cand_base.item_id == history.seen_item.cast("long")),
        "left_anti"
    )

    # =========================================================
    # 4) Popularity fallback ONLY for users with not enough candidates
    # =========================================================
    pop = (
        ev.groupBy(F.col(item_col).cast("long").alias("item_id"))
          .agg(F.count("*").cast("double").alias("raw_score"))
          .orderBy(F.col("raw_score").desc())
          .limit(POP_POOL)
    )

    # normalize pop globally
    pop_stats = pop.agg(F.min("raw_score").alias("mn"), F.max("raw_score").alias("mx")).collect()[0]
    mn, mx = float(pop_stats["mn"]), float(pop_stats["mx"])
    if mx == mn:
        pop = pop.withColumn("score_norm", F.lit(1.0))
    else:
        pop = pop.withColumn("score_norm", (F.col("raw_score") - F.lit(mn)) / (F.lit(mx) - F.lit(mn)))

    pop_small = pop.select("item_id", "score_norm")

    # count candidates per user
    cnt = cand_base.groupBy("user_id").agg(F.count("*").alias("n_cand"))

    # users needing fallback
    need_pop = (
        users.join(cnt, "user_id", "left")
             .fillna({"n_cand": 0})
             .filter(F.col("n_cand") < k)
             .select("user_id")
    )

    n_need_pop = need_pop.count()
    print("Users needing popularity fallback (n_cand < k):", n_need_pop)

    # safe small crossJoin: only missing users × POP_POOL
    pop_per_user = (
        need_pop.crossJoin(F.broadcast(pop_small))
                .withColumn("source", F.lit("pop"))
    )

    # remove already seen for pop too
    pop_per_user = pop_per_user.join(
        history,
        (pop_per_user.user_id == history.user_id) & (pop_per_user.item_id == history.seen_item.cast("long")),
        "left_anti"
    )

    # union all candidates
    cand = cand_base.unionByName(pop_per_user.select("user_id", "item_id", "score_norm", "source"))

    # =========================================================
    # 5) Weighted scoring + dedup (multi-source sum) + topK
    # =========================================================
    cand = cand.withColumn(
        "w",
        F.when(F.col("source") == "als", F.lit(W_ALS))
         .when(F.col("source") == "content", F.lit(W_CONTENT))
         .otherwise(F.lit(W_POP))
    ).withColumn("score", F.col("score_norm") * F.col("w"))

    agg = (
        cand.groupBy("user_id", "item_id")
            .agg(
                F.sum("score").alias("score"),
                F.collect_set("source").alias("sources")
            )
            .withColumn(
                "source",
                F.when(F.size("sources") > 1, F.lit("multi"))
                 .otherwise(F.element_at("sources", 1))
            )
            .drop("sources")
    )

    w_rank = Window.partitionBy("user_id").orderBy(F.col("score").desc())
    ranked = agg.withColumn("rk", F.row_number().over(w_rank)).filter(F.col("rk") <= k)

    out = (
        ranked.orderBy("user_id", "rk")
              .groupBy("user_id")
              .agg(
                  F.collect_list("item_id").alias("reco_items"),
                  F.collect_list(F.round("score", 6)).alias("reco_scores"),
                  F.collect_list("source").alias("reco_sources")
              )
              .withColumn("generated_at", F.current_timestamp())
    )

    # =========================================================
    # 6) Save delta (reduce output partitions to avoid too many files)
    # =========================================================
    out = out.coalesce(16)

    out.write.format("delta").mode("overwrite").save(out_path)

    print("=" * 80)
    print("HYBRID SAVED:", out_path)
    print("k =", k, "| days =", days, "| sample_frac =", sample_frac)
    print("=" * 80)
    out.select("user_id", "reco_items", "reco_sources", "generated_at").show(5, truncate=False)

    # clean
    last_item.unpersist()
    history.unpersist()
    ev.unpersist()

    spark.stop()


if __name__ == "__main__":
    # args:
    # silver_path als_path content_path out_path k days sample_frac
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
