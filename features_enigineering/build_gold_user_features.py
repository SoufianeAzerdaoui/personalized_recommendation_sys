from pyspark.sql import SparkSession, functions as F, Window
import sys

# ============================================================
# SparkSession (Delta)
# ============================================================
spark = (
    SparkSession.builder
    .appName("BuildGoldUserFeatures")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    # Delta configs
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

def build_gold_user_features(input_path: str, output_path: str):
    # ------------------------------------------------------------
    # 1) Read SILVER events (DELTA)
    # ------------------------------------------------------------
    df = spark.read.format("delta").load(input_path)

    df = (
        df.select(
            "user_id", "user_session", "event_time", "event_type",
            "category_code", "brand", "price"
        )
        .filter(F.col("user_id").isNotNull() & F.col("event_time").isNotNull())
    )

    # ------------------------------------------------------------
    # 2) Basic flags
    # ------------------------------------------------------------
    df = df.withColumn("is_purchase", (F.col("event_type") == F.lit("purchase")).cast("int"))
    df = df.withColumn("purchase_price", F.when(F.col("event_type") == "purchase", F.col("price")))
    df = df.withColumn("view_price", F.when(F.col("event_type") == "view", F.col("price")))

    # ------------------------------------------------------------
    # 3) A) Activity & engagement
    # ------------------------------------------------------------
    agg_activity = (
        df.groupBy("user_id")
          .agg(
              F.count("*").alias("total_events"),
              F.countDistinct("user_session").alias("total_sessions"),
              F.max("event_time").alias("last_event_time")
          )
          .withColumn(
              "avg_events_per_session",
              F.when(F.col("total_sessions") > 0, F.col("total_events") / F.col("total_sessions"))
               .otherwise(F.lit(0.0))
          )
          .withColumn("recency_days", F.datediff(F.current_date(), F.to_date("last_event_time")))
    )

    # ------------------------------------------------------------
    # 4) B) Conversion & value
    # ------------------------------------------------------------
    agg_conversion = (
        df.groupBy("user_id")
          .agg(
              F.sum("is_purchase").alias("purchase_count"),
              F.avg("view_price").alias("avg_price_viewed"),
              F.avg("purchase_price").alias("avg_price_purchased")
          )
    )

    # ------------------------------------------------------------
    # 5) C) Preferences
    # ------------------------------------------------------------
    agg_diversity = (
        df.groupBy("user_id")
          .agg(F.countDistinct("category_code").alias("distinct_categories"))
    )

    cat_counts = (
        df.groupBy("user_id", "category_code")
          .agg(F.count("*").alias("cnt"))
    )
    w_cat = Window.partitionBy("user_id").orderBy(F.desc("cnt"), F.asc("category_code"))
    fav_cat = (
        cat_counts
        .withColumn("rn", F.row_number().over(w_cat))
        .filter(F.col("rn") == 1)
        .select("user_id", F.col("category_code").alias("favorite_category"))
    )

    brand_df = df.filter(F.col("brand").isNotNull())
    brand_counts = (
        brand_df.groupBy("user_id", "brand")
                .agg(F.count("*").alias("cnt"))
    )
    w_brand = Window.partitionBy("user_id").orderBy(F.desc("cnt"), F.asc("brand"))
    fav_brand = (
        brand_counts
        .withColumn("rn", F.row_number().over(w_brand))
        .filter(F.col("rn") == 1)
        .select("user_id", F.col("brand").alias("favorite_brand"))
    )

    # ------------------------------------------------------------
    # 6) Final join + conversion_rate
    # ------------------------------------------------------------
    df_user = (
        agg_activity
        .join(agg_conversion, on="user_id", how="left")
        .join(agg_diversity, on="user_id", how="left")
        .join(fav_cat, on="user_id", how="left")
        .join(fav_brand, on="user_id", how="left")
        .withColumn(
            "conversion_rate",
            F.when(F.col("total_events") > 0, F.col("purchase_count") / F.col("total_events"))
             .otherwise(F.lit(0.0))
        )
        .fillna({
            "purchase_count": 0,
            "distinct_categories": 0,
            "conversion_rate": 0.0
        })
    )

    # ------------------------------------------------------------
    # 7) Write GOLD (DELTA)
    # ------------------------------------------------------------
    (
        df_user.write
        .format("delta")
        .mode("overwrite")
        .save(output_path)
    )

    print(f"✅ gold_user_features (DELTA) created: {output_path}")
    print("Schema:")
    df_user.printSchema()


if __name__ == "__main__":
    """
    Usage:
      spark-submit build_gold_user_features.py <input_silver_delta_path> <output_gold_delta_path>
    """
    if len(sys.argv) != 3:
        print("❌ Usage: spark-submit build_gold_user_features.py <input_path> <output_path>")
        sys.exit(1)

    build_gold_user_features(sys.argv[1], sys.argv[2])
    spark.stop()
