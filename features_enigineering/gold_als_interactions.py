from pyspark.sql import SparkSession, functions as F
import sys

# ============================================================
# SparkSession (Delta)
# ============================================================
spark = (
    SparkSession.builder
    .appName("GoldALSInteractions")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    # Delta
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

VALID_EVENTS = ["view", "cart", "remove_from_cart", "purchase"]

def build_gold_als(input_path: str, output_path: str):
    # 1) Read SILVER (DELTA)
    df = spark.read.format("delta").load(input_path)

    # 2) Safety filters
    df = df.select("user_id", "product_id", "event_time", "event_type") \
           .filter(
               F.col("event_type").isin(VALID_EVENTS) &
               F.col("user_id").isNotNull() &
               F.col("product_id").isNotNull() &
               F.col("event_time").isNotNull()
           )

    # 3) Weights
    df_weighted = df.withColumn(
        "weight",
        F.when(F.col("event_type") == "view", 1)
         .when(F.col("event_type") == "cart", 3)
         .when(F.col("event_type") == "purchase", 5)
         .when(F.col("event_type") == "remove_from_cart", -1)
         .otherwise(0)
    )

    # 4) Aggregate per (user_id, product_id)
    df_als = (
        df_weighted
        .groupBy("user_id", "product_id")
        .agg(
            F.sum("weight").cast("double").alias("rating"),
            F.max("event_time").alias("last_event_time")
        )
        .filter(F.col("rating") > 0)
    )

    # 5) Write GOLD (DELTA)
    (df_als.write
        .format("delta")
        .mode("overwrite")
        .save(output_path)
    )

    print(f"✅ gold_als_interactions (DELTA) created: {output_path}")
    df_als.printSchema()


if __name__ == "__main__":
    """
    Usage:
      spark-submit gold_als_interactions.py <input_silver_delta_path> <output_gold_delta_path>
    """
    if len(sys.argv) != 3:
        print("❌ Usage: spark-submit gold_als_interactions.py <input_path> <output_path>")
        sys.exit(1)

    build_gold_als(sys.argv[1], sys.argv[2])
    spark.stop()
