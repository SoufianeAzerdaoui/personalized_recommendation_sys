from pyspark.sql import SparkSession, functions as F
import sys

spark = (
    SparkSession.builder
    .appName("BuildALSSmallDelta")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

def main(input_path: str, output_path: str, min_user: int, min_item: int):
    df = spark.read.format("delta").load(input_path)

    df = df.select("user_id", "product_id", "rating", "last_event_time")

    user_ok = (
        df.groupBy("user_id")
          .count()
          .filter(F.col("count") >= min_user)
          .select("user_id")
    )

    item_ok = (
        df.groupBy("product_id")
          .count()
          .filter(F.col("count") >= min_item)
          .select("product_id")
    )

    df_small = (
        df.join(user_ok, "user_id", "inner")
          .join(item_ok, "product_id", "inner")
          .coalesce(8)
    )

    df_small.show(5, truncate=False)

    df_small.write.format("delta").mode("overwrite").save(output_path)
    print("✅ ALS small dataset saved:", output_path)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: spark-submit build_als_small_delta.py <input> <output> <min_user> <min_item>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    spark.stop()
