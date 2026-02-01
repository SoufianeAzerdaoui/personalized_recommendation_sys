from pyspark.sql import SparkSession, functions as F
import sys

spark = (
    SparkSession.builder
    .appName("BuildContentSmallDelta")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

def main(input_path: str, output_path: str, min_inter: int):
    df = spark.read.format("delta").load(input_path)

    df_small = (
        df.filter(F.col("total_interactions") >= F.lit(min_inter))
          .select("product_id","category_code","main_category","brand",
                  "avg_price","popularity_score","recency_days","total_interactions")
          .coalesce(8)
    )

    print("items small =", df_small.count())
    df_small.write.format("delta").mode("overwrite").save(output_path)
    print("✅ Content small dataset saved:", output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit build_content_small_delta.py <input> <output> <min_inter>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    spark.stop()
