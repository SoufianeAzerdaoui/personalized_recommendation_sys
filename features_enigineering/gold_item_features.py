from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys

# ============================================================
# SparkSession (Delta)
# ============================================================
spark = (
    SparkSession.builder
    .appName("GoldItemFeatures")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    # Delta configs
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

VALID_EVENTS = ["view", "cart", "remove_from_cart", "purchase"]

def build_gold_item_features(input_path: str, output_path: str):
    # -----------------------------
    # 1) Read SILVER (DELTA)
    # -----------------------------
    df = spark.read.format("delta").load(input_path)

    # Safety filters (au cas où)
    df = df.filter(
        F.col("event_type").isin(VALID_EVENTS)
        & F.col("user_id").isNotNull()
        & F.col("product_id").isNotNull()
        & F.col("event_time").isNotNull()
        & (F.col("price").isNull() | (F.col("price") > 0))  # prix peut être null mais si présent >0
    )

    # -----------------------------
    # 2) Helper columns
    # -----------------------------
    df2 = (
        df
        .withColumn("is_purchase", F.when(F.col("event_type") == "purchase", F.lit(1)).otherwise(F.lit(0)))
        .withColumn("is_interaction", F.when(F.col("event_type").isin(["view", "cart", "purchase"]), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "main_category",
            F.when(
                F.col("category_code").isNotNull() & (F.col("category_code") != "Unknown"),
                F.split(F.col("category_code"), r"\.").getItem(0),
            ).otherwise(F.lit("Unknown"))
        )
    )

    # -----------------------------
    # 3) Aggregations per product
    # -----------------------------
    agg = (
        df2.groupBy("product_id")
           .agg(
               # Popularité
               F.sum("is_interaction").alias("total_interactions"),
               F.sum("is_purchase").alias("total_purchases"),
               (F.sum("is_purchase") / F.when(F.sum("is_interaction") > 0, F.sum("is_interaction")).otherwise(F.lit(None))).alias("purchase_rate"),

               # Pricing
               F.avg("price").alias("avg_price"),
               F.min("price").alias("min_price"),
               F.max("price").alias("max_price"),

               # Récence
               F.max("event_time").alias("last_event_time"),
           )
    )

    # -----------------------------
    # 4) Metadata (category_code / brand / main_category)
    #    -> prendre la valeur la plus fréquente (mode)
    # -----------------------------
    # category_code mode (ignore Unknown/null)
    cat_counts = (
        df2.filter(F.col("category_code").isNotNull() & (F.col("category_code") != "Unknown"))
           .groupBy("product_id", "category_code")
           .count()
    )
    w_cat = Window.partitionBy("product_id").orderBy(F.col("count").desc(), F.col("category_code").asc())
    top_cat = (
        cat_counts.withColumn("rn", F.row_number().over(w_cat))
                  .filter(F.col("rn") == 1)
                  .select("product_id", F.col("category_code").alias("category_code"))
    )

    # brand mode (ignore Unknown/null)
    brand_counts = (
        df2.filter(F.col("brand").isNotNull() & (F.col("brand") != "Unknown"))
           .groupBy("product_id", "brand")
           .count()
    )
    w_brand = Window.partitionBy("product_id").orderBy(F.col("count").desc(), F.col("brand").asc())
    top_brand = (
        brand_counts.withColumn("rn", F.row_number().over(w_brand))
                    .filter(F.col("rn") == 1)
                    .select("product_id", F.col("brand").alias("brand"))
    )

    # main_category mode (ignore Unknown/null)
    main_counts = (
        df2.filter(F.col("main_category").isNotNull() & (F.col("main_category") != "Unknown"))
           .groupBy("product_id", "main_category")
           .count()
    )
    w_main = Window.partitionBy("product_id").orderBy(F.col("count").desc(), F.col("main_category").asc())
    top_main = (
        main_counts.withColumn("rn", F.row_number().over(w_main))
                   .filter(F.col("rn") == 1)
                   .select("product_id", F.col("main_category").alias("main_category"))
    )

    df_item = (
        agg.join(top_cat, on="product_id", how="left")
           .join(top_main, on="product_id", how="left")
           .join(top_brand, on="product_id", how="left")
    )

    # Optionnel: remplacer les nulls metadata par "Unknown"
    df_item = df_item.fillna({
        "category_code": "Unknown",
        "main_category": "Unknown",
        "brand": "Unknown"
    })

    # -----------------------------
    # 5) Recency days + popularity score
    # -----------------------------
    df_item = df_item.withColumn(
        "recency_days",
        F.datediff(F.current_date(), F.to_date("last_event_time")).cast("int")
    )

    # popularity_score (robuste)
    df_item = df_item.withColumn(
        "popularity_score",
        (
            F.log1p(F.col("total_interactions").cast("double"))
            + F.lit(5.0) * F.coalesce(F.col("purchase_rate").cast("double"), F.lit(0.0))
            - F.lit(0.01) * F.coalesce(F.col("recency_days").cast("double"), F.lit(0.0))
        )
    )

    # -----------------------------
    # 6) Final schema
    # -----------------------------
    df_item_final = df_item.select(
        "product_id",
        "total_interactions",
        "total_purchases",
        "purchase_rate",
        "popularity_score",
        "avg_price",
        "min_price",
        "max_price",
        "category_code",
        "main_category",
        "brand",
        "last_event_time",
        "recency_days",
    )

    # -----------------------------
    # 7) Write GOLD (DELTA)
    # -----------------------------
    (
        df_item_final.write
        .format("delta")
        .mode("overwrite")
        .save(output_path)
    )

    print(f"✅ gold_item_features (DELTA) created: {output_path}")
    df_item_final.printSchema()


if __name__ == "__main__":
    """
    Usage:
      spark-submit gold_item_features.py <input_silver_delta_path> <output_gold_delta_path>
    """
    if len(sys.argv) != 3:
        print("❌ Usage: spark-submit gold_item_features.py <input_path> <output_path>")
        sys.exit(1)

    build_gold_item_features(sys.argv[1], sys.argv[2])
    spark.stop()
