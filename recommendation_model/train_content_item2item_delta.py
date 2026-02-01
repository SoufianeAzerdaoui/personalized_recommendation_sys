from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Normalizer, BucketedRandomProjectionLSH
)
from pyspark import StorageLevel
import sys


def main(input_path: str, output_path: str, k: int):
    spark = (
        SparkSession.builder
        .appName("TrainContentItem2Item_LaptopSafe")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # laptop-safe partitions
    N_PARTS = 16

    df = spark.read.format("delta").load(input_path)

    # 1) Clean/select
    df = (
        df.select(
            "product_id", "category_code", "main_category", "brand",
            "avg_price", "popularity_score", "recency_days"
        )
        .dropna(subset=["product_id"])
        .fillna({"avg_price": 0.0, "popularity_score": 0.0, "recency_days": 0})
        .fillna({"category_code": "Unknown", "main_category": "Unknown", "brand": "Unknown"})
        .dropDuplicates(["product_id"])
        .repartition(N_PARTS, "main_category")
    )

    # 2) Feature pipeline
    idx_cat = StringIndexer(inputCol="category_code", outputCol="category_idx", handleInvalid="keep")
    idx_main = StringIndexer(inputCol="main_category", outputCol="main_idx", handleInvalid="keep")
    idx_brand = StringIndexer(inputCol="brand", outputCol="brand_idx", handleInvalid="keep")

    ohe = OneHotEncoder(
        inputCols=["category_idx", "main_idx", "brand_idx"],
        outputCols=["category_ohe", "main_ohe", "brand_ohe"],
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=["category_ohe", "main_ohe", "brand_ohe",
                   "avg_price", "popularity_score", "recency_days"],
        outputCol="features_raw"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=False,
        withStd=True
    )

    normalizer = Normalizer(inputCol="features_scaled", outputCol="features", p=2.0)

    pipe = Pipeline(stages=[idx_cat, idx_main, idx_brand, ohe, assembler, scaler, normalizer])
    pipe_model = pipe.fit(df)

    vec = (
        pipe_model.transform(df)
        .select("product_id", "main_category", "features")
        .repartition(N_PARTS, "main_category")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    n_items = vec.count()
    print("items =", n_items)

    # Safety for k
    k = min(k, max(1, n_items - 1))
    print("k used =", k)

    # 3) LSH (restrictive)
    lsh = BucketedRandomProjectionLSH(
        inputCol="features",
        outputCol="hashes",
        bucketLength=0.8,
        numHashTables=1
    )
    lsh_model = lsh.fit(vec)

    # VERY IMPORTANT: tighten distance to reduce candidate explosion
    maxDist = 0.20

    a = vec.alias("a")
    b = vec.alias("b")

    joined = (
        lsh_model.approxSimilarityJoin(a, b, maxDist, distCol="dist")
        .select(
            F.col("datasetA.product_id").alias("product_id"),
            F.col("datasetA.main_category").alias("main_category"),
            F.col("datasetB.product_id").alias("sim_product_id"),
            F.col("datasetB.main_category").alias("sim_main_category"),
            F.col("dist")
        )
        .filter(F.col("product_id") != F.col("sim_product_id"))
        .filter(F.col("main_category") == F.col("sim_main_category"))
        .drop("sim_main_category")
    )

    # cosine similarity from L2 distance (vectors are L2-normalized)
    joined = joined.withColumn("score", F.lit(1.0) - (F.col("dist") * F.col("dist")) / F.lit(2.0))

    # IMPORTANT: filter weak similarities to reduce shuffle/output
    joined = joined.filter(F.col("score") >= F.lit(0.40))

    # 4) Top-K
    w = Window.partitionBy("product_id").orderBy(F.col("score").desc())
    ranked = joined.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= F.lit(k))

    topk = (
        ranked.groupBy("product_id")
        .agg(
            F.sort_array(
                F.collect_list(F.struct(F.col("score"), F.col("sim_product_id"))),
                asc=False
            ).alias("arr")
        )
        .select(
            "product_id",
            F.expr("transform(arr, x -> x.sim_product_id)").alias("similar_items"),
            F.expr("transform(arr, x -> round(x.score, 6))").alias("similar_scores"),
        )
        .withColumn("generated_at", F.current_timestamp())
    )

    (topk.coalesce(8)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_path)
    )

    print("✅ Content item2item saved:", output_path)

    vec.unpersist()
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit train_content_item2item_delta.py <input> <output> <k>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
