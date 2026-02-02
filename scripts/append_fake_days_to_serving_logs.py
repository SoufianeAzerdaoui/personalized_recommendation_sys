from pyspark.sql import SparkSession, functions as F

PATH = "data_lake/serving/reco_served_logs_delta"

def main():
    spark = (
        SparkSession.builder
        .appName("AppendFakeDaysToServingLogs")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.format("delta").load(PATH)

    base = df.orderBy(F.col("served_at").desc()).limit(30)

    # crée 4 jours supplémentaires: J-1, J-2, J-3, J-4
    clones = []
    for d in [1, 2, 3, 4]:
        clones.append(
            base.withColumn("served_at", F.col("served_at") - F.expr(f"INTERVAL {d} DAYS"))
        )

    to_append = clones[0]
    for c in clones[1:]:
        to_append = to_append.unionByName(c)

    (to_append.write.format("delta").mode("append").save(PATH))
    print("✅ Added fake logs for days: -1, -2, -3, -4")

if __name__ == "__main__":
    main()
