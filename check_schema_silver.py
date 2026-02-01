from pyspark.sql import SparkSession

def main():
    spark = (
        SparkSession.builder
        .appName("CheckSchemaSilverEvents")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    path = "data_lake/silver/events_clean_delta"
    df = spark.read.format("delta").load(path)

    df.printSchema()
    df.show(5, truncate=False)
    print("count =", df.count())

if __name__ == "__main__":
    main()
