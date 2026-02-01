from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("check_small")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .getOrCreate())

df = spark.read.format("delta").load("data_lake/gold/gold_als_interactions_small_delta")
df.printSchema()
print("rows =", df.count())
df.show(5, False)

spark.stop()
