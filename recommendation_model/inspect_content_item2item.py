from pyspark.sql import SparkSession
import sys

spark = (SparkSession.builder
    .appName("InspectContentItem2Item")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate())

path = sys.argv[1]
df = spark.read.format("delta").load(path)

print("count =", df.count())
df.printSchema()
df.show(5, truncate=False)

spark.stop()
