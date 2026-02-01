from pyspark.sql import SparkSession
import sys

spark = (SparkSession.builder
  .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
  .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
  .getOrCreate())

path = sys.argv[1]
df = spark.read.format("delta").load(path)
df.printSchema()
print("rows =", df.count())
df.show(10, False)
spark.stop()
