from pyspark.sql import SparkSession

PATH = "data_lake/serving/reco_served_logs_delta"

spark = (
    SparkSession.builder
    .appName("CheckServingLogs")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

df = spark.read.format("delta").load(PATH)
print("count =", df.count())
df.orderBy("served_at", ascending=False).show(20, truncate=False)
df.printSchema()



"""spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  serving/scripts/check_serving_logs.py

"""