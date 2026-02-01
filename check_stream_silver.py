from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CheckStreamSilver").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load("data_lake/silver/events_stream_silver_delta")
print("silver_stream_count =", df.count())
df.orderBy("event_time", ascending=False).show(5, truncate=False)
