from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CheckGoldFS").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_fs = spark.read.format("delta").load("data_lake/gold/fs_user_rt_delta")
item_fs = spark.read.format("delta").load("data_lake/gold/fs_item_rt_delta")

print("fs_user_count =", user_fs.count())
print("fs_item_count =", item_fs.count())

user_fs.orderBy("window_end", ascending=False).show(5, truncate=False)
item_fs.orderBy("window_end", ascending=False).show(5, truncate=False)
