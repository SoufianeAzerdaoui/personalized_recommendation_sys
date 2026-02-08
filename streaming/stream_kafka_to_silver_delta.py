from pyspark.sql import SparkSession, functions as F, types as T

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "ecom.events.raw"

OUT_SILVER_STREAM = "data_lake/silver/events_stream_silver_delta"
CHK = "data_lake/_checkpoints/events_stream_silver"

SCHEMA = T.StructType([
    T.StructField("event_time", T.TimestampType(), True),
    T.StructField("event_type", T.StringType(), True),
    T.StructField("product_id", T.LongType(), True),
    T.StructField("category_id", T.LongType(), True),
    T.StructField("category_code", T.StringType(), True),
    T.StructField("brand", T.StringType(), True),
    T.StructField("price", T.DoubleType(), True),
    T.StructField("user_id", T.LongType(), True),
    T.StructField("user_session", T.StringType(), True),
    T.StructField("main_category", T.StringType(), True),
    T.StructField("event_date", T.DateType(), True),
    T.StructField("ts_ms", T.LongType(), True),
])

def main():
    spark = (
        SparkSession.builder
        .appName("StreamKafkaToSilverDelta")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )


    parsed = (
        raw.select(F.col("value").cast("string").alias("json_str"))
        .select(F.from_json("json_str", SCHEMA).alias("e"))
        .select("e.*")
        .withColumn("ingest_ts", F.current_timestamp())
    )

    q = (
        parsed.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHK)
        .start(OUT_SILVER_STREAM)
    )

    q.awaitTermination()

if __name__ == "__main__":
    main()
