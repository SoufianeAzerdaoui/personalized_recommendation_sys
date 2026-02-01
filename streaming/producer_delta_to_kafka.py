from pyspark.sql import SparkSession, functions as F

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "ecom.events.raw"
SILVER_EVENTS_PATH = "data_lake/silver/events_clean_delta"

def main(limit_rows: int = 20000):
    spark = SparkSession.builder.appName("ProducerDeltaToKafkaSpark").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.format("delta").load(SILVER_EVENTS_PATH)

    # IMPORTANT : test d'abord avec un limit
    df = df.orderBy(F.col("event_time")).limit(limit_rows)

    payload = (
        df.select(
            "event_time", "event_type", "product_id", "category_id",
            "category_code", "brand", "price", "user_id",
            "user_session", "main_category", "event_date"
        )
        .withColumn("ts_ms", (F.col("event_time").cast("double") * 1000).cast("long"))
        .withColumn("value", F.to_json(F.struct(
            "event_time", "event_type", "product_id", "category_id",
            "category_code", "brand", "price", "user_id",
            "user_session", "main_category", "event_date", "ts_ms"
        )))
        .selectExpr("CAST(user_id AS STRING) AS key", "CAST(value AS STRING) AS value")
        .repartition(3)  # align with 3 partitions topic
    )

    print("rows_to_send =", payload.count())

    # Envoi vers Kafka (batch)
    (payload.write
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", TOPIC)
        .save()
    )

    print("✅ Done: events sent to Kafka topic", TOPIC)

if __name__ == "__main__":
    main()
