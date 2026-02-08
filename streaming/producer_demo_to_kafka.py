from pyspark.sql import SparkSession, functions as F, types as T

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "ecom.events.raw"

def main(rows_per_sec: int = 50):
    spark = SparkSession.builder.appName("ProducerDemoToKafka").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Générateur d'événements continu
    rate = (
        spark.readStream.format("rate")
        .option("rowsPerSecond", rows_per_sec)
        .load()
    )

    # Fabrique des colonnes réalistes
    # - user_id et product_id varient
    # - event_time = maintenant => toujours des nouvelles fenêtres
    events = (
        rate
        .withColumn("event_time", F.current_timestamp())
        .withColumn("event_type",
                    F.when((F.col("value") % 20) == 0, F.lit("purchase"))
                     .when((F.col("value") % 5) == 0, F.lit("cart"))
                     .otherwise(F.lit("view")))
        .withColumn("user_id", (F.col("value") % 5000).cast("long") + F.lit(10000000))
        .withColumn("product_id", (F.col("value") % 2000).cast("long") + F.lit(200000))
        .withColumn("category_id", (F.col("value") % 50).cast("long") + F.lit(10))
        .withColumn("category_code", F.concat(F.lit("cat_"), (F.col("value") % 50).cast("string")))
        .withColumn("brand", F.concat(F.lit("brand_"), (F.col("value") % 20).cast("string")))
        .withColumn("price", (F.rand() * 200).cast("double"))
        .withColumn("user_session", F.concat_ws("-", F.lit("sess"), F.col("user_id").cast("string"), (F.col("value") % 1000).cast("string")))
        .withColumn("main_category", F.concat(F.lit("main_"), (F.col("value") % 10).cast("string")))
        .withColumn("event_date", F.to_date("event_time"))
        .withColumn("ts_ms", (F.col("event_time").cast("double") * 1000).cast("long"))
    )

    payload = (
        events
        .select(
            F.to_json(F.struct(
                "event_time", "event_type", "product_id", "category_id",
                "category_code", "brand", "price", "user_id",
                "user_session", "main_category", "event_date", "ts_ms"
            )).alias("value"),
            F.col("user_id").cast("string").alias("key")
        )
    )

    q = (
        payload.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", TOPIC)
        .option("checkpointLocation", "data_lake/_checkpoints/producer_demo_kafka")
        .outputMode("append")
        .start()
    )

    q.awaitTermination()

if __name__ == "__main__":
    main()
