from pyspark.sql import SparkSession, functions as F

IN_SILVER_STREAM = "data_lake/silver/events_stream_silver_delta"
OUT_USER = "data_lake/gold/fs_user_rt_delta"
OUT_ITEM = "data_lake/gold/fs_item_rt_delta"
CHK_BASE = "data_lake/_checkpoints/feature_store"

def main():
    spark = SparkSession.builder.appName("StreamFeaturesToGoldDelta").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    e = spark.readStream.format("delta").load(IN_SILVER_STREAM)

    # ✅ on utilise ingest_ts (temps réel) pour les fenêtres
    e = e.withColumn("event_ts_rt", F.col("ingest_ts"))

    # ✅ watermark court + fenêtre courte -> résultats rapides
    e = e.withWatermark("event_ts_rt", "10 minutes")

    user_fs = (
        e.groupBy(F.col("user_id"), F.window("event_ts_rt", "5 minutes"))
        .agg(
            F.count("*").alias("events_5m"),
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("views_5m"),
            F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("carts_5m"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases_5m"),
            F.max("event_ts_rt").alias("last_ingest_ts"),
        )
        .select(
            "user_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "events_5m", "views_5m", "carts_5m", "purchases_5m", "last_ingest_ts"
        )
    )

    item_fs = (
        e.groupBy(F.col("product_id"), F.window("event_ts_rt", "5 minutes"))
        .agg(
            F.count("*").alias("events_5m"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases_5m"),
            F.max("event_ts_rt").alias("last_ingest_ts"),
        )
        .select(
            "product_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "events_5m", "purchases_5m", "last_ingest_ts"
        )
    )

    q1 = (
        user_fs.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHK_BASE}/fs_user")
        .start(OUT_USER)
    )

    q2 = (
        item_fs.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHK_BASE}/fs_item")
        .start(OUT_ITEM)
    )

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
