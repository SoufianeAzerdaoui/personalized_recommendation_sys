from pyspark.sql import SparkSession

def main():
    spark = (SparkSession.builder
        .appName("CountPowerBIExports")
        .getOrCreate()
    )

    paths = {
        "serving_logs": "exports/powerbi/serving_logs_flat",
        "serving_items": "exports/powerbi/serving_items_flat",
        "kpis_daily": "exports/powerbi/kpis_daily",
    }

    for name, path in paths.items():
        df = spark.read.parquet(path)
        print(f"{name}: {df.count()} rows")

    spark.stop()

if __name__ == "__main__":
    main()
