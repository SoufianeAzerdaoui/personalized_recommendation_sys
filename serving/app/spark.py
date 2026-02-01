
from pyspark.sql import SparkSession
from serving.app.config import DELTA_COORD

_spark = None

def get_spark() -> SparkSession:
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("RecoServingAPI")
            .config("spark.jars.packages", DELTA_COORD)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark
