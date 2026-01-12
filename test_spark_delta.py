from pyspark.sql import SparkSession
from delta import *

# Initialiser SparkSession avec Delta Lake
builder = SparkSession.builder \
    .appName("MiniTestDelta") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Création d'un DataFrame simple
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
columns = ["name", "age"]
df = spark.createDataFrame(data, columns)

print("=== DataFrame initial ===")
df.show()

# Chemin pour sauvegarder le Delta Table
delta_path = "/tmp/delta_test_table"

# Écriture en Delta Lake
df.write.format("delta").mode("overwrite").save(delta_path)

# Lecture depuis Delta Lake
df_delta = spark.read.format("delta").load(delta_path)

print("=== DataFrame lu depuis Delta Lake ===")
df_delta.show()

# Arrêter Spark
spark.stop()

