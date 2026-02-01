from pyspark.sql import functions as F, types as T
from datetime import datetime

# schema stable pour la table de logs
LOG_SCHEMA = T.StructType([
    T.StructField("request_id", T.StringType(), False),
    T.StructField("served_at", T.TimestampType(), False),
    T.StructField("user_id", T.LongType(), False),
    T.StructField("k", T.IntegerType(), False),
    T.StructField("mode", T.StringType(), False),
    T.StructField("candidate_count", T.IntegerType(), True),
    T.StructField("product_ids", T.ArrayType(T.LongType()), True),
    T.StructField("scores", T.ArrayType(T.DoubleType()), True),
    T.StructField("message", T.StringType(), True),
    T.StructField("user_activity", T.DoubleType(), True),
    T.StructField("latency_ms", T.LongType(), True),
])

def write_serve_log(
    spark,
    out_path: str,
    request_id: str,
    served_at: datetime,
    user_id: int,
    k: int,
    mode: str,
    candidate_count: int | None,
    product_ids: list[int] | None,
    scores: list[float] | None,
    message: str | None = None,
    user_activity: float | None = None,
    latency_ms: int | None = None,
):
    row = [(
        request_id,
        served_at,
        int(user_id),
        int(k),
        mode,
        int(candidate_count) if candidate_count is not None else None,
        [int(x) for x in product_ids] if product_ids is not None else None,
        [float(x) for x in scores] if scores is not None else None,
        message,
        float(user_activity) if user_activity is not None else None,
        int(latency_ms) if latency_ms is not None else None,
    )]

    df = spark.createDataFrame(row, schema=LOG_SCHEMA)

    # append en Delta (création auto si dossier vide)
    (
        df.write.format("delta")
        .mode("append")
        .save(out_path)
    )
