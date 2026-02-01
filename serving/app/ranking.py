# serving/app/ranking.py

from pyspark.sql import functions as F
from serving.app.config import USER_ACTIVITY_DIV

def explode_offline_recos(df_user_row):
    """
    Input schema (1 row):
      user_id, reco_items (array<long>), reco_scores (array<double>)
    Output schema:
      product_id, offline_score
    """
    return (
        df_user_row
        .select(F.explode(F.arrays_zip("reco_items", "reco_scores")).alias("z"))
        .select(
            F.col("z.reco_items").cast("long").alias("product_id"),
            F.col("z.reco_scores").cast("double").alias("offline_score"),
        )
    )

def compute_user_activity(user_rt_one_row) -> float:
    """
    user_activity = min(1, events_5m / USER_ACTIVITY_DIV)
    """
    if user_rt_one_row is None:
        return 0.0
    try:
        return min(1.0, float(user_rt_one_row["events_5m"]) / float(USER_ACTIVITY_DIV))
    except Exception:
        return 0.0

def item_popularity_df(fs_item_df):
    """
    item_popularity_rt = log1p(events_5m + 2*purchases_5m)
    """
    return (
        fs_item_df
        .select(
            F.col("product_id").cast("long").alias("product_id"),
            F.log1p(F.col("events_5m") + 2 * F.col("purchases_5m")).alias("item_popularity_rt")
        )
    )

def trending_fallback(fs_item_df, k: int):
    """
    Fallback cold-start:
      last window_end then sort by trend_score = events_5m + 2*purchases_5m
    Returns: product_id, final_score
    """
    last_win = fs_item_df.select(F.max("window_end").alias("mx")).collect()[0]["mx"]

    return (
        fs_item_df
        .filter(F.col("window_end") == last_win)
        .withColumn("final_score", (F.col("events_5m") + 2 * F.col("purchases_5m")).cast("double"))
        .select(F.col("product_id").cast("long").alias("product_id"), "final_score")
        .orderBy(F.col("final_score").desc())
        .limit(k)
    )
