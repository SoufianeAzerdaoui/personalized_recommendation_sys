from fastapi import APIRouter
from datetime import datetime
from pyspark.sql import functions as F
import time
import uuid

from serving.app.config import SERVE_LOGS_PATH
from serving.app.logging import write_serve_log

from serving.app.spark import get_spark
from serving.app.config import (
    HYBRID_RECOS_PATH, FS_USER_PATH, FS_ITEM_PATH,
    ALPHA, BETA, GAMMA, CANDIDATES_N
)
from serving.app.schemas import RecommendResponse, RecommendationItem
from serving.app.ranking import (
    explode_offline_recos,
    compute_user_activity,
    item_popularity_df,
    trending_fallback
)

router = APIRouter()

@router.get("/health")
def health():
    _ = get_spark()
    return {"status": "ok"}

@router.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10):
    spark = get_spark()
    t0 = time.time()
    request_id = str(uuid.uuid4())
    served_at = datetime.utcnow()

    # Load offline hybrid recos (1 row/user)
    offline = (
        spark.read.format("delta")
        .load(HYBRID_RECOS_PATH)
        .filter(F.col("user_id") == user_id)
        .select("user_id", "reco_items", "reco_scores")
        .limit(1)
    )

    fs_item = spark.read.format("delta").load(FS_ITEM_PATH)

    # === Cold-start fallback
    if offline.count() == 0:
        tr = trending_fallback(fs_item, k)

        recs = [
            RecommendationItem(product_id=int(r["product_id"]), score=float(round(r["final_score"], 4)))
            for r in tr.collect()
        ]

        latency_ms = int((time.time() - t0) * 1000)

        write_serve_log(
            spark=spark,
            out_path=SERVE_LOGS_PATH,
            request_id=request_id,
            served_at=served_at,
            user_id=user_id,
            k=k,
            mode="fallback_trending",
            candidate_count=None,
            product_ids=[r.product_id for r in recs],
            scores=[r.score for r in recs],
            message="fallback=trending_items",
            user_activity=None,
            latency_ms=latency_ms,
        )

        return RecommendResponse(
            user_id=user_id,
            generated_at=datetime.utcnow().isoformat(),
            recommendations=recs,
            mode="fallback_trending",
            message="fallback=trending_items"
        )

    # Explode arrays -> product_id + offline_score
    offline_exploded = explode_offline_recos(offline).orderBy(F.col("offline_score").desc()).limit(CANDIDATES_N)

    # User RT activity
    user_rt = (
        spark.read.format("delta").load(FS_USER_PATH)
        .filter(F.col("user_id") == user_id)
        .orderBy(F.col("window_end").desc())
        .limit(1)
    )
    user_row = user_rt.collect()[0] if user_rt.count() > 0 else None
    user_activity = compute_user_activity(user_row)

    # Item RT popularity
    items_pop = item_popularity_df(fs_item)

    # Re-ranking
    scored = (
        offline_exploded.join(items_pop, "product_id", "left")
        .fillna(0.0, subset=["item_popularity_rt"])
        .withColumn(
            "final_score",
            F.lit(ALPHA) * F.col("offline_score")
            + F.lit(BETA) * F.col("item_popularity_rt")
            + F.lit(GAMMA) * F.lit(user_activity)
        )
        .orderBy(F.col("final_score").desc())
        .limit(k)
    )

    recs = [
        RecommendationItem(product_id=int(r["product_id"]), score=float(round(r["final_score"], 4)))
        for r in scored.collect()
    ]

    latency_ms = int((time.time() - t0) * 1000)

    write_serve_log(
        spark=spark,
        out_path=SERVE_LOGS_PATH,
        request_id=request_id,
        served_at=served_at,
        user_id=user_id,
        k=k,
        mode="hybrid_rerank",
        candidate_count=CANDIDATES_N,
        product_ids=[r.product_id for r in recs],
        scores=[r.score for r in recs],
        message=None,
        user_activity=user_activity,
        latency_ms=latency_ms,
    )

    return RecommendResponse(
        user_id=user_id,
        generated_at=datetime.utcnow().isoformat(),
        recommendations=recs,
        mode="hybrid_rerank"
    )
