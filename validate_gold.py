#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -----------------------------
# Spark Session (Delta)
# -----------------------------
spark = (
    SparkSession.builder
    .appName("ValidateGoldsDelta")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    # Delta configs
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -----------------------------
# Helpers
# -----------------------------
def fail(msg: str):
    print(f"❌ FAIL: {msg}")
    spark.stop()
    sys.exit(1)

def ok(msg: str):
    print(f"✅ {msg}")

def read_delta(path: str):
    try:
        return spark.read.format("delta").load(path)
    except Exception as e:
        fail(f"Cannot read DELTA at {path}. Error: {e}")

def assert_expected_columns(df, expected_cols, strict=True):
    actual = set(df.columns)
    expected = set(expected_cols)

    missing = expected - actual
    extra = actual - expected

    if missing:
        fail(f"Missing columns: {sorted(list(missing))}")
    if strict and extra:
        fail(f"Extra columns present (strict schema): {sorted(list(extra))}")

    ok("Schema columns OK")

def null_report(df, cols):
    exprs = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in cols]
    return df.select(exprs)

def assert_no_nulls(df, cols):
    rep = null_report(df, cols).collect()[0].asDict()
    bad = {c: v for c, v in rep.items() if v != 0}
    if bad:
        fail(f"Nulls found in critical columns: {bad}")
    ok("Null checks OK")

def assert_unique(df, key_cols):
    total = df.count()
    distinct_cnt = df.select(*key_cols).distinct().count()
    if total != distinct_cnt:
        fail(f"Uniqueness failed for keys {key_cols}: total={total}, distinct={distinct_cnt}")
    ok(f"Uniqueness OK for keys {key_cols}")

def assert_range(df, condition_col, condition_desc):
    bad = df.filter(~condition_col).count()
    if bad != 0:
        fail(f"Range/business rule failed: {condition_desc}. Bad rows={bad}")
    ok(f"Rule OK: {condition_desc}")

def print_metrics_basic(df, name, id_col=None):
    print("\n" + "=" * 80)
    print(f"📊 METRICS — {name}")
    print("=" * 80)
    print(f"Rows: {df.count()}")
    if id_col:
        print(f"Distinct {id_col}: {df.select(id_col).distinct().count()}")
    print("Schema:")
    df.printSchema()

# -----------------------------
# GOLD 1 — ALS interactions
# -----------------------------
def validate_gold_als(path: str):
    print("\n" + "#" * 90)
    print(f"🥇 VALIDATION GOLD 1 — gold_als_interactions (DELTA)\nPath: {path}")
    print("#" * 90)

    df = read_delta(path)

    expected_cols = ["user_id", "product_id", "rating", "last_event_time"]
    assert_expected_columns(df, expected_cols, strict=False)

    assert_no_nulls(df, ["user_id", "product_id", "rating"])
    assert_range(df, F.col("rating") > 0, "rating > 0")
    assert_unique(df, ["user_id", "product_id"])

    print_metrics_basic(df, "gold_als_interactions")
    num_users = df.select("user_id").distinct().count()
    num_items = df.select("product_id").distinct().count()
    num_interactions = df.count()
    sparsity = 1 - (num_interactions / (num_users * num_items)) if num_users and num_items else None

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_interactions}")
    print(f"Sparsity: {sparsity}")

    print("\nRating describe:")
    df.select("rating").describe().show()

    print("Top 10 ratings:")
    df.orderBy(F.desc("rating")).show(10, truncate=False)

    ok("GOLD 1 validation PASSED")

# -----------------------------
# GOLD 2 — User features
# -----------------------------
def validate_gold_user_features(path: str):
    print("\n" + "#" * 90)
    print(f"🥇 VALIDATION GOLD 2 — gold_user_features (DELTA)\nPath: {path}")
    print("#" * 90)

    df = read_delta(path)

    expected_cols = [
        "user_id",
        "total_events",
        "total_sessions",
        "avg_events_per_session",
        "last_event_time",
        "recency_days",
        "purchase_count",
        "conversion_rate",
        "avg_price_viewed",
        "avg_price_purchased",
        "distinct_categories",
        "favorite_category",
        "favorite_brand",
    ]
    assert_expected_columns(df, expected_cols, strict=False)

    assert_no_nulls(df, ["user_id", "total_events", "total_sessions", "purchase_count", "conversion_rate"])
    assert_range(df, F.col("total_events") >= 0, "total_events >= 0")
    assert_range(df, F.col("total_sessions") >= 0, "total_sessions >= 0")
    assert_range(df, F.col("purchase_count") >= 0, "purchase_count >= 0")
    assert_range(df, (F.col("conversion_rate") >= 0) & (F.col("conversion_rate") <= 1), "conversion_rate in [0,1]")
    assert_unique(df, ["user_id"])

    print_metrics_basic(df, "gold_user_features", id_col="user_id")

    print("\nConversion rate describe:")
    df.select("conversion_rate").describe().show()

    print("Top 10 most active users (total_events):")
    df.orderBy(F.desc("total_events")).show(10, truncate=False)

    print("Top 10 buyers (purchase_count):")
    df.orderBy(F.desc("purchase_count")).show(10, truncate=False)

    ok("GOLD 2 validation PASSED")

# -----------------------------
# GOLD 3 — Item features
# -----------------------------
def validate_gold_item_features(path: str):
    print("\n" + "#" * 90)
    print(f"🥇 VALIDATION GOLD 3 — gold_item_features (DELTA)\nPath: {path}")
    print("#" * 90)

    df = read_delta(path)

    expected_cols = [
        "product_id",
        "total_interactions",
        "total_purchases",
        "purchase_rate",
        "popularity_score",
        "avg_price",
        "min_price",
        "max_price",
        "category_code",
        "main_category",
        "brand",
        "last_event_time",
        "recency_days",
    ]
    assert_expected_columns(df, expected_cols, strict=False)

    assert_no_nulls(df, ["product_id", "total_interactions", "total_purchases", "purchase_rate", "popularity_score"])
    assert_range(df, F.col("total_interactions") >= 0, "total_interactions >= 0")
    assert_range(df, F.col("total_purchases") >= 0, "total_purchases >= 0")
    assert_range(df, (F.col("purchase_rate") >= 0) & (F.col("purchase_rate") <= 1), "purchase_rate in [0,1]")
    assert_range(df, F.col("avg_price").isNull() | (F.col("avg_price") > 0), "avg_price > 0 (if not null)")
    assert_range(df, F.col("min_price").isNull() | (F.col("min_price") > 0), "min_price > 0 (if not null)")
    assert_range(df, F.col("max_price").isNull() | (F.col("max_price") > 0), "max_price > 0 (if not null)")
    assert_unique(df, ["product_id"])

    print_metrics_basic(df, "gold_item_features", id_col="product_id")

    print("\nPopularity score describe:")
    df.select("popularity_score").describe().show()

    print("Top 10 popular items (popularity_score):")
    df.orderBy(F.desc("popularity_score")).show(10, truncate=False)

    print("Top 10 categories by item count:")
    df.groupBy("main_category").count().orderBy(F.desc("count")).show(10, truncate=False)

    print("Top 10 brands by item count:")
    df.groupBy("brand").count().orderBy(F.desc("count")).show(10, truncate=False)

    ok("GOLD 3 validation PASSED")

# -----------------------------
# Main
# -----------------------------
def main():
    """
    Usage:
      spark-submit validate_golds_delta.py <gold1_delta> <gold2_delta> <gold3_delta>
    """
    if len(sys.argv) != 4:
        print("❌ Usage: spark-submit validate_golds_delta.py <gold1_path> <gold2_path> <gold3_path>")
        sys.exit(1)

    gold1 = sys.argv[1]
    gold2 = sys.argv[2]
    gold3 = sys.argv[3]

    validate_gold_als(gold1)
    validate_gold_user_features(gold2)
    validate_gold_item_features(gold3)

    print("\n" + "=" * 80)
    print("🎉 ALL GOLD DATASETS VALIDATED SUCCESSFULLY (Train-Ready).")
    print("=" * 80)

    spark.stop()
    sys.exit(0)

if __name__ == "__main__":
    main()
