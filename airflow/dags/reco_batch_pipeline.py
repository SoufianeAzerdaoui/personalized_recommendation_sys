from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/home/onizuka/projects/bigdata-recommender"

# Commande Spark standard avec Delta
SPARK_SUBMIT = f"""
cd {PROJECT_DIR} && \
source venv_spark/bin/activate && \
spark-submit \
  --packages io.delta:delta-core_2.12:2.4.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
"""

default_args = {
    "owner": "onizuka",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="reco_batch_pipeline",
    default_args=default_args,
    description="Batch pipeline: features -> train ALS/content -> hybrid -> export powerbi -> checks",
    start_date=datetime(2026, 2, 1),
    schedule_interval="@daily",   # tu peux mettre "@hourly" plus tard
    catchup=False,
    max_active_runs=1,
) as dag:

    # 1) Feature engineering (interactions ALS)
    t_features = BashOperator(
        task_id="features_gold_als_interactions",
        bash_command=f"{SPARK_SUBMIT} features_enigineering/gold_als_interactions.py",
    )

    # 2) Train ALS
    t_train_als = BashOperator(
        task_id="train_als",
        bash_command=f"{SPARK_SUBMIT} recommendation_model/train_als_delta.py",
    )

    # 3) Train content item2item
    t_train_content = BashOperator(
        task_id="train_content_item2item",
        bash_command=f"{SPARK_SUBMIT} recommendation_model/train_content_item2item_delta.py",
    )

    # 4) Build hybrid recos (writes to data_lake/serving/hybrid_recos_delta)
    t_build_hybrid = BashOperator(
        task_id="build_hybrid_recos",
        bash_command=f"{SPARK_SUBMIT} recommendation_model/build_hybrid_recos.py",
    )

    # 5) Export Power BI (si ton script existe déjà)
    # 👉 Si ton export est dans "scripts/export_powerbi.py" adapte le nom ici.
    # Si tu n'as pas de script unique, dis-moi et je te fais un "export_powerbi.py" clean.
    t_export_powerbi = BashOperator(
        task_id="export_powerbi",
        bash_command=f"{SPARK_SUBMIT} scripts/export_powerbi.py",
    )

    # 6) Health check API (FastAPI) (doit être up)
    t_check_api = BashOperator(
        task_id="check_serving_api_health",
        bash_command="curl -sf http://localhost:8000/health",
    )

    # 7) Check serving logs Delta (tu l’as déjà)
    t_check_serving_logs = BashOperator(
        task_id="check_serving_logs_delta",
        bash_command=f"{SPARK_SUBMIT} serving/scripts/check_serving_logs.py",
    )

    # Dépendances
    t_features >> [t_train_als, t_train_content] >> t_build_hybrid >> t_export_powerbi
    t_check_api >> t_check_serving_logs
