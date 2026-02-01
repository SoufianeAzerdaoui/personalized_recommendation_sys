# serving/app/config.py

DELTA_COORD = "io.delta:delta-core_2.12:2.4.0"

HYBRID_RECOS_PATH = "data_lake/serving/hybrid_recos_delta"
FS_USER_PATH = "data_lake/gold/fs_user_rt_delta"
FS_ITEM_PATH = "data_lake/gold/fs_item_rt_delta"

# reranking weights
ALPHA = 0.7
BETA = 0.2
GAMMA = 0.1

# serving params
CANDIDATES_N = 50
USER_ACTIVITY_DIV = 20.0  # min(1, events_5m / 20)


SERVE_LOGS_PATH = "data_lake/serving/reco_served_logs_delta"
