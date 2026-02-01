# serving/app/main.py

from fastapi import FastAPI
from serving.app.api import router

app = FastAPI(title="Recommendation Serving API")

app.include_router(router)
