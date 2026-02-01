



lancer l'API via terminal  : uvicorn serving.app.main:app --host 0.0.0.0 --port 8000

curl "http://localhost:8000/health"

curl "http://localhost:8000/recommend?user_id=85786403&k=10"
