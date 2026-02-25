from fastapi import FastAPI
from model.model_loader import load_artifacts
from api.schemas import TextRequest
from services.inference_service import predict_sentiment
import logging
import redis
import os

from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
app = FastAPI()
Instrumentator().instrument(app).expose(app)

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))


# Load artifacts when the server starts
@app.on_event("startup")
def startup_event():
    load_artifacts()
    app.state.redis = redis.Redis(
        host=redis_host, port=redis_port, decode_responses=True
    )


# Check health of the checkpoint
@app.get("/health")
def health_check():
    return {"status": "healthly"}


# Endpoint for model predictions
@app.post("/predict")
def predict(request: TextRequest):
    return predict_sentiment(request.text, app.state.redis)
