from fastapi import FastAPI
from fastapi import HTTPException
from model.model_loader import load_artifacts, load_all_artifacts, get_loaded_versions
from api.schemas import TextRequest
from services.inference_service import predict_sentiment
import logging
import redis
import os
from typing import Optional

from prometheus_fastapi_instrumentator import Instrumentator

LOAD_ALL_MODELS = os.getenv("LOAD_ALL_MODELS", "False").lower() == "true"
default_version = os.getenv("DEFAULT_VERSION", "v1")
logging.basicConfig(level=logging.INFO)
app = FastAPI()
Instrumentator().instrument(app).expose(app)

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))


# Load artifacts when the server starts
@app.on_event("startup")
def startup_event():
    if LOAD_ALL_MODELS:
        load_all_artifacts()
    else:
        load_artifacts(default_version)
    app.state.redis = redis.Redis(
        host=redis_host, port=redis_port, decode_responses=True
    )


# Check health of the checkpoint
@app.get("/health")
def health_check():
    return {"status": "healthly"}


# get the list of all the loaded models
@app.get("/models")
def models():
    return {
        "loaded versions": get_loaded_versions(),
        "default_version": default_version,
    }


# Endpoint for model predictions
@app.post("/predict")
def predict(request: TextRequest, model_version: Optional[str] = None):
    try:
        return predict_sentiment(request.text, model_version, app.state.redis)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
