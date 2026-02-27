from model.model_loader import predict_raw
import time
import uuid
import logging
import redis
import json
import hashlib
from prometheus_client import Counter
from datetime import datetime, timezone
import os
import random

default_version = os.getenv("DEFAULT_VERSION", "v1")
canary_version = os.getenv("CANARY_VERSION", "v2")
canary_percentage = int(os.getenv("CANARY_PERCENT", "0"))

logger = logging.getLogger("inference")
# # Add redis client
# redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
CACHE_HITS = Counter("cache_hits_total", "Total Cache Hits")
CACHE_MISS = Counter("cache_miss_total", "Total Cache Misses")


def predict_sentiment(text: str, model_version: str | None, redis_client):
    request_id = str(uuid.uuid4())
    # A/B version splitting
    chosen_version = model_version
    x = random.randint(1, 100)
    if not chosen_version:
        # x = random.randint(1, 100)
        if x <= canary_percentage:
            chosen_version = canary_version
        else:
            chosen_version = default_version

    # Store the data in cache with a hashed key and not the original text to avoid data leakage
    cache_key = (
        f"sentiment:{chosen_version}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    )
    # Check if the text exists in cache key
    cached = redis_client.get(cache_key)
    if cached:
        CACHE_HITS.inc()  # To include the count of cache hit executions in prometheus monitoring
        start = time.perf_counter()
        result = json.loads(cached)
        latency_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(latency_ms, 2)  # Cache Latency
        result["request_id"] = request_id
        result["cache"] = "HIT"
        return result

    # If prediction not found in cache
    start = time.perf_counter()
    # Default to v1 model if chosen model does not exist to provide smooth fallback
    fall_back = False
    try:
        # fall_back = False
        result = predict_raw(text, chosen_version)
    except ValueError:
        chosen_version = default_version
        fall_back = True
        result = predict_raw(text, chosen_version)

    computed_at = datetime.now(timezone.utc).isoformat()
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 2)  # Model Latency
    result["request_id"] = request_id
    result["cache"] = "MISS"
    result["requested_version"] = model_version if model_version else "auto"
    result["served_version"] = chosen_version
    result["fallback_used"] = fall_back
    CACHE_MISS.inc()  # To include the count of cache miss executions in prometheus monitoring

    to_cached = {
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "model_version": chosen_version,
        "computed_at": computed_at,
    }

    # Store result in redis
    redis_client.setex(cache_key, 60, json.dumps(to_cached))

    logger.info(
        "inference_completed",
        extra={
            "request_id": result["request_id"],
            "latency_ms": result["latency_ms"],
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "inp_length": len(text) if text else 0,
            "requested_version": model_version if model_version else "auto",
            "default_version": default_version,
            "canary_version": canary_version,
            "canary_percent": canary_percentage,
            "roll": x,
        },
    )

    print(
        "debug routing--> ",
        {
            "request_id": result["request_id"],
            "latency_ms": result["latency_ms"],
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "inp_length": len(text) if text else 0,
            "requested_version": model_version if model_version else "auto",
            "default_version": default_version,
            "canary_version": canary_version,
            "canary_percent": canary_percentage,
            "roll": x,
        },
    )

    return result
