from model.model_loader import predict_raw
import time
import uuid
import logging
import redis
import json
import hashlib
from prometheus_client import Counter

logger = logging.getLogger("inference")
# # Add redis client
# redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
CACHE_HITS = Counter("cache_hits_total", "Total Cache Hits")
CACHE_MISS = Counter("cache_miss_total", "Total Cache Misses")


def predict_sentiment(text: str, redis_client):
    request_id = str(uuid.uuid4())
    cache_key = f"sentiment:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    # Check if the text exists in cache key
    cached = redis_client.get(cache_key)
    if cached:
        CACHE_HITS.inc()
        start = time.perf_counter()
        result = json.loads(cached)
        latency_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(latency_ms, 2)  # Cache Latency
        result["request_id"] = request_id
        result["cache"] = "HIT"
        return result

    # If prediction not found in cache
    start = time.perf_counter()
    result = predict_raw(text)
    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 2)  # Model Latency
    result["request_id"] = request_id
    result["cache"] = "MISS"
    CACHE_MISS.inc()

    # Store result in redis
    redis_client.setex(cache_key, 60, json.dumps(result))

    logger.info(
        "inference_completed",
        extra={
            "request_id": result["request_id"],
            "latency_ms": result["latency_ms"],
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "inp_length": len(text) if text else 0,
        },
    )

    return result
