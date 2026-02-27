# 🚀 Real-Time ML Inference System

Production-style ML inference prototype designed to demonstrate **model serving, versioning, rollout safety, low-latency inference, and observability**.

This project focuses on **ML systems thinking** rather than just model training.

---

## 🚀 Features

### 🔁 Model Versioning

* Supports multiple model versions (`v1`, `v2`)
* Default model version is configurable
* Explicit version override via query parameter (`?model_version=vX`)

### 🧠 Lazy Loading

* Models are loaded on demand if not already in memory
* Optimizes startup time and memory usage

### 🎯 Canary / A/B Routing

* Optional percentage-based traffic splitting between model versions
* Configurable via environment variables
* Safe fallback to default version if requested version is unavailable

### ⚡ Low-Latency Inference

* Redis-based response caching
* Version-aware cache keys to prevent cross-version contamination
* Latency measurement included in API response
* Cache HIT / MISS visibility

### 📊 Monitoring & Observability

* Structured logging
* Unique `request_id` per request
* Prometheus metrics exposed via `/metrics`
* Cache hit/miss counters
* Loaded model visibility endpoint (`/models`)

### 🐳 Containerized Architecture

* Separate containers for:

  * API + inference service
  * Redis cache
* Managed using Docker Compose

---

## 🏗 Architecture Overview

```
Client
  ↓
FastAPI (HTTP Layer)
  ↓
Inference Service (Routing + Caching + Canary Logic)
  ↓
Model Loader (Artifact Management)
  ↓
Redis (External Cache)
```

---

### Key Design Decisions

* **Version-aware cache keys** prevent mixing predictions from different model versions.
* **Separation of concerns**:

  * `api/` → HTTP layer
  * `services/` → orchestration logic
  * `model/` → artifact loading and prediction
  * `infra/` → deployment configuration
* **Safe fallback strategy** ensures system stability if a requested model version is unavailable.

---

## 📂 Project Structure

```
api/            FastAPI endpoints + schemas
services/       Inference logic (routing, caching, latency measurement)
model/          Model artifacts and loader (v1/, v2/)
infra/          Dockerfile + docker-compose.yml
monitoring/     Monitoring and Logging
tests/          Test cases
requirements.txt
README.md
```

---

## 🐳 Running the System (Docker)

### Prerequisites

* Docker Desktop installed

### Run Everything

```bash
docker compose -f infra/docker-compose.yml up --build
```

Once running, access:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check → [http://localhost:8000/health](http://localhost:8000/health)
* Metrics → [http://localhost:8000/metrics](http://localhost:8000/metrics)
* Loaded Models → [http://localhost:8000/models](http://localhost:8000/models)

---

## 📡 API Usage

### Predict (Default Model)

**Endpoint**

```
POST /predict
```

**Body**

```json
{
  "text": "I love this system"
}
```

---

### Predict (Explicit Model Version)

```
POST /predict?model_version=v1
```

---

## ⚙ Configuration (Environment Variables)

Configured via Docker Compose.

| Variable                | Purpose                                   |
| ----------------------- | ----------------------------------------- |
| `DEFAULT_MODEL_VERSION` | Default model used if none specified      |
| `LOAD_ALL_MODELS`       | Load all versions at startup (true/false) |
| `CANARY_VERSION`        | Model version used for canary rollout     |
| `CANARY_PERCENT`        | % of traffic routed to canary version     |

---

## 📊 Observability

The system exposes:

* `request_id` for traceability
* Cache HIT / MISS indicator
* Latency measurement (ms)

### Prometheus Metrics

* `cache_hits_total`
* `cache_miss_total`
* `loaded_model_versions`

---

## 🧠 System Design Notes

* **Horizontal scaling:** Multiple API replicas can share Redis.
* **Safe model rollout:** Canary percentage enables gradual deployment of new versions.
* **Lazy loading:** Supports multiple model versions without increasing startup time.
* **Failure handling:** Falls back to default version if requested version is unavailable.
* **Extensibility:** Designed to allow CI/CD integration and future model registry integration.
