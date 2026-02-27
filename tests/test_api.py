from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "default_version" in data
    assert ("loaded versions" in data) or ("loaded_versions" in data)


def test_predict_default():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "I love this"})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "request_id" in data
        assert "cache" in data
        assert "requested_version" in data
        assert "served_version" in data
