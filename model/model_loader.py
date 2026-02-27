import pickle
import os
from prometheus_client import Gauge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# vectorizer = None
# model = None
artifacts = {}

loaded_models = Gauge(
    "loaded model versions", "Number of model versions currently loaded"
)


def load_artifacts(version: str):
    global artifacts
    # default_version = "v1"
    vectorizer_path = os.path.join(BASE_DIR, version, "vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, version, "model.pkl")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    artifacts[version] = (vectorizer, model)
    loaded_models.set(len(artifacts))
    print("artifacts have been loaded")


# Load all artifacts at once
def load_all_artifacts():
    for version in os.listdir(BASE_DIR):
        version_dir = os.path.join(BASE_DIR, version)
        if not os.path.isdir(version_dir):
            continue
        vectorizer_path = os.path.join(version_dir, "vectorizer.pkl")
        model_path = os.path.join(version_dir, "model.pkl")
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            load_artifacts(version)


def predict_raw(text: str, version: str):
    if version not in artifacts:
        try:
            load_artifacts(version)
        except Exception:
            raise ValueError(f"Model Version {version} does not exist")
    vectorizer, model = artifacts[version]
    if vectorizer is None or model is None:
        raise Exception("Artifacts are missing")

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    label = "Positive" if int(prediction) == 1 else "Negative"

    return {"prediction": label, "confidence": float(probability)}


def get_loaded_versions():
    return sorted(artifacts.keys())
