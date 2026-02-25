import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorizer = None
model = None


def load_artifacts():
    global vectorizer, model
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, "model.pkl")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("artifacts have been loaded")


def predict_raw(text: str):
    if vectorizer is None or model is None:
        raise Exception("Artifacts are missing")

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    label = "Positive" if int(prediction) == 1 else "Negative"

    return {"prediction": label, "confidence": float(probability)}
