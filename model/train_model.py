from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Dummy training data
texts = [
    "I love this product",
    "This is amazing",
    "Very happy with the service",
    "I hate this",
    "This is terrible",
    "Very bad experience",
]

labels = [1, 1, 1, 0, 0, 0]

# Create Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training completed. model saved")
