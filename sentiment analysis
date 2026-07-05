"""
Sentiment Analysis on Movie Reviews
Classifies text reviews as Positive or Negative using TF-IDF + Logistic Regression.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
# Replace this with a larger dataset (e.g. IMDB Reviews from Kaggle) for production use.
# Expected CSV format: columns "review" (text) and "sentiment" (positive/negative)
data = pd.read_csv("reviews.csv")
print(data.head())

# -----------------------------
# Features and Target
# -----------------------------
X = data["review"]
y = data["sentiment"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Try it on a custom review
# -----------------------------
sample_review = ["This movie was absolutely wonderful, I loved every minute of it!"]
sample_tfidf = vectorizer.transform(sample_review)
print("\nSample prediction:", model.predict(sample_tfidf)[0])
