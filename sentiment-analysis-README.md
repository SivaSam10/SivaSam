# Sentiment Analysis on Movie Reviews

A text classification model that predicts whether a movie review is **positive** or **negative** using TF-IDF vectorization and Logistic Regression.

## Problem Statement

Understanding sentiment in text (reviews, tweets, feedback) is a core NLP task with applications in product feedback analysis, social media monitoring, and customer support triage. This project builds a simple, interpretable baseline for binary sentiment classification.

## Dataset

This repo ships with a small (50-sample) demo dataset of movie reviews (`reviews.csv`) so the pipeline runs out of the box with no external downloads.

**For real-world results, swap in a larger dataset** such as the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (50,000 reviews) — just keep the same column format (`review`, `sentiment`) and the script will work unchanged.

## Approach

1. **Text Vectorization** — Converted raw review text into numerical features using **TF-IDF** (Term Frequency–Inverse Document Frequency), removing English stop words and capping at 5,000 features.
2. **Train/Test Split** — 80/20 split, stratified on sentiment label.
3. **Model** — `LogisticRegression`, a strong, fast, and interpretable baseline for text classification.
4. **Evaluation** — Accuracy, precision, recall, F1-score, and confusion matrix.

## Results (on demo dataset)

| Metric | Value |
|---|---|
| Accuracy | 70% |

> Note: These results are based on the small 50-sample demo dataset included in this repo, meant to demonstrate the pipeline. Accuracy improves significantly with a larger, real-world dataset like IMDB (50K reviews).

## Tech Stack

- Python
- pandas
- scikit-learn

## How to Run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

## Future Improvements

- Swap in the full IMDB dataset (50K reviews) for realistic performance
- Compare Logistic Regression against Naive Bayes and SVM
- Try word embeddings (Word2Vec, GloVe) instead of TF-IDF
- Deploy as a simple Streamlit app for live review predictions
