import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def train_model():
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real])
    data = data.sample(frac=1).reset_index(drop=True)

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer
