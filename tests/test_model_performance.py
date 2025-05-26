# tests/test_model_performance.py

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "SpamModel.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "spam.csv")

def test_model_accuracy_threshold():
    data = pd.read_csv(DATA_PATH)
    # your test code here


def test_model_accuracy_threshold():
    # Load dataset
    data = pd.read_csv(DATA_PATH)
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Train/test split (match your training logic)
    X_train, X_test, y_train, y_test = train_test_split(
        data['Message'], data['Spam'], test_size=0.25, random_state=42
    )

    # Load trained model
    model = load(MODEL_PATH)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Assert that model is good enough
    assert accuracy >= 0.80, f"Model accuracy too low: {accuracy:.4f}"
