import os
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "SpamModel.pkl")
DATA_PATH =  os.getenv("DATASET_PATH", 'spam.csv')

# Load and prepare data
def load_data():
    print(f"Loading dataset from: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    
    return train_test_split(data['Message'], data['Spam'], test_size=0.25, random_state=42)

# Load model
def load_model():
    return load(MODEL_PATH)

def test_model_accuracy_threshold():
    X_train, X_test, y_train, y_test = load_data()
    model = load_model()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy:.4f}")
    assert accuracy >= 0.80, f"❌ Model accuracy too low: {accuracy:.4f}"

def test_model_f1_score_threshold():
    X_train, X_test, y_train, y_test = load_data()
    model = load_model()
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ Model F1 score: {f1:.4f}")
    assert f1 >= 0.70, f"❌ Model F1 score too low: {f1:.4f}"