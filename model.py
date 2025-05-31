import os
import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

def load_data():
    dataset_path = os.getenv("DATASET_PATH", 'spam.csv')#"gs://spam-data-pipeline/full/spam.csv")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    return df['Message'], df['Spam']

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    classes = ['ham', 'spam']

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    dump(model, "SpamModel.pkl")
    print("Model trained and saved to SpamModel.pkl")

if __name__ == "__main__":
    train_model()
