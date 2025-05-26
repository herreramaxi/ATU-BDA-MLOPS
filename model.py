import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
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

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    model.fit(X_train, y_train)
    dump(model, "SpamModel.pkl")
    print("Model trained and saved to SpamModel.pkl")

if __name__ == "__main__":
    train_model()
