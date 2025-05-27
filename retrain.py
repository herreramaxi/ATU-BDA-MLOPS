import os
import glob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump
import sys

FULL_DATA_PATH = "data/full/spam.csv"
NEW_DATA_DIR = "data/new-data/"
MERGED_DATA_PATH = FULL_DATA_PATH
MODEL_OUTPUT_PATH = "SpamModel.pkl"

def load_existing_data():
    if os.path.exists(FULL_DATA_PATH):
        return pd.read_csv(FULL_DATA_PATH)
    else:
        print("⚠️ No existing dataset found. Starting from scratch.")
        return pd.DataFrame(columns=["Category", "Message"])

def load_new_data():
    new_files = glob.glob(os.path.join(NEW_DATA_DIR, "*.csv"))
    if not new_files:
        return pd.DataFrame(columns=["Category", "Message"])
    
    df_list = [pd.read_csv(f) for f in new_files]
    return pd.concat(df_list, ignore_index=True)

def merge_data(df_full, df_new):
    combined = pd.concat([df_full, df_new], ignore_index=True)
    combined.drop_duplicates(subset=["Message"], inplace=True)
    return combined

def train_model(data):
    data["Spam"] = data["Category"].apply(lambda x: 1 if x.lower() == "spam" else 0)
    X = data["Message"]
    y = data["Spam"]

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(X, y)
    dump(model, MODEL_OUTPUT_PATH)
    print(f"Model trained and saved as {MODEL_OUTPUT_PATH}")

def main():
    df_new = load_new_data()

    if df_new.empty:
        print("No new data found. Exiting early.")
        sys.exit(100)  # Special exit code so GitHub can catch it

    df_full = load_existing_data()
    merged = merge_data(df_full, df_new)
    print(f"Merged dataset contains {len(merged)} messages.")

    os.makedirs(os.path.dirname(MERGED_DATA_PATH), exist_ok=True)
    merged.to_csv(MERGED_DATA_PATH, index=False)
    print(f"Updated dataset saved to {MERGED_DATA_PATH}")

    train_model(merged)

if __name__ == "__main__":
    main()
