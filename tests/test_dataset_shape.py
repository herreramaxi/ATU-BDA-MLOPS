import os
import pandas as pd

# Load and prepare data
def load_data():
    dataset_path = os.getenv("DATASET_PATH", 'spam.csv')

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    return df

def test_dataset_shape():
    df = load_data()
    assert 'Category' in df.columns and 'Message' in df.columns, "❌ Dataset must contain 'Category' and 'Message' columns."


def test_dataset_is_not_empty():
    df = load_data()
    
    assert not df.empty, "❌ Dataset is empty. Please check the data source."