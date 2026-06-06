import os
import pandas as pd
import numpy as np

DATA_DIR = "./data"
REQUIRED_FILES = [
    "train_scaled.csv",
    "val_scaled.csv",
    "test_scaled.csv",
    "feature_scaler.pkl"
]

def test_data_pipeline_outputs():
    # 1. Check if all required files exist
    for file in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, file)
        assert os.path.exists(path), f"Missing required file: {file}"

    # 2. Load the train and test data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_scaled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_scaled.csv"))

    # 3. Check for NaNs
    assert train_df.isnull().sum().sum() == 0, "Found NaNs in train_scaled.csv"
    assert test_df.isnull().sum().sum() == 0, "Found NaNs in test_scaled.csv"

    # 4. Check for correct target labels (0 to 5)
    train_labels = train_df['target_label'].unique()
    assert set(train_labels).issubset({0, 1, 2, 3, 4, 5}), f"Unexpected labels found: {train_labels}"
    
    # 5. Check structural columns
    assert 'flight_id' in train_df.columns, "Missing flight_id column"
    assert 'target_label' in train_df.columns, "Missing target_label column"
    
    # Ensure %time is NOT dropped yet, we preserve it for flexibility
    assert '%time' in train_df.columns, "Missing %time column"

    print("✅ All data pipeline tests passed successfully!")

if __name__ == "__main__":
    test_data_pipeline_outputs()
