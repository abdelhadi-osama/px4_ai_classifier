import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

MASTER_FILE = "./data/MASTER_TRAINING_DATA.csv"
OUTPUT_DIR = "./data"

# The 6 redundant IMU columns found during correlation analysis
COLS_TO_DROP = [
    "mavros-imu-data_raw.field.angular_velocity.x",
    "mavros-imu-data_raw.field.angular_velocity.y",
    "mavros-imu-data_raw.field.angular_velocity.z",
    "mavros-imu-data_raw.field.linear_acceleration.x",
    "mavros-imu-data_raw.field.linear_acceleration.y",
    "mavros-imu-data_raw.field.linear_acceleration.z"
]

def preprocess_and_split():
    print(f"Loading {MASTER_FILE}...")
    df = pd.read_csv(MASTER_FILE)
    
    # 1. Drop redundant features
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors='ignore')
    
    # 2. Patch NaNs (Forward fill then backward fill to ensure no gaps)
    print("Patching NaN values...")
    # The safest way to do this in pandas without losing columns is to assign the filled values back
    for col in df.columns:
        if col not in ['%time', 'flight_id', 'target_label']:
            df[col] = df.groupby('flight_id')[col].transform(lambda x: x.ffill().bfill())
    
    # 3. Stratified Flight-Level Splitting
    print("Performing Stratified Flight-Level Split...")
    
    # Determine the 'Primary Class' of each flight (the maximum label it reaches)
    flight_classes = df.groupby('flight_id')['target_label'].max().to_dict()
    
    # Group flights by their primary class
    class_to_flights = defaultdict(list)
    for f_id, cls in flight_classes.items():
        class_to_flights[cls].append(f_id)
        
    train_flights = []
    val_flights = []
    test_flights = []
    
    # Stratified split inside each class bucket
    np.random.seed(42) # For reproducibility
    for cls, flights in class_to_flights.items():
        np.random.shuffle(flights)
        n = len(flights)
        
        # If there's only 1 flight, it MUST go to train so the model learns it
        if n == 1:
            train_flights.append(flights[0])
            continue
            
        # If there are 2 flights, put 1 in train, 1 in test
        if n == 2:
            train_flights.append(flights[0])
            test_flights.append(flights[1])
            continue
            
        train_end = int(n * 0.70)
        val_end = train_end + max(1, int(n * 0.15)) # Ensure val gets at least 1 if n >= 3
        
        train_flights.extend(flights[:train_end])
        val_flights.extend(flights[train_end:val_end])
        test_flights.extend(flights[val_end:])
        
    print(f"  -> Train flights: {len(train_flights)}")
    print(f"  -> Val flights:   {len(val_flights)}")
    print(f"  -> Test flights:  {len(test_flights)}")
    
    # Extract DataFrames
    train_df = df[df['flight_id'].isin(train_flights)].copy()
    val_df = df[df['flight_id'].isin(val_flights)].copy()
    test_df = df[df['flight_id'].isin(test_flights)].copy()
    
    # 4. Feature Scaling (Fit ONLY on Train!)
    print("Applying StandardScaler...")
    feature_cols = [c for c in df.columns if c not in ['%time', 'flight_id', 'target_label']]
    
    scaler = StandardScaler()
    
    # Fit and transform training features
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # Only transform validation and test features
    if not val_df.empty:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    if not test_df.empty:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 5. Save everything
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_scaled.csv"), index=False)
    if not val_df.empty:
        val_df.to_csv(os.path.join(OUTPUT_DIR, "val_scaled.csv"), index=False)
    if not test_df.empty:
        test_df.to_csv(os.path.join(OUTPUT_DIR, "test_scaled.csv"), index=False)
        
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    
    print("\n✅ Preprocessing & Splitting Complete!")
    print(f"Train Rows: {len(train_df)} | Target Classes present: {sorted(train_df['target_label'].unique())}")
    print(f"Val Rows:   {len(val_df)} | Target Classes present: {sorted(val_df['target_label'].unique())}")
    print(f"Test Rows:  {len(test_df)} | Target Classes present: {sorted(test_df['target_label'].unique())}")

if __name__ == "__main__":
    preprocess_and_split()
