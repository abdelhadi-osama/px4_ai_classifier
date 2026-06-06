import pandas as pd
import numpy as np

DATA_PATH = "./data/MASTER_TRAINING_DATA.csv"

def run_text_eda():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("\n--- 1. MISSING VALUES (NaNs) ---")
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    top_missing = missing_percentages.sort_values(ascending=False).head(10)
    print("Top columns with missing values (%):")
    print(top_missing)
    
    print("\n--- 2. TIMESTAMP GAPS (Dropped Packets) ---")
    df['time_diff'] = df.groupby('flight_id')['%time'].diff()
    time_diffs = df['time_diff'].dropna()
    print(f"Median Δt: {time_diffs.median():.4f} seconds (Expected ~0.01 for 100Hz IMU)")
    print(f"Max Δt (Longest Drop): {time_diffs.max():.4f} seconds")
    print(f"99th Percentile Δt: {np.percentile(time_diffs, 99):.4f} seconds")
    
    print("\n--- 3. FLIGHT DURATIONS (Timesteps per Flight) ---")
    flight_lengths = df.groupby('flight_id').size()
    print(f"Mean Duration: {flight_lengths.mean():.0f} rows")
    print(f"Min Duration:  {flight_lengths.min()} rows")
    print(f"Max Duration:  {flight_lengths.max()} rows")
    
    short_flights = flight_lengths[flight_lengths < 100]
    if not short_flights.empty:
        print(f"\nWarning: Found {len(short_flights)} flights with fewer than 100 rows:")
        print(short_flights)

if __name__ == "__main__":
    run_text_eda()
