import pandas as pd
import numpy as np

DATA_PATH = "./data/MASTER_TRAINING_DATA.csv"

def run_correlation_analysis():
    print(f"Loading {DATA_PATH} for Correlation Analysis...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop non-feature columns for this analysis
    cols_to_drop = ['%time', 'flight_id', 'target_label']
    feature_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    corr_matrix = feature_df.corr(numeric_only=True)
    
    # Extract upper triangle of the correlation matrix to avoid duplicates (A-B and B-A)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with high correlation (absolute value > 0.95)
    high_corr_threshold = 0.95
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > high_corr_threshold)]
    
    print(f"\n--- Highly Correlated Pairs (|r| > {high_corr_threshold}) ---")
    found_any = False
    for col in upper_tri.columns:
        for row in upper_tri.index:
            val = upper_tri.loc[row, col]
            if pd.notna(val) and abs(val) > high_corr_threshold:
                print(f"[{val:+.4f}] {row} <--> {col}")
                found_any = True
                
    if not found_any:
         print("No pairs found above the threshold.")
         
    print(f"\nPotential features to drop due to redundancy ({len(to_drop)}):")
    for feat in to_drop:
        print(f" - {feat}")

if __name__ == "__main__":
    run_correlation_analysis()
