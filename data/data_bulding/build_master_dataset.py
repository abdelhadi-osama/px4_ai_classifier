import pandas as pd
import os
import re

CLEAN_DATA_DIR = "./data/clean_data"
OUTPUT_FILE = "./data/MASTER_TRAINING_DATA.csv"

# Label Mapping
# 0: Normal
# 1: Engine
# 2: Elevator
# 3: Rudder
# 4: Aileron
# 5: Multi-Fault

def determine_base_class(folder_name):
    name = folder_name.lower()
    
    # Check for multi-fault first (contains '__' or multiple failure keywords)
    fault_count = sum(1 for keyword in ["engine", "elevator", "rudder", "aileron"] if keyword in name)
    if fault_count > 1 or "__" in name:
        return 5
        
    if "engine" in name:
        return 1
    elif "elevator" in name:
        return 2
    elif "rudder" in name:
        return 3
    elif "aileron" in name:
        return 4
    elif "no_failure" in name or "no_ground_truth" in name:
        return 0
    else:
        return 0

def filter_columns(df):
    """
    Applies the "Decision Filter" to drop useless math/string columns
    and keep only physical kinematics and commands.
    """
    cols_to_keep = ['%time']
    
    # Regex patterns for columns we WANT to keep
    keep_patterns = [
        r'imu.*(orientation|angular_velocity|linear_acceleration)\.[xyzw]$',
        r'twist\.twist\.(linear|angular)\.[xyz]$',
        r'rel_alt\.field\.data$',
        r'nav_info.*(commanded|measured)$',
        r'battery\.field\.(voltage|current|percentage)$',
        r'rc_out\.field\.channels[0-7]$'
    ]
    
    # Regex patterns for columns we explicitly WANT to drop
    # Even if they match a keep pattern, if they match drop, we drop them.
    drop_patterns = [
        r'covariance',
        r'header',
        r'ground_truth_failure' # We drop this after we use it for labeling
    ]
    
    for col in df.columns:
        # Check if it should be dropped
        should_drop = any(re.search(pat, col) for pat in drop_patterns)
        if should_drop:
            continue
            
        # Check if it should be kept
        should_keep = any(re.search(pat, col) for pat in keep_patterns)
        if should_keep:
            cols_to_keep.append(col)
            
    return df[cols_to_keep]

def build_master_dataset():
    all_flights_data = []
    flight_folders = sorted([f.path for f in os.scandir(CLEAN_DATA_DIR) if f.is_dir()])
    
    print("Building MASTER_TRAINING_DATA.csv...\n")
    
    for flight_id, folder in enumerate(flight_folders):
        flight_name = os.path.basename(folder)
        csv_path = os.path.join(folder, "unified_asof_data.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        # 1. Apply Labels (The "When Did It Break?" Problem)
        df["target_label"] = 0
        target_class = determine_base_class(flight_name)
        
        if target_class != 0 and "ground_truth_failure.data" in df.columns:
            # Mask where failure is active
            is_failing = df["ground_truth_failure.data"].fillna(0) > 0
            df.loc[is_failing, "target_label"] = target_class
            # Forward fill the label
            df["target_label"] = df["target_label"].replace(0, pd.NA).ffill().fillna(0).astype(int)
            
        # 2. Filter Columns (The "Useless Pages" Problem)
        filtered_df = filter_columns(df)
        
        # 3. Add Flight ID (For splitting later)
        filtered_df.insert(1, "flight_id", flight_id)
        
        # Add the target label back (since it was filtered out by regex)
        filtered_df["target_label"] = df["target_label"]
        
        all_flights_data.append(filtered_df)
        print(f"  [+] Added {flight_name} | Class: {target_class} | Shape: {filtered_df.shape}")

    # 4. Tape it all together
    master_df = pd.concat(all_flights_data, ignore_index=True)
    
    # Save
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n======================================")
    print(f"✅ SUCCESS! Master dataset created.")
    print(f"📍 Location: {OUTPUT_FILE}")
    print(f"📊 Total Rows: {len(master_df)}")
    print(f"🧮 Total Columns: {len(master_df.columns)}")
    print("======================================")
    
    print("\nLabel Distribution in Master Data:")
    print(master_df["target_label"].value_counts().sort_index())

if __name__ == "__main__":
    build_master_dataset()
