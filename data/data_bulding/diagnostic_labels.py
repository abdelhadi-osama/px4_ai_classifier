import os
import glob
import pandas as pd

SOURCE_DIR = "./alfa_dataset/processed"

def determine_base_class(folder_name):
    name = folder_name.lower()
    fault_count = sum(1 for keyword in ["engine", "elevator", "rudder", "aileron"] if keyword in name)
    if fault_count > 1 or "__" in name: return 5, "Multi-Fault"
    if "engine" in name: return 1, "Engine"
    if "elevator" in name: return 2, "Elevator"
    if "rudder" in name: return 3, "Rudder"
    if "aileron" in name: return 4, "Aileron"
    if "no_failure" in name or "no_ground_truth" in name: return 0, "Normal"
    return -1, "UNKNOWN"

def run_diagnostics():
    flight_folders = sorted([f.path for f in os.scandir(SOURCE_DIR) if f.is_dir() and "txt" not in f.name])
    
    print("--- 1. GLOBAL FOLDER CLASSIFICATION TEST ---")
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, -1:0}
    for folder in flight_folders:
        flight_name = os.path.basename(folder)
        cls_id, cls_name = determine_base_class(flight_name)
        class_counts[cls_id] += 1
        if cls_id in [3, 5, -1]: # Highlight specific cases
            print(f"[{cls_id}] {cls_name:<12} : {flight_name}")
            
    print(f"\nExpected Flights per Class based on folder names: {class_counts}\n")
    
    print("--- 2. RAW FAILURE STATUS DATA CHECK ---")
    # We will check exactly what values exist in the raw failure_status files
    for folder in flight_folders:
        flight_name = os.path.basename(folder)
        cls_id, _ = determine_base_class(flight_name)
        
        # Only check failure flights
        if cls_id == 0: continue
            
        fail_files = glob.glob(os.path.join(folder, "*failure_status*.csv"))
        if not fail_files:
            print(f"[!] Class {cls_id} flight missing failure file: {flight_name}")
            continue
            
        fail_file = fail_files[0]
        try:
            df = pd.read_csv(fail_file)
            # Find the data column (not the timestamp)
            data_col = next((c for c in df.columns if "%time" not in c), None)
            if data_col:
                unique_vals = df[data_col].dropna().unique()
                
                # If it's class 3 (Rudder), highlight it
                if cls_id == 3:
                     print(f"👉 RUDDER TEST: {flight_name}")
                     print(f"   Column: {data_col} | Unique Values: {unique_vals}")
                elif 1.0 not in unique_vals and len(unique_vals) > 0:
                     # Alert if a file uses something other than 1.0
                     print(f"[?] WEIRD VALUES: {flight_name} -> {unique_vals}")
        except Exception as e:
            print(f"Error reading {fail_file}: {e}")

if __name__ == "__main__":
    run_diagnostics()
