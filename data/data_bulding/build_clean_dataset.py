import pandas as pd
import os
import glob

# The 29 sensors present in 100% of flights (excluding IMU which is master)
# We use exactly this list to ensure Neural Network feature uniformity.
COMMON_SENSORS = [
    "-diagnostics.csv",
    "-mavlink-from.csv",
    "-mavros-battery.csv",
    "-mavros-global_position-compass_hdg.csv",
    "-mavros-global_position-global.csv",
    "-mavros-global_position-local.csv",
    "-mavros-global_position-raw-fix.csv",
    "-mavros-global_position-raw-gps_vel.csv",
    "-mavros-global_position-rel_alt.csv",
    "-mavros-imu-atm_pressure.csv",
    "-mavros-imu-data_raw.csv",
    "-mavros-imu-mag.csv",
    "-mavros-imu-temperature.csv",
    "-mavros-local_position-odom.csv",
    "-mavros-local_position-pose.csv",
    "-mavros-local_position-velocity.csv",
    "-mavros-nav_info-airspeed.csv",
    "-mavros-nav_info-errors.csv",
    "-mavros-nav_info-pitch.csv",
    "-mavros-nav_info-roll.csv",
    "-mavros-nav_info-velocity.csv",
    "-mavros-nav_info-yaw.csv",
    "-mavros-rc-in.csv",
    "-mavros-rc-out.csv",
    "-mavros-setpoint_raw-target_global.csv",
    "-mavros-state.csv",
    "-mavros-time_reference.csv",
    "-mavros-vfr_hud.csv",
    "-mavros-wind_estimation.csv"
]

MASTER_SUFFIX = "-mavros-imu-data.csv"
TIME_COL = "%time"
SOURCE_DIR = "./alfa_dataset/processed"
OUTPUT_BASE_DIR = "./data/clean_data"

def process_flight(flight_path):
    flight_name = os.path.basename(flight_path)
    print(f"\n[🚀] Processing: {flight_name}")
    
    imu_file = os.path.join(flight_path, flight_name + MASTER_SUFFIX)
    if not os.path.exists(imu_file):
        print(f"  [❌] FATAL: Master IMU file not found for {flight_name}. Skipping.")
        return

    # 1. Load Master Clock (IMU)
    master_df = pd.read_csv(imu_file).sort_values(TIME_COL)
    master_df = master_df.rename(columns={c: f"imu.{c}" for c in master_df.columns if c != TIME_COL})
    
    # 2. Merge exactly the 100% common sensors
    for suffix in COMMON_SENSORS:
        sensor_file = os.path.join(flight_path, flight_name + suffix)
        if not os.path.exists(sensor_file):
            print(f"  [!] Missing highly expected sensor: {suffix}")
            continue
            
        prefix = suffix.strip("-").replace(".csv", "")
        try:
            # Using on_bad_lines='skip' because mavlink-from.csv has malformed rows in some flights
            other_df = pd.read_csv(sensor_file, on_bad_lines='skip').sort_values(TIME_COL)
            if other_df.empty:
                continue
                
            other_df = other_df.rename(columns={c: f"{prefix}.{c}" for c in other_df.columns if c != TIME_COL})
            master_df = pd.merge_asof(master_df, other_df, on=TIME_COL, direction='backward')
        except Exception as e:
            print(f"  [!] Failed to merge {suffix}: {e}")

    # 3. Handle Failure Status (The 'y' Label)
    # We dynamically look for any file containing 'failure_status'
    failure_files = glob.glob(os.path.join(flight_path, "*failure_status*.csv"))
    if failure_files:
        fail_file = failure_files[0]
        try:
            fail_df = pd.read_csv(fail_file).sort_values(TIME_COL)
            # Find the column that contains the actual 1.0/NaN data
            data_col = next((c for c in fail_df.columns if c != TIME_COL), None)
            if data_col:
                # Rename it so EVERY flight has the exact same label column name
                fail_df = fail_df[[TIME_COL, data_col]].rename(columns={data_col: "ground_truth_failure.data"})
                master_df = pd.merge_asof(master_df, fail_df, on=TIME_COL, direction='backward')
        except Exception as e:
            print(f"  [!] Failed to merge failure status: {e}")
    else:
        # If no failure file exists (no_failure flights), we still add the column filled with NaNs
        master_df["ground_truth_failure.data"] = pd.NA

    # 4. Final Cleanup (Forward fill missing values)
    master_df = master_df.ffill()
    
    # Save to the structure requested: data/clean_data/[flight_folder]/unified_asof_data.csv
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, flight_name), exist_ok=True)
    out_path = os.path.join(OUTPUT_BASE_DIR, flight_name, "unified_asof_data.csv")
    master_df.to_csv(out_path, index=False)
    print(f"  [✅] Saved: {len(master_df)} rows, {len(master_df.columns)} columns")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} not found.")
    else:
        flight_folders = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir() and "txt" not in f.name]
        for folder in sorted(flight_folders):
            process_flight(folder)
