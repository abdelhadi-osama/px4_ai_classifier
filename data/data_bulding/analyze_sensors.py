import re
from collections import defaultdict

def analyze_sensors(tree_file_path):
    flight_sensors = defaultdict(list)
    current_flight = None

    # Read the tree file
    with open(tree_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # Identify flight folders (they look like '├── carbonZ...' or '└── carbonZ...')
        match = re.search(r'(├── |└── )(carbonZ_[a-zA-Z0-9_-]+)$', line)
        if match:
            current_flight = match.group(2)
            continue
            
        # Identify CSV files within the flight folder
        if current_flight and '.csv' in line:
            # Extract the filename
            filename_match = re.search(r'([a-zA-Z0-9_-]+\.csv)$', line)
            if filename_match:
                filename = filename_match.group(1)
                
                # Remove the flight name prefix to get just the sensor suffix
                # E.g., 'carbonZ_..._engine_failure-mavros-battery.csv' -> '-mavros-battery.csv'
                sensor_suffix = filename.replace(current_flight, "")
                flight_sensors[current_flight].append(sensor_suffix)

    # Now, find the intersection
    total_flights = len(flight_sensors)
    print(f"Total flights found: {total_flights}\n")
    
    sensor_counts = defaultdict(int)
    for flight, sensors in flight_sensors.items():
        for sensor in sensors:
            sensor_counts[sensor] += 1

    common_sensors = []
    missing_sensors = []

    print("--- Sensors present in ALL flights ---")
    for sensor, count in sorted(sensor_counts.items()):
        if count == total_flights:
            common_sensors.append(sensor)
            print(f"[100%] {sensor}")
        else:
            missing_sensors.append((sensor, count))

    print(f"\nTotal Common Sensors: {len(common_sensors)}")

    print("\n--- Sensors missing from some flights ---")
    for sensor, count in sorted(missing_sensors, key=lambda x: x[1], reverse=True):
        print(f"[{count}/{total_flights}] {sensor}")

if __name__ == "__main__":
    analyze_sensors("data/tree_raw_data.txt")
