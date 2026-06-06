# PX4 AI Classifier

A Deep Learning-based MVP system that automatically detects and classifies mechanical failures by analyzing flight logs from fixed-wing UAVs.

## Project Architecture
This project focuses on identifying 6 different flight states (Normal + 5 Fault Types) using PyTorch, processing multi-sensor temporal data (IMU, GPS, Actuators).

The data pipeline is modularized inside the `data/` directory:
*   `data/data_bulding/`: Contains scripts to aggregate and clean the raw drone logs (e.g., `merge_asof`).
*   `data/preprossing/`: Contains scripts for correlation analysis, missing value patching, and Stratified Group splitting.
*   `data/EDA/`: Contains Exploratory Data Analysis (EDA) Jupyter notebooks and diagnostic scripts.
*   `tests/`: Verification scripts to ensure pipeline integrity.
*   `data/`: The root of the data folder houses the final processed `_scaled.csv` files ready for training. *(Note: Raw data is ignored in version control to save space).*

## Labeling Map
*   `0`: Normal / No Failure
*   `1`: Engine Failure
*   `2`: Elevator Failure
*   `3`: Rudder Failure
*   `4`: Aileron Failure
*   `5`: Multi-Fault

## How to Run the Data Pipeline
If you have downloaded the raw dataset (`alfa_dataset`), you can regenerate the exact Train/Val/Test splits by running these three scripts in order:

1.  **Uniformity & Synchronization (merge_asof):**
    This unifies disparate sensors (GPS, Actuators) onto the 100Hz IMU timeline.
    ```bash
    python data/data_bulding/build_clean_dataset.py
    ```

2.  **Master Dataset Consolidation:**
    Applies the temporal fault labels based on the folder names and concatenates all 47 flights into one dataset.
    ```bash
    python data/data_bulding/build_master_dataset.py
    ```

3.  **Preprocessing & Stratified Splitting:**
    Fills missing values, applies Stratified Group Splitting (by `flight_id`) to ensure no temporal leakage and balanced classes, and fits the `StandardScaler`.
    ```bash
    python data/preprossing/preprocess_and_split.py
    ```

## Running Tests
To verify that the dataset was generated flawlessly:
```bash
python tests/test_pipeline.py
```
