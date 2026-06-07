"""
Configuration File for PX4 AI Classifier
----------------------------------------
This file contains all the adjustable hyperparameters and paths for the project.
Modify these values to experiment with different model architectures or data slicing strategies.
"""

import os

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# The root directory where the scaled data is saved
DATA_DIR = "./data"

# Exact paths to the scaled Train, Val, and Test CSVs generated in Phase 3
TRAIN_CSV = os.path.join(DATA_DIR, "train_scaled.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_scaled.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_scaled.csv")

# Path to the saved StandardScaler (used for inference in Streamlit)
SCALER_PATH = os.path.join(DATA_DIR, "feature_scaler.pkl")


# ==========================================
# 2. DATA LOADER & ROLLING WINDOW PARAMETERS
# ==========================================
# WINDOW_SIZE: The number of sequential rows (timesteps) the model looks at to make 1 prediction.
# At 100Hz, a window of 50 equals 0.5 seconds of flight data.
WINDOW_SIZE = 50

# STEP_SIZE: How many rows to slide the window forward.
# E.g., if step=10, Window 1 is rows 0-49, Window 2 is rows 10-59.
# Smaller step size = more overlapping training data, but higher RAM usage.
STEP_SIZE = 10

# BATCH_SIZE: How many windows to feed into the Neural Network at once during training.
# Standard values: 32, 64, 128. Reduce this if you get Out Of Memory (OOM) errors.
BATCH_SIZE = 64

# NUM_WORKERS: How many CPU threads to use for loading data in PyTorch.
# Set to 0 if you encounter multi-processing errors on Windows/Mac.
NUM_WORKERS = 4


# ==========================================
# 3. COLUMNS TO IGNORE DURING TRAINING
# ==========================================
# These columns exist in the CSV for structural reasons (like grouping flights),
# but MUST NOT be fed into the Neural Network as features (X).
COLUMNS_TO_DROP = ['%time', 'flight_id', 'target_label']


# ==========================================
# 4. LABEL MAPPING (For Reference & Streamlit)
# ==========================================
CLASS_NAMES = {
    0: "Normal",
    1: "Engine Failure",
    2: "Elevator Failure",
    3: "Rudder Failure",
    4: "Aileron Failure",
    5: "Multi-Fault"
}
NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# 5. MODEL HYPERPARAMETERS (Phase 5)
# ==========================================
# Common
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# CNN Specific
CNN_FILTERS_1 = 32
CNN_FILTERS_2 = 64
CNN_DROPOUT   = 0.5

# LSTM Specific
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3 # Dropout between LSTM layers
FC_DROPOUT       = 0.5 # Dropout in final fully connected layer

# Hybrid Specific
HYBRID_CNN_FILTERS = 32
HYBRID_LSTM_HIDDEN = 64
HYBRID_DROPOUT     = 0.5

# ==========================================
# 6. EXPERIMENT TRACKING (WandB)
# ==========================================
WANDB_PROJECT = "px4-ai-classifier"
