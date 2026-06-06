import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src import config

class PX4FlightDataset(Dataset):
    """
    PyTorch Dataset for PX4 UAV Flight Logs using Rolling Windows.
    Transforms 2D tabular sequences into 3D tensors: [window_size, num_features]
    """
    def __init__(self, csv_file, window_size=config.WINDOW_SIZE, step_size=config.STEP_SIZE):
        print(f"Loading {csv_file} for Dataset creation...")
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.step_size = step_size
        
        # Determine exactly which columns are physical features (X)
        self.feature_cols = [col for col in self.df.columns if col not in config.COLUMNS_TO_DROP]
        
        # 1. Group by flight_id to solve the "Flight Boundary Problem"
        self.flight_groups = self.df.groupby('flight_id')
        
        self.X_windows = []
        self.y_labels = []
        
        self._build_windows()

    def _build_windows(self):
        """
        Iterates over each flight and slices the sequence into overlapping windows.
        Extracts features (X) and assigns the label of the final row (y).
        """
        for flight_id, group in self.flight_groups:
            # Extract features and labels as numpy arrays for speed
            features = group[self.feature_cols].values
            labels = group['target_label'].values
            
            num_rows = len(features)
            # If a flight is incredibly short, skip it (should be filtered out in EDA, but safety first)
            if num_rows < self.window_size:
                continue
                
            # Slide a window over the flight data
            for i in range(0, num_rows - self.window_size + 1, self.step_size):
                # X: The window of features. Shape: (window_size, num_features)
                window_x = features[i : i + self.window_size]
                
                # y: Solving the "Labeling the Window" Problem
                # We predict the label of the LAST row in the window
                window_y = labels[i + self.window_size - 1]
                
                self.X_windows.append(window_x)
                self.y_labels.append(window_y)
                
        # Convert the lists of windows into massive PyTorch Tensors
        self.X_windows = torch.tensor(np.array(self.X_windows), dtype=torch.float32)
        self.y_labels = torch.tensor(np.array(self.y_labels), dtype=torch.long)
        
        print(f"  -> Generated {len(self.X_windows)} windows. Tensor shape: {self.X_windows.shape}")

    def __len__(self):
        return len(self.y_labels)

    def __getitem__(self, idx):
        return self.X_windows[idx], self.y_labels[idx]

def get_dataloaders():
    """
    Helper function to generate the Train, Val, and Test DataLoaders using config settings.
    """
    print("--- Initializing DataLoaders ---")
    train_dataset = PX4FlightDataset(config.TRAIN_CSV)
    val_dataset   = PX4FlightDataset(config.VAL_CSV)
    test_dataset  = PX4FlightDataset(config.TEST_CSV)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=config.NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    num_features = len(train_dataset.feature_cols)
    return train_loader, val_loader, test_loader, num_features

if __name__ == "__main__":
    # Quick Diagnostic Test to verify shapes
    try:
        train_loader, val_loader, test_loader, num_features = get_dataloaders()
        for X_batch, y_batch in train_loader:
            print(f"\n[DIAGNOSTIC PASS]")
            print(f"Batch X Shape : {X_batch.shape} -> [batch_size, window_size, num_features]")
            print(f"Batch y Shape : {y_batch.shape} -> [batch_size]")
            print(f"Feature count : {num_features}")
            break
    except Exception as e:
        print(f"\n[DIAGNOSTIC FAIL] Error: {e}")
