import sys
import os
import torch

# Ensure the root directory is in the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataloader.dataset import get_dataloaders
from src import config

def test_dataloader_pipeline():
    print("Testing PyTorch DataLoader Pipeline...")
    
    # 1. Check if config is accessible
    assert config.WINDOW_SIZE > 0, "Config WINDOW_SIZE is invalid"
    assert config.BATCH_SIZE > 0, "Config BATCH_SIZE is invalid"
    
    # 2. Attempt to initialize DataLoaders
    try:
        train_loader, val_loader, test_loader, num_features = get_dataloaders()
    except Exception as e:
        assert False, f"Failed to initialize DataLoaders: {e}"
        
    # 3. Check Train Loader Shapes
    for X_batch, y_batch in train_loader:
        assert X_batch.shape[0] <= config.BATCH_SIZE, "Batch size mismatch"
        assert X_batch.shape[1] == config.WINDOW_SIZE, "Window size mismatch"
        assert X_batch.shape[2] == num_features, "Feature count mismatch"
        
        assert y_batch.shape[0] == X_batch.shape[0], "Label batch size mismatch"
        
        # Verify labels are within expected bounds (0 to 5)
        assert torch.all(y_batch >= 0) and torch.all(y_batch < config.NUM_CLASSES), "Invalid target label found"
        break # Only test the first batch to save time
        
    print("✅ DataLoader tests passed successfully!")
    print(f"Verified Tensor Shape: {X_batch.shape} -> [batch_size, window_size, features]")

if __name__ == "__main__":
    test_dataloader_pipeline()
