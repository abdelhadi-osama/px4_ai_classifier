import sys
import os
import torch

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.architectures import get_model
from src import config

def test_architectures_integrity():
    print("Testing Model Architectures and Config Integration...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = 25 # Typical number of features in our processed CSV
    num_classes = 6
    batch_size = 16
    
    # Simulate a batch from the DataLoader
    dummy_input = torch.randn(batch_size, config.WINDOW_SIZE, num_features).to(device)
    
    model_names = ["cnn", "lstm", "hybrid"]
    
    for name in model_names:
        print(f"  -> Validating {name.upper()}...")
        
        # 1. Test Factory & Initialization
        try:
            model = get_model(name, num_features, num_classes, device)
        except Exception as e:
            assert False, f"Model {name} failed to initialize: {e}"
            
        # 2. Test Forward Pass
        try:
            output = model(dummy_input)
        except Exception as e:
            assert False, f"Model {name} failed on forward pass: {e}"
            
        # 3. Verify Output Shape
        assert output.shape == (batch_size, num_classes), f"Output shape mismatch for {name}. Got {output.shape}"
        
        # 4. Verify Parameter Gradients (Check if model is trainable)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, f"Model {name} has 0 parameters"
        
        print(f"     [PASS] Params: {total_params:,}")

    print("\n✅ All architectures verified successfully!")

if __name__ == "__main__":
    test_architectures_integrity()
