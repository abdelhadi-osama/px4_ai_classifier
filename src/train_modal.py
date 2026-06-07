import modal
import os
import time
from src import config

# 1. Define the Cloud Environment (The Image)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "joblib",
        "wandb"
    )
    .add_local_dir("./src", remote_path="/root/src")
    .add_local_dir("./data", remote_path="/root/data")
)

# 2. Create the Modal App and Volume for Persistent Storage
app = modal.App("px4-ai-classifier-train")
volume = modal.Volume.from_name("px4-results-vol", create_if_missing=True)

# 3. Define the Training Function on the Cloud
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600, # 1 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/root/results": volume} # Persistent storage mount
)
def train_cloud(model_name: str = "all"):
    """
    This function runs on the Modal cloud with robust error handling and persistence.
    """
    from src.train import run_training, plot_metrics, evaluate_on_test
    from src.dataloader.dataset import get_dataloaders
    import torch
    import shutil
    import pandas as pd
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cloud Training started on: {device}")

    # Set up data loaders
    train_loader, val_loader, test_loader, num_features = get_dataloaders()
    
    results = []
    model_names = ["cnn", "lstm", "hybrid"] if model_name == "all" else [model_name]
    
    best_overall_f1 = -1
    best_overall_model_name = ""

    for name in model_names:
        try:
            # 1. Run Training
            model, history = run_training(name, train_loader, val_loader, num_features, device)
            
            # 2. Generate and Save Plots
            plot_metrics(history, name)
            
            # 3. Load best weights and Evaluate on Test Set
            checkpoint_path = f"best_{name}.pt"
            model.load_state_dict(torch.load(checkpoint_path))
            
            f1, acc = evaluate_on_test(model, test_loader, device, name)
            results.append({"Model": name.upper(), "Test Accuracy": acc, "Test F1-Score": f1})
            
            # 4. Move Files to Persistent Volume
            # Move model weights
            shutil.copy(checkpoint_path, f"/root/results/best_{name}.pt")
            # Move plots (learning curves and confusion matrix)
            shutil.copy(f"{name}_learning_curves.png", f"/root/results/{name}_learning_curves.png")
            shutil.copy(f"{name}_cm.png", f"/root/results/{name}_cm.png")

            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_model_name = name
                
            print(f"✅ Finished {name.upper()} successfully.")

        except Exception as e:
            print(f"❌ ERROR training {name.upper()}: {str(e)}")
            continue

    # Final Summary Table
    if results:
        df_results = pd.DataFrame(results)
        print("\n" + "="*40)
        print("MODAL CLOUD TRAINING RESULTS")
        print("="*40)
        print(df_results.to_string(index=False))
        print("="*40)

        # Save the absolute winner to volume
        if best_overall_model_name:
            shutil.copy(f"/root/results/best_{best_overall_model_name}.pt", "/root/results/final_best_model.pt")
            print(f"🏆 Best model ({best_overall_model_name}) saved to volume.")
    
    # Commit changes to the volume
    volume.commit()
    return results

@app.local_entrypoint()
def main(model: str = "all", download: bool = True):
    """
    Local entry point: 
    'modal run src/train_modal.py --model all --download'
    """
    print(f"--- Launching Modal Cloud Training (Model: {model}) ---")
    results = train_cloud.remote(model)
    
    if download:
        print("\n--- Retrieving Results from Cloud ---")
        output_dir = "training_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # We can't directly 'return' large files easily, so we zip them in a small helper function
        # Or more simply, we use the local 'modal volume get' command or the App's capabilities.
        # For this setup, we will notify the user where to find them.
        print(f"✅ Cloud training finished. Files are stored in Modal Volume 'px4-results-vol'.")
        print(f"To download files to your local machine, run:")
        print(f"  modal volume get px4-results-vol / ./{output_dir}")
