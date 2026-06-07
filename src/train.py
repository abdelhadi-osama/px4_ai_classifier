import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import time
import argparse
import shutil
import wandb

from src import config
from src.dataloader.dataset import get_dataloaders
from src.models.architectures import get_model

def calculate_class_weights(train_loader):
    """
    Calculates weights to handle imbalanced failure classes.
    Makes the model 'care' more about rare failures.
    """
    print("Calculating class weights for imbalanced handling...")
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    
    counts = np.bincount(all_labels, minlength=config.NUM_CLASSES)
    # Inverse frequency weighting
    weights = len(all_labels) / (config.NUM_CLASSES * (counts + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    return running_loss / total, correct / total

def run_training(model_name, train_loader, val_loader, num_features, device):
    print(f"\n>>> Starting Training for: {model_name.upper()} <<<")
    
    # Initialize WandB for this specific model run
    wandb.init(
        project=config.WANDB_PROJECT,
        name=f"train_{model_name}_{int(time.time())}",
        config={
            "model_type": model_name,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "window_size": config.WINDOW_SIZE
        }
    )

    model = get_model(model_name, num_features, config.NUM_CLASSES, device)
    
    # Loss & Optimizer
    weights = calculate_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    checkpoint_path = f"best_{model_name}.pt"
    
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    wandb.finish()
    return model, history

def plot_metrics(history, model_name):
    """Generates plots to check for overfitting."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_learning_curves.png")
    # plt.show()  # Disabled for non-interactive environments

def evaluate_on_test(model, test_loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # Metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Use explicit labels to handle cases where some classes might be missing from the test set
    target_labels = sorted(list(config.CLASS_NAMES.keys()))
    target_names = [config.CLASS_NAMES[i] for i in target_labels]
    
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=target_labels,
        target_names=target_names, 
        output_dict=True
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES.values(), 
                yticklabels=config.CLASS_NAMES.values())
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{model_name}_cm.png")
    # plt.show()
    
    return f1, report['accuracy']

def main():
    parser = argparse.ArgumentParser(description="Train PX4 AI Classifier Models")
    parser.add_argument("--model", type=str, default="all", choices=["cnn", "lstm", "hybrid", "all"], 
                        help="Which model to train (default: all)")
    parser.add_argument("--download", action="store_true", help="Download results if in Colab")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    train_loader, val_loader, test_loader, num_features = get_dataloaders()
    
    results = []
    model_names = ["cnn", "lstm", "hybrid"] if args.model == "all" else [args.model]
    
    best_overall_f1 = -1
    best_overall_model_name = ""

    for name in model_names:
        model, history = run_training(name, train_loader, val_loader, num_features, device)
        plot_metrics(history, name)
        
        # Load best weights before test evaluation
        model.load_state_dict(torch.load(f"best_{name}.pt"))
        f1, acc = evaluate_on_test(model, test_loader, device, name)
        results.append({"Model": name.upper(), "Test Accuracy": acc, "Test F1-Score": f1})
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model_name = name

    # Final Comparison Table
    df_results = pd.DataFrame(results)
    print("\n" + "="*40)
    print("FINAL MODEL COMPARISON")
    print("="*40)
    print(df_results.to_string(index=False))
    print("="*40)
    
    # Save the absolute best model
    if best_overall_model_name:
        winner_src = f"best_{best_overall_model_name}.pt"
        winner_dst = "final_best_model.pt"
        shutil.copy(winner_src, winner_dst)
        print(f"\n🏆 WINNER: {best_overall_model_name.upper()} saved as {winner_dst}")

    # Automated Colab Download
    if args.download:
        try:
            from google.colab import files
            print("Zipping and downloading results...")
            os.system("zip -r training_results.zip best_*.pt final_best_model.pt *.png")
            files.download("training_results.zip")
        except ImportError:
            print("Not in Colab. Skipping download.")

if __name__ == "__main__":
    main()
