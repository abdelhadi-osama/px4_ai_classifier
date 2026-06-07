
import torch
import pandas as pd
from src import config
from src.dataloader.dataset import get_dataloaders
from src.models.architectures import get_model
from sklearn.metrics import accuracy_score, f1_score
import os

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    _, _, test_loader, num_features = get_dataloaders()
    
    results = []
    models_to_test = ["cnn", "lstm", "hybrid"]
    
    for name in models_to_test:
        checkpoint_path = f"training_results/best_{name}.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {name}, checkpoint not found.")
            continue
            
        model = get_model(name, num_features, config.NUM_CLASSES, device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        results.append({"Model": name.upper(), "Accuracy": acc, "F1-Score": f1})
        
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("FINAL LOCAL EVALUATION RESULTS")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)
    
    best_model = df.loc[df['F1-Score'].idxmax()]
    print(f"\n🏆 THE WINNER IS: {best_model['Model']} with F1-Score: {best_model['F1-Score']:.4f}")

if __name__ == "__main__":
    evaluate()
