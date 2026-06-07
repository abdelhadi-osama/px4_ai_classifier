import torch
import torch.nn as nn
from src import config

class CNN1DModel(nn.Module):
    """
    1D-CNN (The 'Pattern Detector')
    Scientific Consensus: Best for high-frequency vibration and sharp spike detection in sensor data.
    Input Shape: [Batch, Seq_Len (50), Features]
    Internal Transpose: [Batch, Features, Seq_Len (50)]
    """
    def __init__(self, num_features, num_classes):
        super(CNN1DModel, self).__init__()
        
        # Block 1: Feature Extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=config.CNN_FILTERS_1, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.CNN_FILTERS_1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 50 -> 25
        )
        
        # Block 2: Deep Patterns
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=config.CNN_FILTERS_1, out_channels=config.CNN_FILTERS_2, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.CNN_FILTERS_2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 25 -> 12
        )
        
        # Global Average Pooling (Pro move: Location invariance + Overfitting prevention)
        self.gap = nn.AdaptiveAvgPool1d(1) # [Batch, config.CNN_FILTERS_2, 1]
        
        self.classifier = nn.Sequential(
            nn.Linear(config.CNN_FILTERS_2, 64),
            nn.ReLU(),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Seq, Features] -> Transpose to [Batch, Features, Seq]
        x = x.transpose(1, 2)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        x = self.gap(x) # [Batch, Filters, 1]
        x = x.view(x.size(0), -1) # Flatten
        
        logits = self.classifier(x)
        return logits


class LSTMModel(nn.Module):
    """
    LSTM (The 'Memory' Model)
    Scientific Consensus: Best for long-term temporal context and drift detection.
    Input Shape: [Batch, Seq_Len (50), Features]
    """
    def __init__(self, num_features, num_classes):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # LSTM output: [Batch, Seq, Hidden]
        # We take only the last hidden state: out[:, -1, :]
        out, (hn, cn) = self.lstm(x)
        last_hidden = out[:, -1, :]
        
        logits = self.classifier(last_hidden)
        return logits


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM (The 'Pro' Move)
    Scientific Consensus: CNN extracts spatial features (spikes), LSTM reasons about them over time.
    """
    def __init__(self, num_features, num_classes):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN acts as a denoising front-end
        self.cnn_front = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=config.HYBRID_CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.HYBRID_CNN_FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # 50 -> 25
        )
        
        # LSTM processes the high-level features extracted by CNN
        self.lstm = nn.LSTM(
            input_size=config.HYBRID_CNN_FILTERS, 
            hidden_size=config.HYBRID_LSTM_HIDDEN,
            num_layers=1,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.HYBRID_LSTM_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(config.HYBRID_DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 1. CNN (needs channels first)
        # x: [Batch, 50, Features] -> [Batch, Features, 50]
        x = x.transpose(1, 2)
        x = self.cnn_front(x) # [Batch, Filters, 25]
        
        # 2. Preparation for LSTM (needs [Batch, Seq, Features])
        # [Batch, Filters, 25] -> [Batch, 25, Filters]
        x = x.transpose(1, 2)
        
        # 3. LSTM
        out, (hn, cn) = self.lstm(x)
        last_hidden = out[:, -1, :]
        
        # 4. Final Classification
        logits = self.classifier(last_hidden)
        return logits


def get_model(model_name, num_features, num_classes, device="cpu"):
    """
    Factory function to easily instantiate and move models to GPU.
    """
    if model_name.lower() == "cnn":
        model = CNN1DModel(num_features, num_classes)
    elif model_name.lower() == "lstm":
        model = LSTMModel(num_features, num_classes)
    elif model_name.lower() == "hybrid":
        model = HybridCNNLSTM(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return model.to(device)


if __name__ == "__main__":
    # Diagnostic Test & Debugging Info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_feat = 20
    num_cls = config.NUM_CLASSES
    dummy_input = torch.randn(8, config.WINDOW_SIZE, num_feat).to(device)
    
    models = ["cnn", "lstm", "hybrid"]
    
    for m_name in models:
        print(f"\n--- Testing {m_name.upper()} Architecture ---")
        try:
            model = get_model(m_name, num_feat, num_cls, device)
            output = model(dummy_input)
            print(f"Success! Input: {dummy_input.shape} -> Output: {output.shape}")
            
            # Print parameter count for complexity analysis
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total Parameters: {total_params:,}")
        except Exception as e:
            print(f"FAILED: {e}")
