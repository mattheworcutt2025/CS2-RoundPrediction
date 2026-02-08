"""
CS2 Round Prediction - LSTM Deep Learning Model
Predicts round winner using sequential round data
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Paths
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")
LOG_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/logs")

# Hyperparameters
SEQUENCE_LENGTH = 5  # Use last 5 rounds as context
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CS2RoundLSTM(nn.Module):
    """LSTM model for CS2 round prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(CS2RoundLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take output from last timestep
        last_output = lstm_out[:, -1, :]
        
        # FC layers
        out = self.fc(last_output)
        return out

def create_sequences(X, y, seq_length):
    """Create sequences of rounds for LSTM input"""
    sequences = []
    labels = []
    
    for i in range(seq_length, len(X)):
        sequences.append(X[i-seq_length:i])
        labels.append(y[i])
    
    return np.array(sequences), np.array(labels)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    actuals = []
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend((outputs > 0.5).cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            
            total_loss += loss.item()
            predictions.extend((outputs > 0.5).cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    return total_loss / len(dataloader), accuracy, predictions, actuals

def main():
    print("=" * 60)
    print("CS2 Round Prediction - LSTM Deep Learning Model")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading preprocessed data...")
    X = np.load(DATA_PATH / "X_features.npy")
    y = np.load(DATA_PATH / "y_labels.npy")
    print(f"Loaded {len(y)} rounds with {X.shape[1]} features")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    print(f"Creating sequences (length={SEQUENCE_LENGTH})...")
    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    print(f"Sequences shape: {X_seq.shape}")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_size = X.shape[1]
    model = CS2RoundLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 20
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    model.load_state_dict(torch.load(MODEL_PATH / "best_model.pth"))
    test_loss, test_acc, test_preds, test_actuals = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_actuals, test_preds, target_names=['T Wins', 'CT Wins']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_actuals, test_preds)
    print(cm)
    
    # Save training history and results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(best_val_acc),
        'epochs_trained': len(history['train_loss']),
        'hyperparameters': {
            'sequence_length': SEQUENCE_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE
        },
        'classification_report': classification_report(test_actuals, test_preds, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open(LOG_PATH / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(LOG_PATH / "training_curves.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to {LOG_PATH}")
    print(f"Best model saved to {MODEL_PATH / 'best_model.pth'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
