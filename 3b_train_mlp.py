"""
CS2 Round Prediction - MLP Deep Learning Model
Trains on snapshot data with ~100 features
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Paths
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")
LOG_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/logs")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 200
EARLY_STOP_PATIENCE = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CS2RoundMLP(nn.Module):
    """MLP model for CS2 round prediction"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super(CS2RoundMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CS2RoundMLPLarge(nn.Module):
    """Larger MLP with residual connections"""
    
    def __init__(self, input_dim, dropout=0.3):
        super(CS2RoundMLPLarge, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, 512)
        
        self.block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        
        self.downsample1 = nn.Linear(512, 256)
        self.downsample2 = nn.Linear(256, 128)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # Block 1 with residual
        identity = x
        x = self.block1(x)
        x = self.relu(x + identity)
        
        # Block 2 with residual
        identity = self.downsample1(x)
        x = self.block2(x)
        x = self.relu(x + identity)
        
        # Block 3 with residual
        identity = self.downsample2(x)
        x = self.block3(x)
        x = self.relu(x + identity)
        
        x = self.classifier(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y.float())
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        probs = outputs.detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss / len(dataloader), accuracy, auc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            
            total_loss += loss.item()
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss / len(dataloader), accuracy, auc, all_preds, all_labels, all_probs

def main():
    print("=" * 60)
    print("CS2 Round Prediction - MLP Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading snapshot data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")
    
    with open(DATA_PATH / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    
    print(f"Loaded {len(y)} snapshots with {X.shape[1]} features")
    print(f"CT wins: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"T wins: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler params for inference
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    with open(MODEL_PATH / "scaler_params.json", 'w') as f:
        json.dump(scaler_params, f)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Try both models
    models_to_try = [
        ("MLP_Standard", CS2RoundMLP(X.shape[1], hidden_dims=[512, 256, 128, 64], dropout=0.3)),
        ("MLP_Large", CS2RoundMLPLarge(X.shape[1], dropout=0.3)),
    ]
    
    best_overall = {'accuracy': 0, 'model_name': None}
    
    for model_name, model in models_to_try:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        model = model.to(DEVICE)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }
        
        best_val_acc = 0
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc, train_auc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc, val_auc, _, _, _ = evaluate(
                model, val_loader, criterion, DEVICE
            )
            
            scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                torch.save(model.state_dict(), MODEL_PATH / f"{model_name}_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {train_acc:.4f} ({train_auc:.4f}) | "
                      f"Val: {val_acc:.4f} ({val_auc:.4f}) | "
                      f"Best: {best_val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Evaluate on test set
        print(f"\n{'='*40}")
        print(f"Test Evaluation: {model_name}")
        print(f"{'='*40}")
        
        model.load_state_dict(torch.load(MODEL_PATH / f"{model_name}_best.pth"))
        test_loss, test_acc, test_auc, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, criterion, DEVICE
        )
        
        print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test AUC-ROC: {test_auc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=['T Wins', 'CT Wins']))
        
        if test_acc > best_overall['accuracy']:
            best_overall['accuracy'] = test_acc
            best_overall['auc'] = test_auc
            best_overall['model_name'] = model_name
            best_overall['history'] = history
            best_overall['test_preds'] = test_preds
            best_overall['test_labels'] = test_labels
            best_overall['test_probs'] = test_probs
    
    # Save best results
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_overall['model_name']}")
    print(f"Test Accuracy: {best_overall['accuracy']:.4f} ({best_overall['accuracy']*100:.2f}%)")
    print(f"Test AUC-ROC: {best_overall['auc']:.4f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'best_model': best_overall['model_name'],
        'test_accuracy': float(best_overall['accuracy']),
        'test_auc': float(best_overall['auc']),
        'num_features': X.shape[1],
        'num_samples': len(y),
        'feature_names': feature_names,
        'confusion_matrix': confusion_matrix(
            best_overall['test_labels'], 
            best_overall['test_preds']
        ).tolist()
    }
    
    with open(LOG_PATH / "mlp_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    history = best_overall['history']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].axhline(y=0.88, color='r', linestyle='--', label='Target (88%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history['train_auc'], label='Train AUC')
    axes[2].plot(history['val_auc'], label='Val AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC-ROC')
    axes[2].set_title('Training and Validation AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(LOG_PATH / "mlp_training_curves.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to {LOG_PATH}")
    print(f"Best model saved to {MODEL_PATH / best_overall['model_name']}_best.pth")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
