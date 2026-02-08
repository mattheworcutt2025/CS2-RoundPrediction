"""
CS2 Round Prediction - XGBoost Baseline
For comparison with deep learning approaches
"""
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from datetime import datetime

# Paths
DATA_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
LOG_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/logs")
MODEL_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/models")

def main():
    print("=" * 60)
    print("CS2 Round Prediction - XGBoost Baseline")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading snapshot data...")
    X = np.load(DATA_PATH / "X_snapshots.npy")
    y = np.load(DATA_PATH / "y_snapshots.npy")
    
    with open(DATA_PATH / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    
    print(f"Loaded {len(y)} snapshots with {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Train XGBoost with different configs
    configs = [
        {
            'name': 'XGB_Default',
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'XGB_Deep',
            'params': {
                'n_estimators': 300,
                'max_depth': 10,
                'learning_rate': 0.05,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        {
            'name': 'XGB_Large',
            'params': {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.03,
                'min_child_weight': 5,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    ]
    
    best_model = None
    best_accuracy = 0
    best_config = None
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training: {config['name']}")
        print(f"{'='*50}")
        
        model = XGBClassifier(**config['params'])
        
        # Cross-validation
        print("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train on full training set
        print("Training on full training set...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test AUC-ROC: {auc:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_config = config
    
    # Final evaluation with best model
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_config['name']}")
    print(f"Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['T Wins', 'CT Wins']))
    
    # Feature importance
    importance = best_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\nTop 20 Features:")
    for i in range(min(20, len(feature_names))):
        print(f"  {i+1}. {feature_names[indices[i]]}: {importance[indices[i]]:.4f}")
    
    # Save results
    results = {
        'best_model': best_config['name'],
        'test_accuracy': float(best_accuracy),
        'test_auc': float(roc_auc_score(y_test, y_prob)),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'feature_importance': {
            feature_names[i]: float(importance[i]) 
            for i in indices[:30]
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    with open(LOG_PATH / "xgboost_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_n = 25
    plt.barh(range(top_n), importance[indices[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features - {best_config["name"]}')
    plt.tight_layout()
    plt.savefig(LOG_PATH / "xgboost_feature_importance.png", dpi=150)
    plt.close()
    
    # Save model
    best_model.save_model(str(MODEL_PATH / "xgboost_best.json"))
    
    print(f"\nResults saved to {LOG_PATH}")
    print(f"Model saved to {MODEL_PATH / 'xgboost_best.json'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
