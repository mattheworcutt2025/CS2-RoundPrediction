# CS2 Round Prediction — Execution Plan v2

**Updated:** 2026-02-08 (Post-Research)  
**Target:** 90%+ accuracy using mid-round snapshots

---

## Key Insight

Previous approach predicted at **round start** → 53% accuracy.  
New approach predicts at **multiple points during the round** → 85-90%+ possible.

The difference: "Who will win this round?" (hard) vs "Given this game state, who wins?" (easier).

---

## Phase 1: Data Exploration (2-3 hours)

### 1.1 Understand Tick Structure
```python
# Questions to answer:
# - How many ticks per round?
# - What's the tick rate? (64 or 128 tick)
# - How does `second` column relate to real time?
```

### 1.2 Map Weapon Codes
```python
# Create weapon_mapping.json:
# {
#   "ak47": {"type": "rifle", "category": "T_rifle"},
#   "m4a1": {"type": "rifle", "category": "CT_rifle"},
#   "awp": {"type": "sniper", "category": "sniper"},
#   ...
# }
```

### 1.3 Validate Data Quality
- [ ] Check for missing rounds
- [ ] Verify player counts (should be 5v5 mostly)
- [ ] Confirm bomb_state aligns with round_end
- [ ] Check for disconnected players / bots

### 1.4 Sample Match Analysis
```bash
python 1b_explore_detailed.py  # New script
```

**Deliverable:** `exploration_report.md` with findings

---

## Phase 2: Snapshot Extraction (4-6 hours)

### 2.1 Define Snapshot Points

| Snapshot | When | Trigger |
|----------|------|---------|
| round_start | After freeze time ends | First tick where players can move |
| first_blood | After first kill | player_death event |
| periodic_1 | 30 sec into round | time-based |
| periodic_2 | 60 sec into round | time-based |
| periodic_3 | 90 sec into round | time-based |
| bomb_plant | When bomb is planted | bomb_state event |
| post_plant_1 | 10 sec after plant | time-based |
| post_plant_2 | 20 sec after plant | time-based |

### 2.2 Feature Extraction Per Snapshot

```python
def extract_snapshot(match_path, round_num, tick):
    """
    Returns feature dict for given game state.
    """
    features = {
        # Team composition
        'ct_alive': int,
        't_alive': int,
        'ct_health_total': float,
        't_health_total': float,
        
        # Equipment
        'ct_rifles': int,
        't_rifles': int,
        'ct_awps': int,
        't_awps': int,
        'ct_armor_count': int,
        't_armor_count': int,
        'ct_helmet_count': int,
        't_helmet_count': int,
        'ct_defuser_count': int,
        
        # Utility
        'ct_flashbangs': int,
        't_flashbangs': int,
        'ct_smokes': int,
        't_smokes': int,
        'ct_molotovs': int,
        't_molotovs': int,
        
        # Economy
        'ct_money_total': float,
        't_money_total': float,
        'ct_equipment_value': float,
        't_equipment_value': float,
        
        # Game state
        'bomb_planted': bool,
        'time_remaining': float,  # seconds
        'ct_score': int,
        't_score': int,
        'round_num': int,
        
        # Derived
        'man_advantage': int,  # t_alive - ct_alive
        'health_advantage': float,
        'is_pistol_round': bool,
        'round_type': str,  # eco/force/full
        
        # Map (one-hot or label)
        'map_name': str,
    }
    return features
```

### 2.3 Weapon Classification

```python
RIFLES = ['ak47', 'm4a1', 'm4a4', 'sg556', 'aug', 'famas', 'galilar']
SNIPERS = ['awp', 'ssg08', 'g3sg1', 'scar20']
SMGS = ['mac10', 'mp9', 'mp7', 'mp5', 'ump45', 'p90', 'bizon']
SHOTGUNS = ['nova', 'xm1014', 'sawedoff', 'mag7']
PISTOLS = ['glock', 'usp', 'hkp2000', 'p250', 'fiveseven', 'tec9', 'cz75', 'deagle', 'elite']
HEAVY = ['negev', 'm249']
```

### 2.4 Output Format

```
data/
├── snapshots_train.parquet   # 80% of data
├── snapshots_val.parquet     # 10% of data
├── snapshots_test.parquet    # 10% of data
├── feature_names.json
└── label_mapping.json
```

**Script:** `2b_extract_snapshots.py`

---

## Phase 3: Baseline Model (2-3 hours)

### 3.1 Quick Validation with XGBoost/RF

Even though course requires deep learning, validate features first:

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Train
model = XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test)}")
```

**Target:** 80%+ accuracy confirms features are working.

### 3.2 Feature Importance Analysis

```python
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Deliverable:** Feature importance plot

---

## Phase 4: Deep Learning Model (3-4 hours)

### 4.1 MLP Architecture

```python
class CS2RoundMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
```

### 4.2 Training Configuration

```python
config = {
    'batch_size': 256,
    'learning_rate': 0.001,
    'epochs': 100,
    'early_stopping_patience': 15,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'scheduler': 'ReduceLROnPlateau'
}
```

### 4.3 Script

`3b_train_mlp.py`

---

## Phase 5: Advanced Models (If Needed) (4-6 hours)

### 5.1 Sequence Model (LSTM over snapshots)

If we want to capture intra-round dynamics:

```python
class CS2SequenceModel(nn.Module):
    """
    Input: Sequence of snapshots from same round
    Output: Round winner prediction
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, 128, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### 5.2 Transformer Variant

```python
class CS2Transformer(nn.Module):
    def __init__(self, feature_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(feature_dim, n_heads),
            num_layers=n_layers
        )
        self.classifier = nn.Linear(feature_dim, 1)
```

---

## Phase 6: Scaling (Optional) (2-4 hours)

### 6.1 Expand to Full Dataset

If accuracy is good on 530 matches:
- Process remaining folders in D:\CS2_Data
- Expect 10x-20x more data

### 6.2 Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kfold.split(X, y):
    # Train and evaluate
```

---

## Phase 7: Documentation & Reporting (3-4 hours)

### 7.1 Jupyter Notebook Structure

```
CS2_Analysis.ipynb
├── 1. Introduction & Problem Statement
├── 2. Data Overview
├── 3. Exploratory Data Analysis
│   ├── Class distribution
│   ├── Feature distributions
│   └── Correlation analysis
├── 4. Feature Engineering
├── 5. Model Development
│   ├── Baseline (RF/XGB)
│   ├── Deep Learning (MLP)
│   └── Comparison
├── 6. Results & Evaluation
│   ├── Accuracy, Precision, Recall, F1
│   ├── ROC-AUC curve
│   ├── Confusion matrix
│   └── Feature importance
├── 7. Discussion
└── 8. Conclusion
```

### 7.2 Visualizations

- [ ] Training curves (loss, accuracy)
- [ ] ROC curve
- [ ] Confusion matrix heatmap
- [ ] Feature importance bar chart
- [ ] Accuracy by round type (pistol vs buy)
- [ ] Prediction confidence histogram

---

## Timeline

| Phase | Est. Time | Status |
|-------|-----------|--------|
| 1. Exploration | 2-3 hrs | ⏳ Next |
| 2. Snapshot Extraction | 4-6 hrs | Pending |
| 3. Baseline Model | 2-3 hrs | Pending |
| 4. Deep Learning | 3-4 hrs | Pending |
| 5. Advanced (optional) | 4-6 hrs | If needed |
| 6. Scaling (optional) | 2-4 hrs | If time permits |
| 7. Documentation | 3-4 hrs | Final |

**Total:** 14-24 hours (can be done over weekend)

---

## Decision Points

### After Phase 3 (Baseline):
- **If accuracy > 85%:** Proceed to deep learning
- **If accuracy 75-85%:** Improve features, then deep learning
- **If accuracy < 75%:** Re-evaluate feature extraction

### After Phase 4 (Deep Learning):
- **If accuracy > 88%:** Document and submit
- **If accuracy 80-88%:** Try Phase 5 (LSTM/Transformer)
- **If accuracy < 80%:** Debug features or try ensemble

---

## Files to Create

```
CS2_RoundPrediction/
├── README.md              ✅ Done
├── RESEARCH.md            ✅ Done  
├── PLAN.md                ✅ Done (this file)
├── 1_explore_data.py      ✅ Exists (basic)
├── 1b_explore_detailed.py ⏳ To create
├── 2_preprocess_data.py   ✅ Exists (deprecated)
├── 2b_extract_snapshots.py ⏳ To create
├── 3_train_lstm.py        ✅ Exists (deprecated)
├── 3b_train_mlp.py        ⏳ To create
├── 3c_train_sequence.py   ⏳ If needed
├── data/
│   ├── snapshots_*.parquet ⏳ To generate
│   └── feature_names.json  ⏳ To generate
├── models/
├── logs/
└── notebooks/
    └── CS2_Analysis.ipynb  ⏳ Final deliverable
```
