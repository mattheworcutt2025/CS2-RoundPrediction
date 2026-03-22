# CS2 Round Prediction — Deep Learning Project

**Course:** MGTA 611
**Team:** Ratul Sarker, Sam Matthew  
**Last Updated:** 2026-02-08

---

## 1. Problem Statement

**Goal:** Predict which team (CT or T) will win a given round in Counter-Strike 2 using deep learning.

**Why it matters:**
- Live betting markets, esports analytics
- Team strategy optimization
- Understanding what factors drive round outcomes

**Constraint:** Must use deep learning (not XGBoost/traditional ML) per course requirements.

---

## 2. Data Overview

**Source:** Kaggle CS2 Dataset  
**Location:** `D:\CS2_Data\csds\2023\12\10\`  
**Size:** ~92GB total, using December 10, 2023 subset

### Raw Data Structure
Each match is a folder with parquet files:
```
{match_uuid}/
├── round_end        # Round outcomes (winner_team_code: 2=CT, 3=T)
├── round_start      # Round start events
├── player_status    # Health, armor, money per tick
├── player_info      # Team assignments per round
├── player_death     # Kill events
├── item_equip       # Weapon/equipment pickups
└── ...
```

### Processed Dataset
- **Matches:** 530
- **Rounds:** 9,746
- **Features:** 15
- **Class Balance:** 48.9% CT wins / 51.1% T wins ✓

---

## 3. Current Implementation

### 3.1 Feature Engineering (v1)
| # | Feature | Description |
|---|---------|-------------|
| 1 | round_num | Round number (1-30+) |
| 2 | ct_score | CT team score |
| 3 | t_score | T team score |
| 4 | ct_health | Avg CT health at round start |
| 5 | ct_armor | Avg CT armor at round start |
| 6 | ct_money | Total CT money |
| 7 | t_health | Avg T health at round start |
| 8 | t_armor | Avg T armor at round start |
| 9 | t_money | Total T money |
| 10 | ct_deaths_prev | CT deaths in previous round |
| 11 | t_deaths_prev | T deaths in previous round |
| 12 | ct_players | CT player count |
| 13 | t_players | T player count |
| 14 | econ_diff | CT money - T money |
| 15 | (padding) | Reserved |

### 3.2 Model Architecture (v1)
```
Bidirectional LSTM
├── Input: (batch, seq_len=5, features=15)
├── LSTM: 2 layers, 128 hidden, dropout=0.3
├── FC: 256 → 64 → 32 → 1
└── Output: Sigmoid (P(CT wins))
```

### 3.3 Training Configuration
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Cross-Entropy
- **Batch Size:** 64
- **Early Stopping:** Patience=20
- **Device:** CPU (no CUDA available)

---

## 4. Results (Baseline v1)

| Metric | Value |
|--------|-------|
| Test Accuracy | **53.6%** |
| Val Accuracy (best) | 58.9% |
| Epochs Trained | 49 (early stopped) |
| Training Time | ~3 minutes |

### Confusion Matrix
```
              Predicted
            T     CT
Actual T  [383   365]
       CT [313   401]
```

### Analysis
- Model barely beats random (50%)
- Slight overfitting (val > test)
- Features lack predictive power for actual gunfight outcomes

---

## 5. Gap Analysis

### What's Missing

| Factor | Impact | Available in Data? |
|--------|--------|-------------------|
| Weapon loadouts | HIGH — AWP vs pistol is huge | ✓ `item_equip` |
| Player skill/rating | HIGH — pro vs amateur | ✗ Not in dataset |
| Utility usage | MEDIUM — flashes, smokes | Partial |
| Map name | MEDIUM — CT/T sided maps | ✓ Likely in metadata |
| Site take success | MEDIUM — bomb plant location | ✓ `bomb_planted` |
| Kill feed sequence | MEDIUM — who died first | ✓ `player_death` |
| Round type | MEDIUM — pistol/eco/force/full | Derivable |

### Why Current Model Underperforms
1. **Economy ≠ Outcome** — Full buys lose to ecos regularly
2. **No weapon context** — $4700 rifle vs $4700 saved SMG = different
3. **No temporal dynamics** — Kill order within round matters
4. **Sequence too short** — 5 rounds may miss momentum patterns

---

## 6. Improvement Plan

### Option A: Enhanced Features (Recommended First)
**Effort:** Medium | **Expected Gain:** +5-10%

Add from existing data:
- [ ] Weapon loadout encoding (rifle/SMG/AWP/pistol counts per team)
- [ ] Round type classification (pistol/eco/force/full buy)
- [ ] Map indicator (one-hot if available)
- [ ] First blood indicator from previous round
- [ ] Bomb plant success rate

### Option B: Transformer Architecture
**Effort:** Medium | **Expected Gain:** +3-7%

Replace LSTM with:
- Self-attention over round sequences
- Better long-range dependency capture
- Positional encoding for round order

### Option C: Intra-Round Model
**Effort:** High | **Expected Gain:** +10-15%

Predict mid-round using:
- Kill feed events (who dies, when)
- Real-time player positions
- Utility thrown
- More like live prediction

### Option D: More Data
**Effort:** Low | **Expected Gain:** +2-5%

Expand beyond Dec 10:
- Use full December 2023 folder
- More matches = better generalization

---

## 7. Recommended Next Steps

### Phase 1: Feature Engineering (Tonight)
1. Explore `item_equip` for weapon data
2. Add round type classification
3. Add first blood indicator
4. Retrain and compare

### Phase 2: Architecture Tuning (If needed)
1. Try Transformer encoder
2. Experiment with longer sequences (10-15 rounds)
3. Add attention visualization

### Phase 3: Documentation
1. Jupyter notebook with EDA
2. Training curves and analysis
3. Final report for submission

---

## 8. File Structure

```
CS2_RoundPrediction/
├── README.md              # This file
├── 1_explore_data.py      # Data exploration
├── 2_preprocess_data.py   # Feature extraction
├── 3_train_lstm.py        # LSTM training
├── data/
│   ├── X_features.npy     # Processed features (9746, 15)
│   └── y_labels.npy       # Labels (9746,)
├── models/
│   └── best_model.pth     # Saved PyTorch model
└── logs/
    ├── training_results.json
    └── training_curves.png
```

---

## 9. References

- Dataset: Kaggle CS2 Demo Dataset
- Framework: PyTorch 2.10.0 (CPU)
- Preprocessing: Polars + PyArrow
