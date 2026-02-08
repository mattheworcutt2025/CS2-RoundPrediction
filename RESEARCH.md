# CS2 Round Prediction — Research & Strategy

## Executive Summary

**Target:** 90%+ accuracy on round winner prediction  
**Key Insight:** Prior work achieves 85-88% using **mid-round snapshots**, not round-start prediction.  
**Our Advantage:** Your dataset has tick-by-tick data with weapons, positions, and events — richer than Kaggle.

---

## 1. Literature Review

### 1.1 What Others Achieved

| Paper/Project | Accuracy | Method | Key Features |
|--------------|----------|--------|--------------|
| Kaggle CSGO Classification | 88.41% | Random Forest | Snapshots with weapons, players alive, bomb status |
| ScienceDirect 2025 | 90%+ | Streaming ML + sliding windows | Real-time snapshots, temporal patterns |
| IEEE 2022 (Xenopoulos) | ~85% | Various ML | Economy, round history |
| Valve's in-game system | ~90%+ (est.) | Proprietary | Unknown, likely tick-level data |

### 1.2 Why Round-Start Prediction Fails (~53%)

The current approach predicts at **round start**:
- Economy doesn't determine outcome (eco rounds are won regularly)
- Player skill/aim not captured
- No weapon context (money ≠ actual loadout)
- No tactical state (positions, utility available)

### 1.3 Why Mid-Round Snapshots Work (~88%)

High-accuracy models predict at **arbitrary points during the round**:
- **Players alive** is extremely predictive (5v3 = high CT win prob)
- **Bomb planted** status shifts probabilities dramatically
- **Time remaining** matters for CT retakes
- **Actual weapons** in hand vs theoretical economy

### 1.4 Gap to 90%+

The ScienceDirect paper achieves 90%+ using:
1. **Sliding windows** — Consider past N snapshots, not just current
2. **Temporal features** — How fast are kills happening?
3. **Event sequences** — Who died first? Was bomb planted early?
4. **Explainability** — Feature importance analysis

---

## 2. Our Data Advantage

### 2.1 Available Data (Per Match)

| File | Rows | Key Features |
|------|------|--------------|
| `player_status` | 157K+ | Health, armor, money, weapons, defuser, helmet AT EVERY TICK |
| `player_death` | ~39 | Kill events with attacker, headshot, weapon, position |
| `bomb_state` | varies | Plant/defuse events with site, timestamp |
| `item_equip` | ~571 | Weapon pickups with weapon_type_code |
| `player_vector` | 7M+ | Player positions every tick |
| `round_end` | ~14-30 | Outcomes with win_reason (elimination, bomb, time) |

### 2.2 Feature Richness vs Kaggle

| Feature | Kaggle Dataset | Our Dataset |
|---------|---------------|-------------|
| Players alive | ✓ | ✓ |
| Total health | ✓ | ✓ (per player) |
| Bomb status | ✓ | ✓ (with position) |
| Weapons | One-hot (limited) | Full inventory per player |
| Player positions | ✗ | ✓ (x, y, z) |
| Kill events | ✗ | ✓ (with attacker, headshot) |
| Utility count | Partial | ✓ (flash, smoke, HE, molly) |
| Defuse kit | ✓ | ✓ |

**Conclusion:** We have MORE data than the 88% solutions used.

---

## 3. Revised Problem Formulation

### 3.1 Old Approach (53% accuracy)
```
Input: Round-start state (economy, scores)
Output: CT wins (1) or T wins (0)
When: Predict once at round start
```

### 3.2 New Approach (Target 90%+)
```
Input: Game state at arbitrary tick t
Output: CT wins (1) or T wins (0)
When: Predict at multiple snapshots per round
```

### 3.3 Snapshot Strategy

Create snapshots at key moments:
1. **Round start** (after freeze time)
2. **First blood** (after first kill)
3. **Every 15 seconds** during round
4. **Bomb plant** (if it happens)
5. **Post-plant phases** (every 10 sec after plant)

This gives ~5-15 snapshots per round instead of 1.

---

## 4. Feature Engineering Plan

### 4.1 Core Features (from 88% models)

| Feature | Description | Source |
|---------|-------------|--------|
| `ct_alive` | CT players alive (0-5) | player_status |
| `t_alive` | T players alive (0-5) | player_status |
| `ct_health_total` | Sum of CT health | player_status |
| `t_health_total` | Sum of T health | player_status |
| `ct_armor_count` | CTs with armor | player_status |
| `t_armor_count` | Ts with armor | player_status |
| `bomb_planted` | Is bomb planted (0/1) | bomb_state |
| `time_remaining` | Seconds left in round | tick |
| `ct_score` | CT round wins | round_state |
| `t_score` | T round wins | round_state |

### 4.2 Weapon Features (our advantage)

| Feature | Description | Source |
|---------|-------------|--------|
| `ct_rifles` | Count of rifles (AK, M4, etc.) | player_status.inv_primary |
| `t_rifles` | Count of rifles | player_status.inv_primary |
| `ct_awps` | Count of AWPs | player_status.inv_primary |
| `t_awps` | Count of AWPs | player_status.inv_primary |
| `ct_smgs` | Count of SMGs | player_status.inv_primary |
| `t_smgs` | Count of SMGs | player_status.inv_primary |
| `ct_pistol_only` | Using only pistols | player_status |
| `t_pistol_only` | Using only pistols | player_status |

### 4.3 Utility Features

| Feature | Description | Source |
|---------|-------------|--------|
| `ct_flashbangs` | Total flashbangs | player_status.inv_flashbang |
| `t_flashbangs` | Total flashbangs | player_status.inv_flashbang |
| `ct_smokes` | Total smokes | player_status.inv_smokegrenade |
| `t_smokes` | Total smokes | player_status.inv_smokegrenade |
| `ct_molotovs` | Total molotovs/incendiaries | player_status.inv_molotov |
| `t_molotovs` | Total molotovs | player_status.inv_molotov |
| `ct_defusers` | CTs with defuse kit | player_status.has_defuser |

### 4.4 Temporal/Derived Features

| Feature | Description |
|---------|-------------|
| `man_advantage` | t_alive - ct_alive |
| `health_advantage` | t_health_total - ct_health_total |
| `equipment_advantage` | t_equipment_value - ct_equipment_value |
| `round_type` | Pistol(1), Eco(<$2000), Force($2000-$4000), Full(>$4000) |
| `time_since_first_blood` | Seconds since first kill |
| `time_since_bomb_plant` | Seconds since plant (if planted) |

### 4.5 Map Feature

| Feature | Description | Source |
|---------|-------------|--------|
| `map_name` | One-hot encoded map | header file |

---

## 5. Model Architecture Plan

### 5.1 Option A: Enhanced Tabular Model (Recommended First)

**Why:** Proven to work at 88%, faster iteration, explainable.

```
Model: XGBoost or LightGBM or Random Forest
Input: ~40-50 features per snapshot
Output: P(CT wins)
```

**Note:** Course requires deep learning — use this as baseline, then neural net.

### 5.2 Option B: Deep Learning - MLP

```
Architecture:
  Input (50 features)
  → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(64, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(1, Sigmoid)
```

### 5.3 Option C: Deep Learning - LSTM/Transformer (Sequence of Snapshots)

```
Input: Sequence of N snapshots from same round
Architecture:
  Embedding (per snapshot features)
  → LSTM/Transformer layers
  → Dense output
```

**Use Case:** Capture temporal dynamics within a round.

### 5.4 Option D: Temporal CNN

```
Input: (snapshots, features) treated as 1D signal
Architecture:
  Conv1D layers to capture local patterns
  → Global pooling → Dense
```

---

## 6. Data Pipeline

### 6.1 Extraction (New Script Needed)

```python
For each match:
    For each round:
        Get round outcome (CT/T win)
        For each snapshot_point in [round_start, first_blood, every_15s, bomb_plant, ...]:
            Extract player_status at that tick
            Aggregate by team
            Compute derived features
            Append to dataset with label
```

### 6.2 Expected Dataset Size

- 530 matches × ~14 rounds × ~8 snapshots = **~60,000 samples**
- Compare to Kaggle's 122,411 samples — similar scale

### 6.3 Scaling to Full 50GB

If 530 matches gives ~60K samples:
- Full dataset could have 10,000+ matches
- That's **1M+ samples** — serious deep learning territory

---

## 7. Execution Plan (Revised)

### Phase 1: Data Exploration (2 hours)
- [ ] Sample 10 matches, understand tick structure
- [ ] Map weapon codes to categories (rifle/SMG/sniper/pistol)
- [ ] Verify bomb_state events align with round_end
- [ ] Check for data quality issues

### Phase 2: Snapshot Extraction (4-6 hours)
- [ ] Write `extract_snapshots.py`
- [ ] Create snapshots at 5+ points per round
- [ ] Build full feature vector (~40-50 features)
- [ ] Generate `snapshots.parquet` or `.npy`

### Phase 3: Baseline Model (2 hours)
- [ ] Train XGBoost/RF on snapshot data (baseline)
- [ ] Target: 80%+ accuracy (proves features work)

### Phase 4: Deep Learning (3-4 hours)
- [ ] Implement MLP with same features
- [ ] Tune architecture (depth, width, dropout)
- [ ] Target: Match or beat XGBoost

### Phase 5: Advanced (If Needed) (4-6 hours)
- [ ] Add temporal features (LSTM over snapshot sequence)
- [ ] Experiment with attention/transformer
- [ ] Push for 90%+

### Phase 6: Documentation (3 hours)
- [ ] Jupyter notebook with EDA
- [ ] Training curves and metrics
- [ ] Feature importance analysis
- [ ] Final report

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Data parsing errors | Medium | Start with 10 matches, validate |
| 90% not achievable | Low | Even 85% is strong for course |
| CPU training too slow | Low | Snapshot data is small (~60K rows) |
| Feature engineering mistakes | Medium | Compare to Kaggle features |

---

## 9. Success Criteria

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Accuracy | 80% | 88% | 92%+ |
| F1 Score | 0.78 | 0.86 | 0.90+ |
| ROC-AUC | 0.85 | 0.92 | 0.96+ |

---

## 10. References

1. Kaggle CSGO Round Winner Classification Dataset (88% benchmark)
2. "Explainable e-sports win prediction through ML in streaming" - ScienceDirect 2025 (90%+)
3. IEEE "Predicting Round Result in Counter-Strike: GO Using ML" (2022)
4. GitHub: anantoj/csgo-round-winner-classification
5. GitHub: whiz-coder/CSGO-Round-Winner-Prediction
