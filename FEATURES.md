# CS2 Round Prediction — Comprehensive Feature List

**Goal:** Extract EVERY useful feature from the data to maximize accuracy.

---

## Feature Categories

### 1. Player State Features (from `player_status`)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_alive` | int | CT players alive (0-5) |
| `t_alive` | int | T players alive (0-5) |
| `ct_health_total` | float | Sum of CT player health |
| `t_health_total` | float | Sum of T player health |
| `ct_health_avg` | float | Average CT health |
| `t_health_avg` | float | Average T health |
| `ct_armor_total` | int | Sum of CT armor values |
| `t_armor_total` | int | Sum of T armor values |
| `ct_has_armor_count` | int | CTs with armor > 0 |
| `t_has_armor_count` | int | Ts with armor > 0 |
| `ct_helmet_count` | int | CTs with helmet |
| `t_helmet_count` | int | Ts with helmet |
| `ct_defuser_count` | int | CTs with defuse kit |

### 2. Weapon Features (from `player_status.inv_*`)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_rifles` | int | Count of CT rifles (AK, M4, AUG, SG, Famas, Galil) |
| `t_rifles` | int | Count of T rifles |
| `ct_awps` | int | Count of CT AWPs |
| `t_awps` | int | Count of T AWPs |
| `ct_snipers` | int | Count of CT snipers (AWP, Scout, Auto) |
| `t_snipers` | int | Count of T snipers |
| `ct_smgs` | int | Count of CT SMGs |
| `t_smgs` | int | Count of T SMGs |
| `ct_shotguns` | int | Count of CT shotguns |
| `t_shotguns` | int | Count of T shotguns |
| `ct_pistol_only` | int | CTs with only pistol |
| `t_pistol_only` | int | Ts with only pistol |
| `ct_has_primary` | int | CTs with any primary weapon |
| `t_has_primary` | int | Ts with any primary weapon |

### 3. Utility Features (from `player_status.inv_*`)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_flashbangs` | int | Total CT flashbangs |
| `t_flashbangs` | int | Total T flashbangs |
| `ct_smokes` | int | Total CT smoke grenades |
| `t_smokes` | int | Total T smoke grenades |
| `ct_hegrenades` | int | Total CT HE grenades |
| `t_hegrenades` | int | Total T HE grenades |
| `ct_molotovs` | int | Total CT molotovs/incendiaries |
| `t_molotovs` | int | Total T molotovs |
| `ct_decoys` | int | Total CT decoys |
| `t_decoys` | int | Total T decoys |
| `ct_utility_total` | int | Total CT utility count |
| `t_utility_total` | int | Total T utility count |

### 4. Economy Features (from `player_status`)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_money_total` | float | Sum of CT money |
| `t_money_total` | float | Sum of T money |
| `ct_money_avg` | float | Average CT money |
| `t_money_avg` | float | Average T money |
| `ct_equipment_value` | float | Sum of CT equipment value |
| `t_equipment_value` | float | Sum of T equipment value |
| `ct_equipment_avg` | float | Average CT equipment value |
| `t_equipment_avg` | float | Average T equipment value |

### 5. Game State Features

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `bomb_planted` | bool | bomb_state | Is bomb planted |
| `bomb_site` | int | bomb_state | Which site (A=0, B=1) |
| `time_remaining` | float | tick/second | Seconds left in round |
| `time_elapsed` | float | tick/second | Seconds since round start |
| `round_num` | int | round | Current round number |
| `ct_score` | int | round_state | CT team score |
| `t_score` | int | round_state | T team score |
| `score_diff` | int | derived | CT score - T score |

### 6. Round Type Features

| Feature | Type | Description |
|---------|------|-------------|
| `is_pistol_round` | bool | Round 1 or 13 |
| `is_second_round` | bool | Round 2 or 14 (anti-eco) |
| `ct_round_type` | cat | eco/force/full based on avg money |
| `t_round_type` | cat | eco/force/full based on avg money |
| `is_force_buy_ct` | bool | CT forcing with low money |
| `is_force_buy_t` | bool | T forcing with low money |

### 7. Map Features (from `header`)

| Feature | Type | Description |
|---------|------|-------------|
| `map_de_dust2` | bool | One-hot encoding |
| `map_de_mirage` | bool | One-hot encoding |
| `map_de_inferno` | bool | One-hot encoding |
| `map_de_nuke` | bool | One-hot encoding |
| `map_de_overpass` | bool | One-hot encoding |
| `map_de_vertigo` | bool | One-hot encoding |
| `map_de_ancient` | bool | One-hot encoding |
| `map_de_anubis` | bool | One-hot encoding |

### 8. Player Skill Features (from `header`) ⭐ KEY FOR 90%+

| Feature | Type | Description |
|---------|------|-------------|
| `ct_avg_rank` | float | Average CT player rank |
| `t_avg_rank` | float | Average T player rank |
| `rank_diff` | float | CT rank - T rank |
| `ct_avg_wins` | float | Average CT player wins |
| `t_avg_wins` | float | Average T player wins |
| `wins_diff` | float | CT wins - T wins |

### 9. Derived Advantage Features

| Feature | Type | Description |
|---------|------|-------------|
| `man_advantage` | int | t_alive - ct_alive |
| `health_advantage` | float | t_health_total - ct_health_total |
| `equipment_advantage` | float | t_equipment - ct_equipment |
| `utility_advantage` | int | t_utility - ct_utility |
| `rifle_advantage` | int | t_rifles - ct_rifles |
| `awp_advantage` | int | t_awps - ct_awps |

### 10. Event-Based Features (computed at snapshot time)

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `first_blood_ct` | bool | player_death | CT got first kill |
| `first_blood_t` | bool | player_death | T got first kill |
| `kills_this_round_ct` | int | player_death | CT kills so far |
| `kills_this_round_t` | int | player_death | T kills so far |
| `headshot_kills_ct` | int | player_death | CT headshot kills |
| `headshot_kills_t` | int | player_death | T headshot kills |
| `awp_kills_ct` | int | player_death | Kills with AWP by CT |
| `awp_kills_t` | int | player_death | Kills with AWP by T |

### 11. Damage Features (from `player_hurt`)

| Feature | Type | Description |
|---------|------|-------------|
| `damage_dealt_ct` | float | Total damage by CT so far |
| `damage_dealt_t` | float | Total damage by T so far |
| `damage_taken_ct` | float | Total damage to CT so far |
| `damage_taken_t` | float | Total damage to T so far |

### 12. Flash/Blind Features (from `player_blind`)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_players_flashed` | int | CTs currently flashed |
| `t_players_flashed` | int | Ts currently flashed |
| `ct_flash_duration_total` | float | Total flash duration on CTs |
| `t_flash_duration_total` | float | Total flash duration on Ts |

### 13. Engagement Features (from `weapon_fire`)

| Feature | Type | Description |
|---------|------|-------------|
| `shots_fired_ct` | int | Shots fired by CT this round |
| `shots_fired_t` | int | Shots fired by T this round |
| `fight_intensity` | float | Shots per second (indicates active fight) |

### 14. Temporal Features

| Feature | Type | Description |
|---------|------|-------------|
| `time_since_last_kill` | float | Seconds since last death event |
| `time_since_bomb_plant` | float | Seconds since plant (0 if not planted) |
| `snapshot_type` | cat | round_start/first_blood/periodic/bomb_plant |

### 15. Historical Features (from previous rounds)

| Feature | Type | Description |
|---------|------|-------------|
| `ct_won_last_round` | bool | CT won previous round |
| `t_won_last_round` | bool | T won previous round |
| `ct_win_streak` | int | CT consecutive wins |
| `t_win_streak` | int | T consecutive wins |
| `ct_loss_streak` | int | CT consecutive losses (for loss bonus) |
| `t_loss_streak` | int | T consecutive losses |

---

## Total Feature Count

| Category | Count |
|----------|-------|
| Player State | 13 |
| Weapons | 14 |
| Utility | 12 |
| Economy | 8 |
| Game State | 8 |
| Round Type | 6 |
| Map | 8 |
| Player Skill | 6 |
| Advantages | 6 |
| Events | 10 |
| Damage | 4 |
| Flash | 4 |
| Engagement | 3 |
| Temporal | 3 |
| Historical | 6 |
| **TOTAL** | **~110 features** |

---

## Weapon Classification Reference

```python
WEAPON_CATEGORIES = {
    'rifle': ['ak47', 'm4a1', 'm4a1_silencer', 'm4a4', 'sg556', 'aug', 'famas', 'galilar'],
    'sniper': ['awp', 'ssg08', 'g3sg1', 'scar20'],
    'smg': ['mac10', 'mp9', 'mp7', 'mp5sd', 'ump45', 'p90', 'bizon'],
    'shotgun': ['nova', 'xm1014', 'sawedoff', 'mag7'],
    'pistol': ['glock', 'usp_silencer', 'hkp2000', 'p250', 'fiveseven', 'tec9', 'cz75a', 'deagle', 'elite', 'revolver'],
    'heavy': ['negev', 'm249'],
    'knife': ['knife', 'knife_t'],
    'grenade': ['flashbang', 'hegrenade', 'smokegrenade', 'molotov', 'incgrenade', 'decoy'],
    'equipment': ['c4', 'defuser', 'taser']
}
```

---

## Round Type Classification

```python
def classify_round_type(avg_team_money):
    if avg_team_money < 2000:
        return 'eco'
    elif avg_team_money < 4000:
        return 'force'
    else:
        return 'full_buy'
```

---

## Implementation Priority

### Must Have (Phase 2)
- Player state (alive, health, armor)
- Weapons (rifles, AWPs, SMGs)
- Utility counts
- Bomb status
- Time remaining
- Map
- Player ranks ⭐

### Should Have (Phase 2-3)
- Economy features
- Round type
- Advantages (derived)
- First blood
- Kill counts

### Nice to Have (Phase 5)
- Damage features
- Flash features
- Engagement intensity
- Historical streak features
