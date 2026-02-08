"""
CS2 Round Prediction - Comprehensive Snapshot Extraction
Extracts game state at multiple points per round with ~100 features
"""
import polars as pl
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("D:/CS2_Data/csds/2023/12/10")
OUTPUT_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Weapon categories by weapon code
RIFLES = [7, 8, 10, 13, 16, 39, 60]  # ak47, aug, famas, galilar, m4a1, sg556, m4a1_silencer
SNIPERS = [9, 40]  # awp, ssg08
SMGS = [17, 19, 23, 24, 26, 33, 34]  # mac10, p90, mp5sd, ump45, bizon, mp7, mp9
SHOTGUNS = [25, 27, 29, 35]  # xm1014, mag7, sawedoff, nova
PISTOLS = [1, 2, 3, 4, 30, 32, 36, 61, 64]  # deagle, elite, fiveseven, glock, tec9, hkp2000, p250, usp_silencer, revolver
HEAVY = [14, 28]  # m249, negev

# Team codes
CT_CODE = 2
T_CODE = 3

def load_match_data(match_path):
    """Load all relevant parquet files for a match"""
    data = {}
    files = ['player_status', 'player_info', 'player_death', 'bomb_state', 
             'round_end', 'round_state', 'header', 'player_hurt']
    
    for f in files:
        fpath = match_path / f
        if fpath.exists():
            try:
                data[f] = pl.read_parquet(fpath)
            except:
                data[f] = None
        else:
            data[f] = None
    
    return data

def get_team_assignments(player_info, round_num):
    """Get player IDs for each team in a round"""
    round_info = player_info.filter(pl.col('round') == round_num)
    if len(round_info) == 0:
        return [], []
    
    ct_players = round_info.filter(pl.col('team_code') == CT_CODE)['player_id'].to_list()
    t_players = round_info.filter(pl.col('team_code') == T_CODE)['player_id'].to_list()
    
    return ct_players, t_players

def categorize_weapon(weapon_code):
    """Categorize weapon code into type"""
    if weapon_code in RIFLES:
        return 'rifle'
    elif weapon_code in SNIPERS:
        return 'sniper'
    elif weapon_code in SMGS:
        return 'smg'
    elif weapon_code in SHOTGUNS:
        return 'shotgun'
    elif weapon_code in PISTOLS:
        return 'pistol'
    elif weapon_code in HEAVY:
        return 'heavy'
    else:
        return 'other'

def extract_snapshot_features(data, round_num, tick, ct_players, t_players, 
                               round_start_tick, round_end_tick, header_info,
                               deaths_before_tick, damage_before_tick):
    """
    Extract all features for a single snapshot (game state at a specific tick)
    """
    features = {}
    
    player_status = data['player_status']
    round_state = data['round_state']
    bomb_state = data['bomb_state']
    
    # Get player status at this tick
    ps = player_status.filter(
        (pl.col('round') == round_num) & 
        (pl.col('tick') == tick)
    )
    
    if len(ps) == 0:
        # Try nearest tick
        round_ps = player_status.filter(pl.col('round') == round_num)
        if len(round_ps) == 0:
            return None
        available_ticks = round_ps['tick'].unique().sort()
        nearest_tick = available_ticks.filter(pl.col('tick') <= tick).max()
        if nearest_tick is None:
            nearest_tick = available_ticks.min()
        ps = round_ps.filter(pl.col('tick') == nearest_tick)
    
    if len(ps) == 0:
        return None
    
    # Separate by team
    ct_ps = ps.filter(pl.col('player_id').is_in(ct_players))
    t_ps = ps.filter(pl.col('player_id').is_in(t_players))
    
    # ===== PLAYER STATE FEATURES =====
    # Alive counts (health > 0)
    ct_alive = len(ct_ps.filter(pl.col('health') > 0))
    t_alive = len(t_ps.filter(pl.col('health') > 0))
    
    features['ct_alive'] = ct_alive
    features['t_alive'] = t_alive
    features['man_advantage'] = t_alive - ct_alive
    
    # Health
    ct_health = ct_ps.filter(pl.col('health') > 0)['health'].sum() if ct_alive > 0 else 0
    t_health = t_ps.filter(pl.col('health') > 0)['health'].sum() if t_alive > 0 else 0
    features['ct_health_total'] = float(ct_health) if ct_health else 0.0
    features['t_health_total'] = float(t_health) if t_health else 0.0
    features['ct_health_avg'] = float(ct_health / ct_alive) if ct_alive > 0 else 0.0
    features['t_health_avg'] = float(t_health / t_alive) if t_alive > 0 else 0.0
    features['health_advantage'] = features['t_health_total'] - features['ct_health_total']
    
    # Armor
    ct_armor = ct_ps['armor'].sum() if len(ct_ps) > 0 else 0
    t_armor = t_ps['armor'].sum() if len(t_ps) > 0 else 0
    features['ct_armor_total'] = float(ct_armor) if ct_armor else 0.0
    features['t_armor_total'] = float(t_armor) if t_armor else 0.0
    
    ct_has_armor = len(ct_ps.filter(pl.col('armor') > 0))
    t_has_armor = len(t_ps.filter(pl.col('armor') > 0))
    features['ct_has_armor_count'] = ct_has_armor
    features['t_has_armor_count'] = t_has_armor
    
    # Helmet
    if 'has_helmet' in ct_ps.columns:
        ct_helmet = len(ct_ps.filter(pl.col('has_helmet') == True))
        t_helmet = len(t_ps.filter(pl.col('has_helmet') == True))
    else:
        ct_helmet, t_helmet = 0, 0
    features['ct_helmet_count'] = ct_helmet
    features['t_helmet_count'] = t_helmet
    
    # Defuser
    if 'has_defuser' in ct_ps.columns:
        ct_defuser = len(ct_ps.filter(pl.col('has_defuser') == True))
    else:
        ct_defuser = 0
    features['ct_defuser_count'] = ct_defuser
    
    # ===== WEAPON FEATURES =====
    def count_weapons_by_category(team_ps, category_codes):
        if 'inv_primary' not in team_ps.columns:
            return 0
        return len(team_ps.filter(pl.col('inv_primary').is_in(category_codes)))
    
    features['ct_rifles'] = count_weapons_by_category(ct_ps, RIFLES)
    features['t_rifles'] = count_weapons_by_category(t_ps, RIFLES)
    features['ct_awps'] = count_weapons_by_category(ct_ps, [9])  # AWP only
    features['t_awps'] = count_weapons_by_category(t_ps, [9])
    features['ct_snipers'] = count_weapons_by_category(ct_ps, SNIPERS)
    features['t_snipers'] = count_weapons_by_category(t_ps, SNIPERS)
    features['ct_smgs'] = count_weapons_by_category(ct_ps, SMGS)
    features['t_smgs'] = count_weapons_by_category(t_ps, SMGS)
    features['ct_shotguns'] = count_weapons_by_category(ct_ps, SHOTGUNS)
    features['t_shotguns'] = count_weapons_by_category(t_ps, SHOTGUNS)
    features['ct_heavy'] = count_weapons_by_category(ct_ps, HEAVY)
    features['t_heavy'] = count_weapons_by_category(t_ps, HEAVY)
    
    # Has primary weapon
    ct_has_primary = len(ct_ps.filter(pl.col('inv_primary') > 0)) if 'inv_primary' in ct_ps.columns else 0
    t_has_primary = len(t_ps.filter(pl.col('inv_primary') > 0)) if 'inv_primary' in t_ps.columns else 0
    features['ct_has_primary'] = ct_has_primary
    features['t_has_primary'] = t_has_primary
    
    features['rifle_advantage'] = features['t_rifles'] - features['ct_rifles']
    features['awp_advantage'] = features['t_awps'] - features['ct_awps']
    
    # ===== UTILITY FEATURES =====
    def sum_utility(team_ps, col):
        if col not in team_ps.columns:
            return 0
        val = team_ps[col].sum()
        return int(val) if val else 0
    
    features['ct_flashbangs'] = sum_utility(ct_ps, 'inv_flashbang')
    features['t_flashbangs'] = sum_utility(t_ps, 'inv_flashbang')
    features['ct_smokes'] = sum_utility(ct_ps, 'inv_smokegrenade')
    features['t_smokes'] = sum_utility(t_ps, 'inv_smokegrenade')
    features['ct_hegrenades'] = sum_utility(ct_ps, 'inv_hegrenade')
    features['t_hegrenades'] = sum_utility(t_ps, 'inv_hegrenade')
    features['ct_molotovs'] = sum_utility(ct_ps, 'inv_molotov') + sum_utility(ct_ps, 'inv_incgrenade')
    features['t_molotovs'] = sum_utility(t_ps, 'inv_molotov') + sum_utility(t_ps, 'inv_incgrenade')
    features['ct_decoys'] = sum_utility(ct_ps, 'inv_decoy')
    features['t_decoys'] = sum_utility(t_ps, 'inv_decoy')
    
    ct_utility = features['ct_flashbangs'] + features['ct_smokes'] + features['ct_hegrenades'] + features['ct_molotovs']
    t_utility = features['t_flashbangs'] + features['t_smokes'] + features['t_hegrenades'] + features['t_molotovs']
    features['ct_utility_total'] = ct_utility
    features['t_utility_total'] = t_utility
    features['utility_advantage'] = t_utility - ct_utility
    
    # ===== ECONOMY FEATURES =====
    ct_money = ct_ps['money'].sum() if 'money' in ct_ps.columns and len(ct_ps) > 0 else 0
    t_money = t_ps['money'].sum() if 'money' in t_ps.columns and len(t_ps) > 0 else 0
    features['ct_money_total'] = float(ct_money) if ct_money else 0.0
    features['t_money_total'] = float(t_money) if t_money else 0.0
    features['ct_money_avg'] = float(ct_money / len(ct_ps)) if len(ct_ps) > 0 else 0.0
    features['t_money_avg'] = float(t_money / len(t_ps)) if len(t_ps) > 0 else 0.0
    
    # Equipment value
    eq_col = 'equipment_value_calc' if 'equipment_value_calc' in ct_ps.columns else 'current_equipment_cost'
    if eq_col in ct_ps.columns:
        ct_equip = ct_ps[eq_col].sum()
        t_equip = t_ps[eq_col].sum()
        features['ct_equipment_value'] = float(ct_equip) if ct_equip else 0.0
        features['t_equipment_value'] = float(t_equip) if t_equip else 0.0
    else:
        features['ct_equipment_value'] = 0.0
        features['t_equipment_value'] = 0.0
    features['equipment_advantage'] = features['t_equipment_value'] - features['ct_equipment_value']
    
    # ===== GAME STATE FEATURES =====
    # Bomb status
    bomb_planted = False
    bomb_site = 0
    time_since_plant = 0.0
    
    if bomb_state is not None and len(bomb_state) > 0:
        round_bombs = bomb_state.filter(
            (pl.col('round') == round_num) & 
            (pl.col('tick') <= tick)
        )
        if len(round_bombs) > 0:
            # Check for plant event
            plants = round_bombs.filter(pl.col('event_type').is_in(['planted', 'plant']))
            if len(plants) > 0:
                bomb_planted = True
                plant_tick = plants['tick'].max()
                time_since_plant = (tick - plant_tick) / 64.0  # Assuming 64 tick
                if 'site_code' in plants.columns:
                    bomb_site = int(plants['site_code'].max()) if plants['site_code'].max() else 0
    
    features['bomb_planted'] = int(bomb_planted)
    features['bomb_site'] = bomb_site
    features['time_since_plant'] = time_since_plant
    
    # Time remaining (assuming 1:55 = 115 seconds round time)
    round_duration = 115.0  # seconds
    ticks_elapsed = tick - round_start_tick
    time_elapsed = ticks_elapsed / 64.0  # seconds (64 tick)
    time_remaining = max(0, round_duration - time_elapsed)
    
    if bomb_planted:
        # Bomb timer is 40 seconds
        time_remaining = max(0, 40.0 - time_since_plant)
    
    features['time_elapsed'] = time_elapsed
    features['time_remaining'] = time_remaining
    features['round_progress'] = min(1.0, time_elapsed / round_duration)
    
    # Round number
    features['round_num'] = round_num
    
    # Scores
    if round_state is not None and len(round_state) > 0:
        rs = round_state.filter(
            (pl.col('round') == round_num) & 
            (pl.col('tick') <= tick)
        )
        if len(rs) > 0:
            features['ct_score'] = int(rs['ct_score'].max()) if rs['ct_score'].max() else 0
            features['t_score'] = int(rs['t_score'].max()) if rs['t_score'].max() else 0
        else:
            features['ct_score'] = 0
            features['t_score'] = 0
    else:
        features['ct_score'] = 0
        features['t_score'] = 0
    
    features['score_diff'] = features['ct_score'] - features['t_score']
    
    # Round type
    features['is_pistol_round'] = int(round_num in [1, 13])
    features['is_second_round'] = int(round_num in [2, 14])
    
    avg_ct_money = features['ct_money_avg']
    avg_t_money = features['t_money_avg']
    
    # Round type classification
    def classify_round(avg_money):
        if avg_money < 2000:
            return 0  # eco
        elif avg_money < 4000:
            return 1  # force
        else:
            return 2  # full buy
    
    features['ct_round_type'] = classify_round(avg_ct_money)
    features['t_round_type'] = classify_round(avg_t_money)
    
    # ===== MAP FEATURES =====
    if header_info:
        map_name = header_info.get('map_name', 'unknown')
        maps = ['de_dust2', 'de_mirage', 'de_inferno', 'de_nuke', 
                'de_overpass', 'de_vertigo', 'de_ancient', 'de_anubis']
        for m in maps:
            features[f'map_{m}'] = int(map_name == m)
        
        # Player ranks
        features['ct_avg_rank'] = float(header_info.get('ct_starters_avg_rank', 0) or 0)
        features['t_avg_rank'] = float(header_info.get('t_starters_avg_rank', 0) or 0)
        features['rank_diff'] = features['ct_avg_rank'] - features['t_avg_rank']
        features['ct_avg_wins'] = float(header_info.get('ct_starters_avg_wins', 0) or 0)
        features['t_avg_wins'] = float(header_info.get('t_starters_avg_wins', 0) or 0)
    else:
        for m in ['de_dust2', 'de_mirage', 'de_inferno', 'de_nuke', 
                  'de_overpass', 'de_vertigo', 'de_ancient', 'de_anubis']:
            features[f'map_{m}'] = 0
        features['ct_avg_rank'] = 0.0
        features['t_avg_rank'] = 0.0
        features['rank_diff'] = 0.0
        features['ct_avg_wins'] = 0.0
        features['t_avg_wins'] = 0.0
    
    # ===== EVENT FEATURES =====
    # Kills before this tick
    if deaths_before_tick is not None and len(deaths_before_tick) > 0:
        ct_kills = len(deaths_before_tick.filter(pl.col('player_team_code') == T_CODE))  # CT killed T
        t_kills = len(deaths_before_tick.filter(pl.col('player_team_code') == CT_CODE))  # T killed CT
        features['kills_this_round_ct'] = ct_kills
        features['kills_this_round_t'] = t_kills
        
        # First blood
        if len(deaths_before_tick) > 0:
            first_death = deaths_before_tick.sort('tick').head(1)
            first_death_team = first_death['player_team_code'][0]
            features['first_blood_ct'] = int(first_death_team == T_CODE)  # CT got first kill
            features['first_blood_t'] = int(first_death_team == CT_CODE)  # T got first kill
        else:
            features['first_blood_ct'] = 0
            features['first_blood_t'] = 0
        
        # Headshot kills
        if 'is_headshot' in deaths_before_tick.columns:
            hs_kills_ct = len(deaths_before_tick.filter(
                (pl.col('player_team_code') == T_CODE) & 
                (pl.col('is_headshot') == True)
            ))
            hs_kills_t = len(deaths_before_tick.filter(
                (pl.col('player_team_code') == CT_CODE) & 
                (pl.col('is_headshot') == True)
            ))
            features['headshot_kills_ct'] = hs_kills_ct
            features['headshot_kills_t'] = hs_kills_t
        else:
            features['headshot_kills_ct'] = 0
            features['headshot_kills_t'] = 0
    else:
        features['kills_this_round_ct'] = 0
        features['kills_this_round_t'] = 0
        features['first_blood_ct'] = 0
        features['first_blood_t'] = 0
        features['headshot_kills_ct'] = 0
        features['headshot_kills_t'] = 0
    
    # ===== DAMAGE FEATURES =====
    if damage_before_tick is not None and len(damage_before_tick) > 0:
        # Damage dealt by each team
        ct_damage = damage_before_tick.filter(
            pl.col('attacker_team_code') == CT_CODE
        )['health_removed'].sum()
        t_damage = damage_before_tick.filter(
            pl.col('attacker_team_code') == T_CODE
        )['health_removed'].sum()
        features['damage_dealt_ct'] = float(ct_damage) if ct_damage else 0.0
        features['damage_dealt_t'] = float(t_damage) if t_damage else 0.0
    else:
        features['damage_dealt_ct'] = 0.0
        features['damage_dealt_t'] = 0.0
    
    return features

def process_match(match_path):
    """Process a single match and extract all snapshots"""
    data = load_match_data(match_path)
    
    if data['player_status'] is None or data['round_end'] is None or data['player_info'] is None:
        return [], []
    
    player_status = data['player_status']
    round_end = data['round_end']
    player_info = data['player_info']
    player_death = data.get('player_death')
    player_hurt = data.get('player_hurt')
    
    # Get header info
    header_info = None
    if data['header'] is not None and len(data['header']) > 0:
        header_info = data['header'].to_dicts()[0]
    
    all_features = []
    all_labels = []
    
    # Get unique rounds
    rounds = sorted(round_end['round'].unique().to_list())
    
    for round_num in rounds:
        try:
            # Get round outcome
            round_outcome = round_end.filter(pl.col('round') == round_num)
            if len(round_outcome) == 0:
                continue
            winner_team = round_outcome['winner_team_code'][0]
            label = 1 if winner_team == CT_CODE else 0  # 1 = CT wins
            
            # Get team assignments
            ct_players, t_players = get_team_assignments(player_info, round_num)
            if len(ct_players) == 0 or len(t_players) == 0:
                continue
            
            # Get round tick range
            round_ps = player_status.filter(pl.col('round') == round_num)
            if len(round_ps) == 0:
                continue
            
            round_ticks = round_ps['tick'].unique().sort().to_list()
            if len(round_ticks) < 10:
                continue
            
            round_start_tick = min(round_ticks)
            round_end_tick = max(round_ticks)
            
            # Get deaths for this round
            round_deaths = None
            if player_death is not None:
                round_deaths = player_death.filter(pl.col('round') == round_num)
            
            # Get damage for this round
            round_damage = None
            if player_hurt is not None:
                round_damage = player_hurt.filter(pl.col('round') == round_num)
            
            # ===== SNAPSHOT STRATEGY =====
            # Take snapshots at: start, every 10 seconds, every kill, bomb plant
            snapshot_ticks = set()
            
            # 1. Round start
            snapshot_ticks.add(round_ticks[0])
            
            # 2. Every 10 seconds (640 ticks at 64 tick rate)
            for t in range(round_start_tick, round_end_tick, 640):
                if t in round_ticks or any(abs(t - rt) < 32 for rt in round_ticks):
                    closest = min(round_ticks, key=lambda x: abs(x - t))
                    snapshot_ticks.add(closest)
            
            # 3. After each kill (5 ticks after)
            if round_deaths is not None and len(round_deaths) > 0:
                for death_tick in round_deaths['tick'].to_list():
                    snapshot_tick = death_tick + 5
                    if snapshot_tick <= round_end_tick:
                        closest = min(round_ticks, key=lambda x: abs(x - snapshot_tick))
                        snapshot_ticks.add(closest)
            
            # 4. At bomb plant
            if data['bomb_state'] is not None:
                bomb_events = data['bomb_state'].filter(pl.col('round') == round_num)
                if len(bomb_events) > 0:
                    for event_tick in bomb_events['tick'].to_list():
                        if event_tick in round_ticks:
                            snapshot_ticks.add(event_tick)
            
            # Sort ticks
            snapshot_ticks = sorted(list(snapshot_ticks))
            
            # Extract features for each snapshot
            for tick in snapshot_ticks:
                # Get deaths and damage before this tick
                deaths_before = None
                damage_before = None
                
                if round_deaths is not None and len(round_deaths) > 0:
                    deaths_before = round_deaths.filter(pl.col('tick') < tick)
                
                if round_damage is not None and len(round_damage) > 0:
                    damage_before = round_damage.filter(pl.col('tick') < tick)
                
                features = extract_snapshot_features(
                    data, round_num, tick, ct_players, t_players,
                    round_start_tick, round_end_tick, header_info,
                    deaths_before, damage_before
                )
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label)
                    
        except Exception as e:
            continue
    
    return all_features, all_labels

def main():
    print("=" * 60)
    print("CS2 Round Prediction - Snapshot Extraction")
    print("=" * 60)
    
    match_folders = sorted(list(DATA_PATH.glob("*")))
    print(f"Total matches: {len(match_folders)}")
    
    all_features = []
    all_labels = []
    success_count = 0
    errors = []
    
    for i, match_path in enumerate(tqdm(match_folders, desc="Processing matches")):
        try:
            features, labels = process_match(match_path)
            if len(features) > 0:
                all_features.extend(features)
                all_labels.extend(labels)
                success_count += 1
        except Exception as e:
            errors.append(f"{match_path.name}: {str(e)[:50]}")
            continue
        
        # Save checkpoint every 100 matches
        if (i + 1) % 100 == 0 and len(all_features) > 0:
            print(f"\nCheckpoint at {i+1}: {len(all_labels)} snapshots so far")
            _save_checkpoint(all_features, all_labels, i+1)
    
    print(f"\nMatches processed: {success_count}/{len(match_folders)}")
    print(f"Total snapshots: {len(all_labels)}")
    if errors:
        print(f"Errors: {len(errors)}")

def _save_checkpoint(all_features, all_labels, match_num):
    """Save intermediate results"""
    try:
        feature_names = list(all_features[0].keys())
        X = np.array([[f.get(k, 0) for k in feature_names] for f in all_features], dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(OUTPUT_PATH / f"X_checkpoint_{match_num}.npy", X)
        np.save(OUTPUT_PATH / f"y_checkpoint_{match_num}.npy", y)
    except:
        pass
    
    if len(all_features) == 0:
        print("ERROR: No snapshots extracted!")
        return
    
    # Convert to numpy arrays
    feature_names = list(all_features[0].keys())
    print(f"Features: {len(feature_names)}")
    
    X = np.array([[f[k] for k in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"CT wins: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"T wins: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")
    
    # Save
    np.save(OUTPUT_PATH / "X_snapshots.npy", X)
    np.save(OUTPUT_PATH / "y_snapshots.npy", y)
    
    with open(OUTPUT_PATH / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Features: {feature_names}")

if __name__ == "__main__":
    main()
