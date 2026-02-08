"""
CS2 Round Prediction - Data Preprocessing v2
Fixed: Join player_status with player_info to get team assignments
"""
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path("D:/CS2_Data/csds/2023/12/10")
OUTPUT_PATH = Path("C:/Users/Ratul Sarker/Desktop/CS2_RoundPrediction/data")

def extract_match_features(match_path):
    """Extract features from a single match"""
    features = []
    labels = []
    
    try:
        # Load required files
        if not (match_path / "round_end").exists():
            return None, None
        round_end = pl.read_parquet(match_path / "round_end")
        
        # Load player info for team assignments
        if not (match_path / "player_info").exists():
            return None, None
        player_info = pl.read_parquet(match_path / "player_info")
        
        # Load player status (has health, armor, money)
        player_status = None
        if (match_path / "player_status").exists():
            player_status = pl.read_parquet(match_path / "player_status")
        
        # Load round state for scores
        round_state = None
        if (match_path / "round_state").exists():
            round_state = pl.read_parquet(match_path / "round_state")
        
        # Load player deaths for K/D info
        player_death = None
        if (match_path / "player_death").exists():
            player_death = pl.read_parquet(match_path / "player_death")
        
        # Get unique rounds
        rounds = sorted(round_end['round'].unique().to_list())
        
        for round_num in rounds:
            try:
                # Get round outcome
                round_outcome = round_end.filter(pl.col('round') == round_num)
                if len(round_outcome) == 0:
                    continue
                winner_team = round_outcome['winner_team_code'][0]
                
                # Get player info for this round (team assignments)
                round_players = player_info.filter(pl.col('round') == round_num)
                if len(round_players) == 0:
                    continue
                
                # Get player IDs by team
                ct_players = round_players.filter(pl.col('team_code') == 2)['player_id'].to_list()
                t_players = round_players.filter(pl.col('team_code') == 3)['player_id'].to_list()
                
                # Initialize feature vector
                round_features = []
                
                # Feature 1: Round number (economy phases: 1-3 pistol, 4-6 buy, etc.)
                round_features.append(float(round_num))
                
                # Feature 2-3: Current scores
                if round_state is not None:
                    rs = round_state.filter(pl.col('round') == round_num)
                    if len(rs) > 0 and 'ct_score' in rs.columns and 't_score' in rs.columns:
                        ct_score = float(rs['ct_score'].max()) if rs['ct_score'].max() else 0.0
                        t_score = float(rs['t_score'].max()) if rs['t_score'].max() else 0.0
                    else:
                        ct_score, t_score = 0.0, 0.0
                else:
                    ct_score, t_score = 0.0, 0.0
                round_features.extend([ct_score, t_score])
                
                # Features 4-9: Player status at round start
                if player_status is not None:
                    # Get first tick of round (start state)
                    round_ps = player_status.filter(pl.col('round') == round_num)
                    if len(round_ps) > 0:
                        first_tick = round_ps['tick'].min()
                        start_ps = round_ps.filter(pl.col('tick') == first_tick)
                        
                        # CT stats
                        ct_ps = start_ps.filter(pl.col('player_id').is_in(ct_players))
                        ct_health = float(ct_ps['health'].mean()) if len(ct_ps) > 0 else 100.0
                        ct_armor = float(ct_ps['armor'].mean()) if len(ct_ps) > 0 else 0.0
                        ct_money = float(ct_ps['money'].sum()) if len(ct_ps) > 0 and 'money' in ct_ps.columns else 4000.0
                        
                        # T stats
                        t_ps = start_ps.filter(pl.col('player_id').is_in(t_players))
                        t_health = float(t_ps['health'].mean()) if len(t_ps) > 0 else 100.0
                        t_armor = float(t_ps['armor'].mean()) if len(t_ps) > 0 else 0.0
                        t_money = float(t_ps['money'].sum()) if len(t_ps) > 0 and 'money' in t_ps.columns else 4000.0
                    else:
                        ct_health, ct_armor, ct_money = 100.0, 0.0, 4000.0
                        t_health, t_armor, t_money = 100.0, 0.0, 4000.0
                else:
                    ct_health, ct_armor, ct_money = 100.0, 0.0, 4000.0
                    t_health, t_armor, t_money = 100.0, 0.0, 4000.0
                
                round_features.extend([ct_health, ct_armor, ct_money, t_health, t_armor, t_money])
                
                # Feature 10-11: Previous round deaths
                if player_death is not None and round_num > 1:
                    prev_deaths = player_death.filter(pl.col('round') == round_num - 1)
                    ct_deaths = len(prev_deaths.filter(pl.col('player_id').is_in(ct_players)))
                    t_deaths = len(prev_deaths.filter(pl.col('player_id').is_in(t_players)))
                else:
                    ct_deaths, t_deaths = 0, 0
                round_features.extend([float(ct_deaths), float(t_deaths)])
                
                # Feature 12-13: Team sizes
                round_features.extend([float(len(ct_players)), float(len(t_players))])
                
                # Feature 14: Economy advantage (CT money - T money)
                econ_diff = ct_money - t_money
                round_features.append(econ_diff)
                
                # Ensure exactly 15 features
                round_features = round_features[:15]
                while len(round_features) < 15:
                    round_features.append(0.0)
                
                features.append(round_features)
                labels.append(1 if winner_team == 2 else 0)  # 1 = CT wins
                
            except Exception as e:
                continue
                
    except Exception as e:
        return None, None
    
    if len(features) == 0:
        return None, None
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    print("CS2 Round Prediction - Data Preprocessing v2")
    print("=" * 50)
    
    match_folders = sorted(list(DATA_PATH.glob("*")))
    print(f"Total matches: {len(match_folders)}")
    
    all_features = []
    all_labels = []
    success_count = 0
    
    for match_path in tqdm(match_folders, desc="Processing"):
        features, labels = extract_match_features(match_path)
        if features is not None and len(features) > 0:
            all_features.append(features)
            all_labels.append(labels)
            success_count += 1
    
    if len(all_features) == 0:
        print("ERROR: No valid data extracted!")
        return
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    # Clean data
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nMatches with data: {success_count}/{len(match_folders)}")
    print(f"Total rounds: {len(y)}")
    print(f"Feature shape: {X.shape}")
    print(f"CT wins: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"T wins: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH / "X_features.npy", X)
    np.save(OUTPUT_PATH / "y_labels.npy", y)
    
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nFeatures: round_num, ct_score, t_score, ct_health, ct_armor, ct_money,")
    print("          t_health, t_armor, t_money, ct_deaths_prev, t_deaths_prev,")
    print("          ct_players, t_players, econ_diff, [padding]")

if __name__ == "__main__":
    main()
