import pandas as pd 
from pathlib import Path 
from src.config import load_config

cfg = load_config()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def load_raw_ratings():
    # Load data
    path = RAW_DIR / "u.data"
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    return pd.read_csv(path, sep="\t", names=cols) 

def leave_one_out_split(df, k, seed):
    train_rows, test_rows = [], []

    for user, user_df in df.groupby('user_id'):
        if len(user_df) < k * 2:
            continue 
        
        test_row = user_df.sample(k, random_state=seed)
        train_row = user_df.drop(test_row.index)

        test_rows.append(test_row)
        train_rows.append(train_row)
    
    return pd.concat(train_rows), pd.concat(test_rows)

def get_train_test():
    k = cfg['data']['k']
    seed = cfg['data']['seed']

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"train_k{k}.parquet"
    test_path = PROCESSED_DIR / f"test_k{k}.parquet"

    if train_path.exists() and test_path.exists():
        return (
            pd.read_parquet(train_path),
            pd.read_parquet(test_path)
        )

    df = load_raw_ratings()
    train_df, test_df = leave_one_out_split(df, k=k, seed=seed)

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    return train_df, test_df




