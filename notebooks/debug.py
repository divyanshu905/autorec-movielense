# %% imports
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import get_train_test, load_items, load_genres

# %% now safe to import project code

ROOT = Path.cwd()

print("CWD:", ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import pandas as pd
import numpy as np
from third_iteration import AutoEncoder, pivot_user_item_df  
from Exploration import MovieLenseDataset
from torch.utils.data import DataLoader


# %% Load checkpoint
checkpoint_path = ROOT / "checkpoints" / "autorec_movielens.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

checkpoint = torch.load(
    "./checkpoints/autorec_movielens.pt",
      map_location="cpu"
)

model = AutoEncoder(
    num_items=checkpoint["model_config"]["num_items"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# %%
checkpoint['final_train_loss']

# %% Load Data
train_df, test_df = get_train_test()
train_pivot, mask = pivot_user_item_df(train_df)

dataset = MovieLenseDataset(train_pivot, mask=mask)
loader = DataLoader(dataset)


# %% Predictions

all_preds = []
with torch.no_grad():
    for x, _ in loader:
        all_preds.append(model(x)) 

R_pred = torch.cat(all_preds)
R_pred_df = pd.DataFrame(
    R_pred.numpy(),
    index=train_pivot.index,
    columns=train_pivot.columns
)

melted_train_df = train_pivot.reset_index().melt(
    id_vars=['user_id'],
    var_name='item_id',
    value_name='ratings' 
)

unseen_pairs = melted_train_df[melted_train_df['ratings']==0]

melted_R_pred_df = R_pred_df.reset_index().melt(
    id_vars=['user_id'],
    var_name='item_id',
    value_name='ratings'
) 

final_pred_unseen_pairs = melted_R_pred_df.merge(
    unseen_pairs, 
    how='inner',
    on=['item_id', 'user_id'], 
)

k = 10

final_pred_unseen_pairs['rank'] = (
    final_pred_unseen_pairs.groupby('user_id')['ratings_x']
      .rank(method='first', ascending=False)
)

df_topk = final_pred_unseen_pairs[final_pred_unseen_pairs['rank'] <= k]

eval_df = test_df.merge(
    df_topk,
    how='left', 
    on=['item_id', 'user_id']
).drop(columns=['ratings_y', 'rank', 'timestamp'])

eval_df['predicted'] = eval_df['ratings_x'].notna().astype('int')

genre_dict = load_genres()
items_df = load_items(genre_dict=genre_dict)


eval_df_genre = eval_df.merge(
    items_df, 
    how='left',
    on='item_id'
).explode('genres') 

# %%

sample = train_pivot.sample(10)

sample_pred = df_topk[df_topk['user_id'].isin(sample.index)]

sample_topk = (
    sample_pred.sort_values(["user_id", "ratings_x"], ascending=[True, False])
      .groupby("user_id")
      .head(10)
)

sample_topk['item_id'].value_counts()

# %%

popular_items = pd.DataFrame(train_df['item_id'].value_counts())
recommended_items = pd.DataFrame(df_topk['item_id'].value_counts())

pop_v_rec = popular_items.merge(
    recommended_items,
    how='left',
    on='item_id'
)
pop_v_rec.rename(columns={'count_x':'seen_items', 'count_y':'unseen_items'}, inplace=True)


# %%
import matplotlib.pyplot as plt

plt.scatter(data=pop_v_rec, x='seen_items', y='unseen_items')
plt.show()

# %%
pop_v_rec[pop_v_rec['seen_items'] < 10].sort_values(by='unseen_items', ascending=False)


# %%
train_df[train_df['item_id'] == 1682]

# %%
df_topk[df_topk['item_id']==814]

# %%
