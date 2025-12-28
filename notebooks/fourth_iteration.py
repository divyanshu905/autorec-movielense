# =========================
# Imports (SAFE)
# =========================
# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.data import get_train_test
from Exploration import MovieLenseDataset


# =========================
# Utilities (SAFE)
# =========================
def pivot_user_item_df(df):
    df_pivot = df.pivot(
        index="user_id",
        columns="item_id",
        values="rating"
    )
    mask = ~df_pivot.isna()
    df_pivot_filled = df_pivot.fillna(0)
    return df_pivot_filled, mask


# %%
# =========================
# Model (SAFE)
# =========================
class AutoEncoder(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 78),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(78, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_items)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# =========================
# Training + Eval (EXEC)
# =========================
# def main():
    # ---- Load data ----
train_df, test_df = get_train_test()
train_pivot, mask = pivot_user_item_df(train_df)

dataset = MovieLenseDataset(train_pivot, mask=mask)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # ---- Model ----
num_items = train_pivot.shape[1]
model = AutoEncoder(num_items=num_items)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# ---- Train ----
EPOCHS = 1000
for epoch in range(EPOCHS):
    total_loss = 0.0

    for x, m in loader:
        pred = model(x)
        pos_mask = (x >= 4).float() * m   
        loss = ((pred - x)**2 * m * pos_mask.float()).sum() / m.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(loader)
    print(f"Epoch {epoch} | Loss {epoch_loss:.4f}")

# %%
# ---- Eval ----
model.eval()
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

# ---- Recall@10 ----
hits = 0
for _, row in test_df.iterrows():
    user, item = row.user_id, row.item_id
    seen = set(train_df[train_df.user_id == user].item_id)
    unseen = list(set(train_pivot.columns) - seen)

    scores = R_pred_df.loc[user, unseen]
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
 
    top_k = (
        scores
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    if item in top_k:
        hits += 1

recall_at_k = hits / len(test_df)
print("Recall@10:", recall_at_k)

    # ---- Save checkpoint ----
    # Path("checkpoints").mkdir(exist_ok=True)
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "model_config": {"num_items": num_items},
    #         "epoch": epoch,
    #         "final_train_loss": epoch_loss,
    #         "item_ids": train_pivot.columns.values,
    #     },
    #     "checkpoints/autorec_movielens.pt"
    # )

# =========================
# Entry point (CRITICAL)
# =========================
# if __name__ == "__main__":
#     main()

# %%
def plot_pop_v_rec():
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

    popular_items = pd.DataFrame(train_df['item_id'].value_counts())
    recommended_items = pd.DataFrame(df_topk['item_id'].value_counts())

    pop_v_rec = popular_items.merge(
        recommended_items,
        how='left',
        on='item_id'
    )
    pop_v_rec.rename(columns={'count_x':'seen_items', 'count_y':'unseen_items'}, inplace=True)

    plt.scatter(data=pop_v_rec, x='seen_items', y='unseen_items')

plot_pop_v_rec()
plt.show()
# %%
