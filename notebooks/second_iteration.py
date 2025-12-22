#%% Import
import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
from src.data import get_train_test
import torch.nn.functional as F 
from Exploration import MovieLenseDataset

# %% Data preparation
train_df, test_df = get_train_test()

def pivot_user_item_df(df):
    df_pivot = df.pivot(
        index='user_id',
        columns='item_id',
        values='rating'
    )

    df_pivot_filled = df_pivot.fillna(0)
    mask = ~df_pivot_filled.isna()

    return df_pivot_filled, mask 

train_pivot, mask = pivot_user_item_df(train_df)

# %%  Train Dataset
train_dataset = MovieLenseDataset(train_pivot, mask=mask)
loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


#%%
train_pivot.shape

# %%
class AutoEncoder(nn.Module):
    def __init__(self, num_items):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1681, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1681)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)

        return out 

# %%

num_items = train_pivot.shape[1]
model = AutoEncoder(num_items=num_items)
LR = 0.001 
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
EPOCHS = 1000 

for epoch in range(EPOCHS):
    total_loss = 0 
    for x, mask in loader:
        optimizer.zero_grad()

        preds = model(x)
        loss = ((preds - x)**2 * mask).sum() / mask.sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / 128
    print(f"Epoch : {epoch}, Loss : {epoch_loss}")


# %% Eval 

all_preds = []

model.eval()
with torch.no_grad():
    for x, m in loader:
        pred = model(x) 
        all_preds.append(pred)

R_pred = torch.cat(all_preds, dim=0)

R_pred_df = pd.DataFrame(
    R_pred.detach().numpy(),
    index=train_pivot.index,
    columns=train_pivot.columns
)

recommendations = {}

for user in train_pivot.index:
    seen_items = set(train_df[train_df['user_id'] == user].item_id)
    all_items = set(train_pivot.columns)

    unseen_items = list(all_items - seen_items)
    user_preds = R_pred_df.loc[user, unseen_items]
    topk = user_preds.sort_values(ascending=False).head(10).index

    recommendations[user] = topk

hits = 0
for _, row in test_df.iterrows():
    user = row.user_id
    item = row.item_id

    if item in topk:
        hits += 1
    
recall_at_K = hits/len(test_df)




# %%
print(recall_at_K)
# %%
