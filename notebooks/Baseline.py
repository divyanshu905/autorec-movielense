#%% Import
import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd 
from src.data import get_train_test
import torch 
from torch.utils.data import DataLoader, Dataset 
import torch.nn as nn
import numpy as np


# %%
train_df, test_df = get_train_test()
train_df.head()
train_df_pivot = train_df.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)


# %%
train_df_pivot.isna().sum()


# %%
user_avg_rating = train_df.groupby(['user_id']).mean()['rating']

# %%
class MFDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users 
        self.ratings = ratings
        self.items = items 
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (
            self.users[idx], 
            self.items[idx],
            self.ratings[idx]
        )
    

user2idx = {u : i for i, u in enumerate(train_df['user_id'].unique())}
item2idx = {u : i for i, u in enumerate(train_df['item_id'].unique())}

train_df['user_id'] = train_df['user_id'].map(user2idx)
train_df['item_id'] = train_df['item_id'].map(item2idx)

users = torch.tensor(train_df['user_id'].values, dtype=torch.long)
items = torch.tensor(train_df['item_id'].values, dtype=torch.long)
ratings = torch.tensor(train_df['rating'].values, dtype=torch.float)


dataset = MFDataset(users, items, ratings)
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# %%

class MFModel(nn.Module):
    def __init__(self, n_users, n_items, k):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k) 

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    
    def forward(self, user, item):

        pu = self.user_embedding(user)
        qi = self.item_embedding(item)

        bi = self.item_bias(item).squeeze()
        bu = self.user_bias(user).squeeze()

        dot = (pu * qi).sum(dim=1)
        return dot + bu + bi


# %%
num_users = train_df['user_id'].nunique()
num_items = train_df['item_id'].nunique()

model = MFModel(
    n_users=num_users,
    n_items=num_items,
    k=5
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100 
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for users, items, ratings in train_loader:
        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, ratings)

        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()

    print(f"Epoch : {epoch}, Loss : {total_loss/len(train_loader) : .4f}")


# %%
def predict_for_user(model, user_id, n_items):
    user = torch.tensor([user_id]).repeat(n_items)
    items = torch.arange(n_items)

    with torch.no_grad():
        scores = model(user, items)
    
    return scores 



recommendations = {}
all_items = set(train_df['item_id'])

for user in train_df['user_id'].unique():
    seen_items = set(train_df[train_df['user_id'] == user].item_id)
    unseen_items = np.array(list(all_items - seen_items))
    preds =  predict_for_user(model, user, n_items=num_items).detach().numpy()
    preds = preds[unseen_items]
    topk_idx = np.argsort(preds)[-10:][::-1]
    topk_items = unseen_items[topk_idx]

    recommendations[user] = set(topk_items)

hits = 0
for _, row in test_df.iterrows():
    user = row.user_id
    item = row.item_id 

    if user in recommendations and item in recommendations[user]:
        hits += 1

recall_k = hits / len(test_df)
print(recall_k)
