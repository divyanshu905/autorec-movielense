# Recommendation with Autoencoders
    # Show topN recommendation for a user (reconstruct ratings)
    # Use explicit user/item ratings which is sparse hence autoencoders
    # Masked MSE, recall @ k, precision @ k (Only on missing rating)
    # Data : Movielense data
    # Mean Centered User Rating
    # Eval on missing rating
    # A masked-loss autoencoder trained on MovieLens that outperforms matrix factorization on Recall@10 while producing interpretable latent factors
    # Baseline : Matrix Factorization/User mean

#%% Load Data Lib
import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

#%% Import Lib
import pandas as pd
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.nn.functional as F
from src.data import get_train_test

#%% Load data
train_df, test_df = get_train_test()
train_df.head()

# %% Pivot to user item rating matrix
train_data = train_df.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)
train_data.shape

# %% Fill NaN with 0
train_data_filled = train_data.fillna(0)
train_data_filled.isna().sum()

# %% Create a mask
mask  = ~train_data.isna()

# %% Create dataloader
class MovieLenseDataset(Dataset):
    def __init__(self, train_df_filled, mask):
        super().__init__()
        self.X = torch.tensor(train_df_filled.values, dtype=torch.float32)
        self.mask = torch.tensor(mask.values, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx]
    

train_dataset = MovieLenseDataset(train_data_filled, mask)
loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# %% Build model
class AutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(num_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_items)
    
    def forward(self, x):
        x = F.relu(self.encoder(x))
        out = self.decoder(x)
    
        return out 

# %% Train 
num_items = train_data_filled.shape[1]
hidden_dim = 64
model = AutoEncoder(num_items=num_items, hidden_dim=hidden_dim)
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

NUM_EPOCHS = 100 
for epoch in range(NUM_EPOCHS):
    losses = []
    for x, m in loader:
        pred = model(x)
        loss = ((pred - x)**2 * m).sum() / m.sum()
        losses.append(loss) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = sum(losses) / len(losses)
    print(f"Epoch: {epoch}, Loss: {epoch_loss:.4f}")

# %% Eval

all_preds = []

model.eval() 
with torch.no_grad():
    for x, m in loader:
        pred = model(x)        
        all_preds.append(pred) 

R_pred = torch.cat(all_preds, dim=0) 


R_pred_df = pd.DataFrame(
    R_pred.detach().numpy(),  # convert tensor to NumPy
    index=train_data.index,
    columns=train_data.columns
)

recommendations = {}
for user in train_data.index:
    seen_items = set(train_df[train_df.user_id == user].item_id)
    all_items = set(train_data.columns)

    unseen_items = list(all_items - seen_items)
    user_preds = R_pred_df.loc[user, unseen_items]
    top_k = user_preds.sort_values(ascending=False).head(10).index

    recommendations[user] = top_k

hits = 0 

for _, row in test_df.iterrows():
    user = row.user_id
    item = row.item_id

    if item in recommendations[user]:
        hits += 1 

recall_at_K = hits / len(test_df)
recall_at_K


# %%
