# =====================
# Imports
# =====================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from src.data import get_train_test


# =====================
# Dataset
# =====================
class MovieLenseDataset(Dataset):
    def __init__(self, train_df_filled, mask):
        self.X = torch.tensor(train_df_filled.values, dtype=torch.float32)
        self.mask = torch.tensor(mask.values, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx]


# =====================
# Model
# =====================
class AutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(num_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        return self.decoder(z)


# =====================
# Main (EXECUTION)
# =====================
def main():
    # Load data
    train_df, test_df = get_train_test()

    train_data = train_df.pivot(
        index="user_id",
        columns="item_id",
        values="rating"
    )

    train_data_filled = train_data.fillna(0)
    mask = ~train_data.isna()

    dataset = MovieLenseDataset(train_data_filled, mask)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Model
    num_items = train_data_filled.shape[1]
    model = AutoEncoder(num_items=num_items, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ---- TRAIN ----
    for epoch in range(100):
        losses = []
        for x, m in loader:
            pred = model(x)
            loss = ((pred - x) ** 2 * m).sum() / m.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch} | Loss {sum(losses)/len(losses):.4f}")

    # ---- EVAL ----
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, _ in loader:
            all_preds.append(model(x))

    R_pred = torch.cat(all_preds)

    R_pred_df = pd.DataFrame(
        R_pred.numpy(),
        index=train_data.index,
        columns=train_data.columns
    )

    # Recall@10
    hits = 0
    for _, row in test_df.iterrows():
        user, item = row.user_id, row.item_id
        seen = set(train_df[train_df.user_id == user].item_id)
        unseen = list(set(train_data.columns) - seen)

        top_k = (
            R_pred_df.loc[user, unseen]
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        if item in top_k:
            hits += 1

    print("Recall@10:", hits / len(test_df))


# =====================
# Entry point (CRITICAL)
# =====================
if __name__ == "__main__":
    main()
