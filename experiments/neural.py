""""
Previously I had a dilemma in V0 whether or not NN would be a better choice.
I was plainly told it wasnt by my AI help.
However to view this practically, we can experiment with NN for say soh and determine
how accurate it would scale to the larger MIT V2 dataset.
SAME 8 engineered features (IR_ratio, chargetime_ratio, Tavg, Tmin, Tmax,
C1, Q1, C2) already used for the MIT dataset's SOH regressor.

This is a fair, direct comparison with the same features, same
GroupKFold methodology and same target. 

Existing Gradient Boosting result:
RMSE=0.0148, R2=0.8933
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "V2" / "source"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import models


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SOH_MLP(nn.Module):
    def __init__(self, input_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, epochs=60, lr=1e-3, batch_size=256):
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model = SOH_MLP(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val).float()).numpy()
    return val_pred


def main():
    df = models.load_clean_data()
    data = models.get_model_data(df)
    X = data[models.MODEL_FEATURES].values
    y = data["soh"].values
    groups = data["battery_id"].values
    print(f"{len(data)} rows, {data.battery_id.nunique()} batteries")

    gkf = GroupKFold(n_splits=5)
    rmses, maes, r2s = [], [], []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        pred = train_mlp(Xtr, y[tr], Xte, y[te])
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))

    print(f"\nMLP (same 8 engineered features): GroupKFold 5-fold")
    print(f"  RMSE={np.mean(rmses):.4f}  MAE={np.mean(maes):.4f}  R2={np.mean(r2s):.4f}")
    print(f"\nGradient Boosting: RMSE=0.0148  R2=0.8933")

if __name__ == "__main__":
    main()