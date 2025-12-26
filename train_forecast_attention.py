import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# CONFIG

DATA_DIR = "data_forecast"
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-3
DEVICE = "cpu"


# LOAD DATA

X = np.load(f"{DATA_DIR}/X_forecast.npy")  # (N, T, D)
y = np.load(f"{DATA_DIR}/y_forecast.npy")  # (N,)


# SPLIT: Train / Val / Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# SAVE SPLITS  
np.save(f"{DATA_DIR}/X_train.npy", X_train)
np.save(f"{DATA_DIR}/y_train.npy", y_train)
np.save(f"{DATA_DIR}/X_val.npy", X_val)
np.save(f"{DATA_DIR}/y_val.npy", y_val)
np.save(f"{DATA_DIR}/X_test.npy", X_test)
np.save(f"{DATA_DIR}/y_test.npy", y_test)

print("Saved train / val / test splits")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")


# DATASET

class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    ForecastDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    ForecastDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)


# MODEL

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h_seq, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(h_seq).squeeze(-1), dim=1)
        context = torch.sum(h_seq * attn_w.unsqueeze(-1), dim=1)
        logits = self.fc(context).squeeze(1)
        return logits, attn_w

model = GRUAttention(input_dim=X.shape[-1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# TRAIN LOOP

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, _ = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {total_loss:.3f} | "
        f"Val Acc {correct/total:.3f}"
    )


# SAVE MODEL

torch.save(model.state_dict(), "gru_forecast_attention.pth")
print("Model saved: gru_forecast_attention.pth")
