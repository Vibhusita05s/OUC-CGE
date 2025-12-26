import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Config

DATA_DIR = "data"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
DEVICE =  "cpu"


# Dataset

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Model

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(x)          # h: (1, B, H)
        h = h.squeeze(0)            # (B, H)
        return self.fc(h)


# Main

def main():
    print(" Loading data...")
    X = np.load(f"{DATA_DIR}/X.npy")
    y = np.load(f"{DATA_DIR}/y.npy")

    print("X:", X.shape)
    print("y:", y.shape)

    # Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = SequenceDataset(X_train, y_train)
    val_ds   = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GRUClassifier(input_dim=X.shape[2]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(" Training started...\n")

    for epoch in range(EPOCHS):
        #  Train 
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #  Validation 
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d} | Loss {total_loss:.3f} | Val Acc {acc:.3f}")

    torch.save(model.state_dict(), "gru_engagement.pth")
    print("\ Training complete. Model saved.")

if __name__ == "__main__":
    main()
