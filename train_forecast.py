import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_DIR = "data_forecast"
BATCH_SIZE = 8
EPOCHS = 25

LR = 1e-3
DEVICE =  "cpu"

class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

def main():
    X = np.load(f"{DATA_DIR}/X.npy")
    y = np.load(f"{DATA_DIR}/y.npy")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(
        ForecastDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        ForecastDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    model = GRUForecast(input_dim=X.shape[2]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Forecast training started\n")

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        print(f"Epoch {epoch+1:02d} | Loss {loss_sum:.3f} | Val Acc {correct/total:.3f}")

    torch.save(model.state_dict(), "gru_forecast.pth")
    print("\nForecast model saved")

if __name__ == "__main__":
    main()
