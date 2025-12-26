import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


# CONFIG

DATA_DIR = "data_forecast"
MODEL_PATH = "gru_forecast_attention.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# LOAD TEST DATA ONLY

X = np.load(f"{DATA_DIR}/X_test.npy")
y = np.load(f"{DATA_DIR}/y_test.npy")

print(f"Loaded TEST data: X={X.shape}, y={y.shape}")


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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# INFERENCE

with torch.no_grad():
    logits, _ = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    probs = torch.sigmoid(logits).cpu().numpy()

preds = (probs > 0.5).astype(int)


# CONFUSION MATRIX

cm = confusion_matrix(y, preds)
print("\nConfusion Matrix:")
print(cm)


# CLASSIFICATION REPORT

print("\nClassification Report:")
print(classification_report(y, preds, digits=3))


# ROC CURVE

fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Engagement Drop Forecast")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()

print("\nROC curve saved as roc_curve.png")


# MISCLASSIFIED ANALYSIS

print("\nMisclassified sequences:")
for i in range(len(y)):
    if preds[i] != y[i]:
        print(
            f"Index {i} | True={int(y[i])} | "
            f"Pred={int(preds[i])} | Prob={probs[i]:.2f}"
        )
