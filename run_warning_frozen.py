import numpy as np
import torch
import torch.nn as nn


# CONFIG

DEVICE = "cpu"
MODEL_PATH = "gru_forecast_attention.pth"
SEQ_INDEX = -1          # try 0, 1, 2, -1
WARNING_THRESHOLD = 0.6
TOP_K = 3


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
        logits = self.fc(context)
        return logits, attn_w


# LOAD DATA

X = np.load("data_forecast/X_forecast.npy")   # (N, T, D)
print("Loaded data:", X.shape)

SEQ_INDEX = SEQ_INDEX if SEQ_INDEX < len(X) else -1
x_input = torch.tensor(X[SEQ_INDEX], dtype=torch.float32).unsqueeze(0)


# LOAD & FREEZE MODEL

model = GRUAttention(input_dim=X.shape[-1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

for p in model.parameters():
    p.requires_grad = False   #  FREEZE

model.eval()


# INFERENCE

with torch.no_grad():
    logits, attn = model(x_input)
    prob = torch.sigmoid(logits).item()

print(f"\n Drop Probability: {prob:.2f}")

if prob > WARNING_THRESHOLD:
    print(" WARNING: Engagement likely to DROP in next 10s")
else:
    print(" Engagement likely STABLE")


# ATTENTION ANALYSIS

attn = attn.squeeze(0).cpu().numpy()
top_indices = attn.argsort()[-TOP_K:][::-1]

print("\nMost influential timesteps:")
for idx in top_indices:
    print(f"  Timestep {idx} → attention {attn[idx]:.3f}")
