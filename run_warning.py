import numpy as np
import torch
import torch.nn as nn

DEVICE =  "cpu"
SEQ_INDEX =-1     # change this to test different sequences
THRESHOLD = 0.6
TOP_K = 3

X = np.load("data_forecast/X_test.npy")

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
model.load_state_dict(torch.load("gru_forecast_attention.pth", map_location=DEVICE))
model.eval()

x = torch.tensor(X[SEQ_INDEX], dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits, attn = model(x)
    prob = torch.sigmoid(logits).item()

print(f"\n Drop Probability: {prob:.2f}")

if prob > THRESHOLD:
    print(" WARNING: Engagement likely to DROP in next 10s")
else:
    print(" Engagement likely STABLE")

# Attention analysis
attn = attn.squeeze(0).cpu().numpy()
top_idx = attn.argsort()[-TOP_K:][::-1]

print("\n Most influential timesteps:")
for i in top_idx:
    print(f"Timestep {i} → attention {attn[i]:.3f}")
