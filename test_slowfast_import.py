import torch
from pytorchvideo.models.hub import slowfast_r50

print("Loading pretrained SlowFast model…")
model = slowfast_r50(pretrained=True)
model.eval()

# print number of parameters to confirm load
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ SlowFast loaded successfully with {total_params:,} parameters.")
