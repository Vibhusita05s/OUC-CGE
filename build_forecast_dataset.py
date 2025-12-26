import os
import numpy as np


# CONFIG

IN_DIR = "data"                 # from Phase B
OUT_DIR = "data_forecast"       # new dataset
SEQUENCE_LEN = 3                # previous 30s (example)
PRED_HORIZON = 1                # next 10s
os.makedirs(OUT_DIR, exist_ok=True)


# LOAD DATA
# 
X = np.load(os.path.join(IN_DIR, "X.npy"))  # (N, T, D)
y = np.load(os.path.join(IN_DIR, "y.npy"))  # (N,)

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# 
# BUILD FORECAST DATASET
# 
X_f, y_f = [], []

for i in range(len(X) - PRED_HORIZON):
    current_feat = X[i]
    current_label = y[i]
    future_label = y[i + PRED_HORIZON]

    #  EARLY WARNING LABEL
    drop = 1 if future_label < current_label else 0

    X_f.append(current_feat)
    y_f.append(drop)

X_f = np.array(X_f)
y_f = np.array(y_f)

# 
# SAVE
# 
np.save(os.path.join(OUT_DIR, "X_forecast.npy"), X_f)
np.save(os.path.join(OUT_DIR, "y_forecast.npy"), y_f)

print("\n Phase C dataset built (EARLY WARNING)")
print("X_forecast:", X_f.shape)
print("y_forecast:", y_f.shape)
print("Drop ratio:", y_f.mean())
