import os
import csv
import numpy as np

from utils import video_to_cache_name

CSV_PATH = "sequences.csv"
FEATURE_DIR = "features"
OUT_DIR = "data"

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "low": 0,
    "mid": 1,
    "high": 2,
}

def main():
    X, y = [], []

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if len(row) < 2:
                continue

            print(f"\n[ROW {row_idx}] {row}")

            video_paths = row[:-1]
            label_raw = row[-1].strip().lower()

            if label_raw not in LABEL_MAP:
                print(f"[SKIP] Invalid label: {label_raw}")
                continue

            label = LABEL_MAP[label_raw]

            sequence_feats = []
            valid = True

            for vp in video_paths:
                cache_name = video_to_cache_name(vp)
                feat_path = os.path.join(FEATURE_DIR, cache_name + ".npy")

                if not os.path.exists(feat_path):
                    print(f"[SKIP] Missing feature: {feat_path}")
                    valid = False
                    break

                feat = np.load(feat_path)
                sequence_feats.append(feat)

            if not valid:
                continue

            sequence_feats = np.stack(sequence_feats, axis=0)
            X.append(sequence_feats)
            y.append(label)

            print(f"[OK] Sequence shape {sequence_feats.shape}, label {label}")

    if len(X) == 0:
        print("\n No valid sequences found.")
        return

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)

    print("\n Phase B completed")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()
