import torch
from pytorchvideo.models.hub import slowfast_r50
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def extract_features(video_path):
    # Load pretrained SlowFast model
    model = slowfast_r50(pretrained=True).eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Could not open {video_path}")
        return None

    frames= []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # [3, 224, 224]
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f" No frames found in {video_path}")
        return None

    # Sample exactly 32 frames
    num_frames = 32
    if len(frames) > num_frames:
        idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        frames = [frames[i] for i in idx]
    elif len(frames) < num_frames:
        # Repeat frames if too short
        while len(frames) < num_frames:
            frames += frames
        frames = frames[:num_frames]

    video_tensor = torch.stack(frames)  # [T, 3, 224, 224]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]

    # Create Slow and Fast pathways
    fast_pathway = video_tensor
    slow_pathway = video_tensor[:, :, ::4, :, :]  # sample every 4th frame

    inputs = [slow_pathway, fast_pathway]

    with torch.no_grad():
        features = model(inputs)

    return features.cpu().numpy()


def main(csv_path="mini_train.csv"):
    df = pd.read_csv(csv_path, header=None)
    os.makedirs("features", exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        video_path, label = row[0], row[1]
        out_name = os.path.basename(video_path).replace(".mp4", ".npy")
        out_path = os.path.join("features", out_name)

        # Skip if already done
        if os.path.exists(out_path):
            continue

        features = extract_features(video_path)
        if features is not None:
            np.save(out_path, features)
            print(f" Saved features for {video_path}")
        else:
            print(f" Skipped {video_path} (no frames or error)")


if __name__ == "__main__":
    main("mini_train.csv")
