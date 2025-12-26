import os
import csv
import torch
import numpy as np
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose, Lambda
from video_utils import load_video_clip
from utils import video_to_cache_name, save_feature


DEVICE = "cpu"


def load_model():
    model = slowfast_r50(pretrained=True)
    model.eval()
    model = model.to(DEVICE)
    return model


def preprocess(video, num_frames=32):
    """
    video: Tensor [T, H, W, C]
    Returns SlowFast input with fixed temporal sizes
    """

    # Ensure enough frames
    if video.shape[0] < num_frames:
        pad = num_frames - video.shape[0]
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

    video = video[:num_frames]  # (32, H, W, C)
    video = video.permute(3, 0, 1, 2)  # C T H W

    fast = video                       # 32 frames
    slow = video[:, ::4, :, :]         # 8 frames

    return [slow.unsqueeze(0), fast.unsqueeze(0)]


@torch.no_grad()
def extract_feature(model, video_path):
    video = load_video_clip(video_path)  # numpy array
    video = torch.tensor(video).float() / 255.0
    inputs = preprocess(video)

    inputs = [i.to(DEVICE) for i in inputs]
    features = model(inputs)

    return features.squeeze().cpu().numpy()


def main():
    model = load_model()
    os.makedirs("features", exist_ok=True)

    with open("sequences.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            for video_path in row[:3]:
                cache = video_to_cache_name(video_path)
                out = f"features/{cache}.npy"

                if os.path.exists(out):
                    continue

                print("Extracting:", video_path)
                feat = extract_feature(model, video_path)
                save_feature(feat, out)


if __name__ == "__main__":
    main()
