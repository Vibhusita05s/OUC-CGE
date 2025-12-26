# video_utils.py
import cv2
import numpy as np
from utils import video_to_cache_name, save_feature, load_feature_if_exists

def load_video_clip(video_path, resize=(224, 224), max_frames=64, cache=True):
    """
    Loads a video as a numpy array. Uses caching if desired.
    """
    cache_path = video_to_cache_name(video_path) + ".npy"
    
    #  loading from cache
    if cache:
        cached = load_feature_if_exists(cache_path)
        if cached is not None:
            return cached
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    
    clip_array = np.array(frames)
    
    # Save to cache
    if cache:
        save_feature(clip_array, cache_path)
    
    return clip_array
