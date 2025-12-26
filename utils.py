#file created to small helpers to save/load cached feature vectors using a stable name (so re-extraction is avoided).
# utils.py
import os
import numpy as np
import hashlib

def ensure_dir(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def video_to_cache_name(video_path):
    """Generate a unique cache file name for a video."""
    h = hashlib.md5(video_path.encode()).hexdigest()
    return os.path.join("cache", h)

def save_feature(feature, path):
    """Save a numpy feature array to disk."""
    ensure_dir(os.path.dirname(path))
    np.save(path, feature)

def load_feature_if_exists(path):
    """Load a feature array from disk if it exists."""
    if os.path.exists(path):
        return np.load(path)
    return None
