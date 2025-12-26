#!/usr/bin/env python3
# Copyright (c) Facebook, Inc.

"""
Model registry (inference-only).

PATCH:
- Training-only models are disabled
- Only SlowFast inference model is registered
"""

# Import model registry
from .build import build_model, MODEL_REGISTRY  # noqa

# IMPORTANT:
# This import registers "SlowFast" into MODEL_REGISTRY
from .video_model_builder import SlowFast  # noqa
