#!/usr/bin/env python3
# Copyright (c) Facebook, Inc.
# Inference-only patched version for feature extraction

"""
Head helpers (INFERENCE ONLY)

This file is intentionally stripped of:
- Detectron2
- ROIAlign
- Detection heads
- Training-only components

It is SAFE for CPU feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Basic classification head (used by SlowFast)
# ---------------------------------------------------------
class ResNetBasicHead(nn.Module):
    """
    Basic head for classification or feature extraction.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pool_size = pool_size

        self.avg_pool = nn.AvgPool3d(pool_size, stride=1)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        if num_classes > 0:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
        else:
            # Feature extraction mode
            self.projection = None

        self.act_func = act_func

    def forward(self, inputs):
        # inputs: list of tensors (SlowFast pathway outputs)
        assert isinstance(inputs, (list, tuple))

        x = inputs[0]
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.projection is not None:
            x = self.projection(x)

            if self.act_func == "softmax":
                x = F.softmax(x, dim=1)
            elif self.act_func == "sigmoid":
                x = torch.sigmoid(x)

        return x


# ---------------------------------------------------------
# Feature-only head (no classifier)
# --------------
