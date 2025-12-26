#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Distributed helpers.

PATCHED VERSION:
This file disables all distributed / DDP functionality so that SlowFast
can run on a single CPU or single GPU machine without pytorchvideo
distributed dependencies.
"""

import logging
import pickle
import torch

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# PATCH: Disable distributed operations (single-process safe stubs)
# -------------------------------------------------------------------

def cat_all_gather(tensors, dim=0):
    """
    In distributed mode, this gathers tensors across processes.
    For single-process usage, just return the input.
    """
    return tensors


def get_local_process_group():
    return None


def get_local_rank():
    return 0


def get_local_size():
    return 1


def get_world_size():
    return 1


def get_rank():
    return 0


def is_distributed():
    return False


def init_distributed_training(cfg):
    """
    No-op for single-process usage.
    """
    return None


# -------------------------------------------------------------------
# Functions below are kept for API compatibility
# -------------------------------------------------------------------

def synchronize():
    """
    Synchronize processes. No-op in single-process mode.
    """
    return


def all_gather(tensors):
    """
    Gather tensors from all processes.
    In single-process mode, return input as-is.
    """
    return tensors


def all_reduce(tensor, average=True):
    """
    Reduce tensor across processes.
    In single-process mode, return input tensor.
    """
    return tensor


def save_on_master(*args, **kwargs):
    """
    Save only on master process.
    Single-process mode always saves.
    """
    torch.save(*args, **kwargs)


def load_state_dict(path):
    """
    Load a checkpoint safely.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
