"""Shared utilities: config loading (with `defaults:` merge), seeding, device selection."""
from __future__ import annotations
import os, random, hashlib
from pathlib import Path
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config. If it has `defaults: <file>`, load that first and deep-merge on top."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("defaults", None)
    if parent:
        parent_path = (path.parent / parent)
        cfg = _deep_merge(load_config(parent_path), cfg)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(pref: str = "auto") -> str:
    """auto -> cuda | mps | cpu. Explicit pref respected."""
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def stable_bucket(key: str) -> float:
    """Deterministic float in [0,1) from a string — for reproducible splits independent of file order."""
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF
