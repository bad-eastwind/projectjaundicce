"""Torch dataset + loaders built from manifest.csv.

Images are returned in [0,1] RGB (NOT ImageNet-normalized) because SCIN must estimate the scene
illuminant in linear-ish color space; ImageNet normalization happens INSIDE the model, after SCIN.
"""
from __future__ import annotations
import csv, collections
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T

from jaundice.utils import stable_bucket


def read_manifest(path: str, dataset_id: str = "neo_skin_core") -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if dataset_id and r["dataset_id"] != dataset_id:
                continue
            r["label"] = int(r["label"])
            r["domain"] = int(r.get("domain", 0) or 0)
            rows.append(r)
    return rows


def make_transforms(size: int, train: bool):
    if train:
        return T.Compose([
            T.ToImage(),
            T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.02),   # mild — SCIN handles illuminant, don't over-perturb color
            T.ToDtype(torch.float32, scale=True),  # -> [0,1]
        ])
    return T.Compose([
        T.ToImage(),
        T.Resize((size, size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


class JaundiceDataset(Dataset):
    def __init__(self, rows: list[dict], size: int, train: bool):
        self.rows = rows
        self.tf = make_transforms(size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return {"image": self.tf(img), "label": r["label"], "domain": r["domain"], "path": r["path"]}


def _cap_per_class(rows, cap):
    seen = collections.Counter(); out = []
    for r in rows:
        if seen[r["label"]] < cap:
            out.append(r); seen[r["label"]] += 1
    return out


def _balanced_sampler(labels):
    cnt = collections.Counter(labels)
    w = [1.0 / cnt[l] for l in labels]
    return WeightedRandomSampler(w, num_samples=len(labels), replacement=True)


def _loco_split(rows: list[dict], d: dict) -> dict:
    """Leave-one-domain-out: held-out discovered domain -> test; remaining domains -> train/val."""
    holdout = int(d.get("domain", {}).get("holdout", 0))
    val_frac = d["split"]["val"] / (d["split"]["train"] + d["split"]["val"])
    by = {"train": [], "val": [], "test": []}
    domains = set()
    for r in rows:
        domains.add(r["domain"])
        if r["domain"] == holdout:
            by["test"].append(r)
        else:
            by["val" if stable_bucket("loco/" + r["path"]) < val_frac else "train"].append(r)
    if len(domains) < 2:
        raise RuntimeError(f"leave_domain_out needs >=2 domains but found {domains}. "
                           f"Run: python -m jaundice.data.domains first.")
    print(f"[LOCO] holdout domain={holdout} -> test; domains present: {sorted(domains)}")
    return by


def build_loaders(cfg: dict, smoke: bool = False) -> dict:
    d, t = cfg["data"], cfg["train"]
    size = d["image_size"]
    rows = read_manifest(d["manifest_out"], "neo_skin_core")
    if d.get("split_mode", "random") == "leave_domain_out":
        by_split = _loco_split(rows, d)
    else:
        by_split = {"train": [], "val": [], "test": []}
        for r in rows:
            if r["split"] in by_split:
                by_split[r["split"]].append(r)
    if smoke:
        cap = cfg.get("smoke", {}).get("max_per_class", 8)
        by_split = {s: _cap_per_class(v, cap) for s, v in by_split.items()}

    loaders = {}
    for s in ("train", "val", "test"):
        ds = JaundiceDataset(by_split[s], size, train=(s == "train"))
        if s == "train":
            sampler = _balanced_sampler([r["label"] for r in by_split[s]])
            loaders[s] = DataLoader(ds, batch_size=t["batch_size"], sampler=sampler,
                                    num_workers=t["num_workers"], drop_last=False)
        else:
            loaders[s] = DataLoader(ds, batch_size=t["batch_size"], shuffle=False,
                                    num_workers=t["num_workers"])
        print(f"[loader] {s}: {len(ds)} imgs "
              f"(jaundice={sum(r['label'] for r in by_split[s])}, "
              f"normal={sum(1-r['label'] for r in by_split[s])})")
    return loaders
