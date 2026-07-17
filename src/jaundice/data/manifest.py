"""Build a unified manifest (manifest.csv) across all jaundice datasets.

Columns: dataset_id, modality, population, label, label_name, orig_split, split, path,
         ref_ita, ref_ita_stratum, ref_skin_pixels, ref_skin_fraction

- neo_skin_core (dataset3==dataset4): neonatal SKIN. We assign a deterministic stratified
  train/val/test split (hash of filename -> stable bucket, done per-class).
- eye_roboflow (dataset1) / eye_stock (dataset2): adult SCLERA, already split. We keep orig_split.
  Flagged as extension-only (different modality + population than the core neonatal task).

Leakage note: the core set has NO baby/patient IDs, so we cannot guarantee that multiple photos
of the same infant do not straddle the split. This is logged as a WARNING; near-duplicate grouping
is a TODO (perceptual-hash clustering) before any headline result is trusted.

Run:  PYTHONPATH=src python -m jaundice.data.manifest --config configs/base.yaml
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow direct execution
from jaundice.utils import load_config, stable_bucket
from jaundice.data.skin_tone import reference_skin_tone

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# folder-name -> (label, label_name); case-insensitive substring match
POS_TOKENS = ("jaundice", "jaundiced")
NEG_TOKENS = ("normal",)


def _label_from_folder(name: str):
    n = name.lower()
    if any(t in n for t in POS_TOKENS):
        return 1, "jaundice"
    if any(t in n for t in NEG_TOKENS):
        return 0, "normal"
    return None


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            yield p


def _assign_split(rel_key: str, ratios: dict) -> str:
    b = stable_bucket(rel_key)
    if b < ratios["train"]:
        return "train"
    if b < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def build(cfg: dict) -> list[dict]:
    d = cfg["data"]
    ref_cfg = cfg.get("eval", {}).get("skin_tone_reference", {}) or {}
    ref_size = int(ref_cfg.get("image_size", 128))
    ref_min_skin = int(ref_cfg.get("min_skin_pixels", 20))
    ratios = d["split"]
    rows: list[dict] = []

    # ---- core neonatal skin (dataset3): assign our own stratified split ----
    core_root = Path(d["neo_skin_core"])
    if core_root.exists():
        for cls_dir in sorted(p for p in core_root.iterdir() if p.is_dir()):
            lab = _label_from_folder(cls_dir.name)
            if lab is None:
                continue
            label, label_name = lab
            for img in _iter_images(cls_dir):
                # stratify: bucket per-class so each class hits the target ratios
                split = _assign_split(f"{label_name}/{img.name}", ratios)
                ref = reference_skin_tone(img, size=ref_size, min_skin_pixels=ref_min_skin)
                rows.append(dict(dataset_id="neo_skin_core", modality="skin",
                                 population="neonate", label=label, label_name=label_name,
                                 orig_split="", split=split, path=str(img),
                                 ref_ita=f"{ref.ita:.6f}",
                                 ref_ita_stratum=ref.stratum,
                                 ref_skin_pixels=ref.skin_pixels,
                                 ref_skin_fraction=f"{ref.skin_fraction:.6f}"))
    else:
        print(f"[warn] core root missing: {core_root}", file=sys.stderr)

    # ---- eye/sclera extension sets (dataset1, dataset2): keep their own splits ----
    for ds_id, key in [("eye_roboflow", "eye_roboflow"), ("eye_stock", "eye_stock")]:
        root = d.get(key)
        if not root:
            continue
        root = Path(root)
        if not root.exists():
            print(f"[warn] {ds_id} root missing: {root}", file=sys.stderr)
            continue
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            orig = split_dir.name.lower()
            orig = {"valid": "val"}.get(orig, orig)
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                lab = _label_from_folder(cls_dir.name)
                if lab is None:
                    continue
                label, label_name = lab
                for img in _iter_images(cls_dir):
                    rows.append(dict(dataset_id=ds_id, modality="sclera",
                                     population="adult", label=label, label_name=label_name,
                                     orig_split=orig, split=orig, path=str(img),
                                     ref_ita="", ref_ita_stratum="",
                                     ref_skin_pixels="", ref_skin_fraction=""))
    return rows


def summarize(rows: list[dict]) -> None:
    print(f"\nTotal images: {len(rows)}")
    print(f"{'dataset_id':<14}{'modality':<9}{'split':<7}{'jaundice':>9}{'normal':>8}")
    key = lambda r: (r["dataset_id"], r["modality"], r["split"])
    counts = Counter((r["dataset_id"], r["modality"], r["split"], r["label"]) for r in rows)
    seen = sorted({(r["dataset_id"], r["modality"], r["split"]) for r in rows})
    for ds, mod, sp in seen:
        j = counts.get((ds, mod, sp, 1), 0); n = counts.get((ds, mod, sp, 0), 0)
        print(f"{ds:<14}{mod:<9}{sp:<7}{j:>9}{n:>8}")
    # leakage warning for core
    core_n = sum(1 for r in rows if r["dataset_id"] == "neo_skin_core")
    if core_n:
        print(f"\n[LEAKAGE WARNING] neo_skin_core has NO patient/baby IDs -> multiple photos of the "
              f"same infant may straddle train/val/test. TODO: perceptual-hash near-dup grouping "
              f"before trusting headline numbers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = build(cfg)
    out = Path(args.out or cfg["data"]["manifest_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset_id", "modality", "population", "label", "label_name", "orig_split", "split",
              "path", "ref_ita", "ref_ita_stratum", "ref_skin_pixels", "ref_skin_fraction"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    summarize(rows)


if __name__ == "__main__":
    main()
