"""Pair-safe preprocessing and persisted, leakage-checked dataset manifests."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class PseudoConfig:
    noise_floor: float = 0.02
    scale: float = 0.5
    threshold: float = 0.15
    sigma: float = 1.0

    def __post_init__(self):
        if (
            not 0 <= self.noise_floor < 1
            or self.scale <= 0
            or not 0 <= self.threshold <= 1
            or self.sigma <= 0
        ):
            raise ValueError("Invalid pseudo-supervision parameters")


def pseudo_labels(cloudy, clear, config):
    """Fixed training-calibrated scale; never normalize each test image by its own max."""
    single = cloudy.ndim == 3
    if single:
        cloudy, clear = cloudy[None], clear[None]
    delta = (cloudy - clear).abs().mean(1, keepdim=True)
    radius = max(1, int(3 * config.sigma))
    axis = torch.arange(-radius, radius + 1, device=delta.device, dtype=delta.dtype)
    g = torch.exp(-axis.square() / (2 * config.sigma**2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    smoothed = F.conv2d(F.pad(delta, (radius,) * 4, mode="replicate"), kernel)
    severity = ((smoothed - config.noise_floor).clamp_min(0) / config.scale).clamp(0, 1)
    mask = (severity > config.threshold).to(severity.dtype)
    return (severity[0], mask[0]) if single else (severity, mask)


def read_rgb(path):
    with Image.open(path) as image:
        # Reject multispectral/high-depth data rather than silently corrupt radiometry.
        if image.mode != "RGB":
            raise ValueError(
                f"{path}: expected 8-bit RGB, found {image.mode}. Export aligned RGB first (see docs/DATA.md)."
            )
        return torch.from_numpy(np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1), dtype=np.float32) / np.float32(255))


def index_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")
    indexed = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in EXTENSIONS:
            key = str(p.relative_to(folder).with_suffix(""))
            if key in indexed:
                raise ValueError(f"Ambiguous image stem {key}")
            indexed[key] = p.resolve()
    if not indexed:
        raise ValueError(f"No images in {folder}")
    return indexed


def pair_records(cloudy, clear, name, masks=None, groups=None):
    c, t = index_images(cloudy), index_images(clear)
    m = index_images(masks) if masks else {}
    if c.keys() != t.keys():
        raise ValueError(
            f"Unmatched pairs: cloudy-only {sorted(c.keys() - t.keys())[:5]}, clear-only {sorted(t.keys() - c.keys())[:5]}"
        )
    if masks and m.keys() != c.keys():
        raise ValueError("Provided masks must match every cloudy image")
    records = []
    for key in c:
        with Image.open(c[key]) as a, Image.open(t[key]) as b:
            if a.size != b.size:
                raise ValueError(f"Pair is not aligned in size: {key}")
        if groups is not None and key not in groups:
            raise ValueError(f"Missing scene group for {key}")
        records.append(
            dict(
                id=f"{name}/{key}",
                dataset=name,
                cloudy=str(c[key]),
                clear=str(t[key]),
                mask=str(m[key]) if key in m else None,
                group=f"{name}/{groups[key] if groups is not None else key}",
            )
        )
    return records


def validate_records(records):
    seen_ids, seen_paths, seen_content, group_splits = set(), {}, {}, {}
    for row in records:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Split must be train, validation, or test")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate sample ID {row['id']}")
        seen_ids.add(row["id"])
        group = row.get("group", row["id"])
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError(f"Scene group leaks across splits: {group}")
        group_splits[group] = row["split"]
        shapes = {}
        for field in ("cloudy", "clear"):
            path = str(Path(row[field]).resolve())
            if path in seen_paths and seen_paths[path] != row["split"]:
                raise ValueError(f"Image path leaks across splits: {path}")
            seen_paths[path] = row["split"]
            tensor = read_rgb(path)
            shapes[field] = tensor.shape
            digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if digest in seen_content and seen_content[digest] != row["split"]:
                raise ValueError(f"Duplicate image content leaks across splits: {path}")
            seen_content[digest] = row["split"]
        if shapes["cloudy"] != shapes["clear"]:
            raise ValueError(f"Mismatched pair dimensions: {row['id']}")


def split_records(records, seed=42, fractions=(0.7, 0.15, 0.15)):
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Three positive split fractions must sum to one")
    rng = random.Random(seed)
    for dataset in sorted({r["dataset"] for r in records}):
        subset = [r for r in records if r["dataset"] == dataset]
        groups = sorted({r["group"] for r in subset})
        if len(groups) < 3:
            raise ValueError("At least three independent groups per dataset are required")
        rng.shuffle(groups)
        ntrain = min(len(groups) - 2, max(1, int(len(groups) * fractions[0])))
        nval = min(len(groups) - ntrain - 1, max(1, int(len(groups) * fractions[1])))
        assignments = {
            g: "train" if i < ntrain else "validation" if i < ntrain + nval else "test"
            for i, g in enumerate(groups)
        }
        for row in subset:
            row["split"] = assignments[row["group"]]
    validate_records(records)
    return records


def write_manifest(path, records, seed=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(version=1, seed=seed, records=records), indent=2))


class PairedDataset(Dataset):
    def __init__(self, manifest, split, size=256, augment=False, pseudo=None):
        path = Path(manifest).resolve()
        payload = json.loads(path.read_text())
        records = payload["records"]
        for row in records:
            for key in ("cloudy", "clear", "mask"):
                if row.get(key):
                    row[key] = str((path.parent / row[key]).resolve())
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"Manifest has no {split} images")
        if size is not None and size < 16:
            raise ValueError("Image size must be >=16")
        self.size, self.augment, self.pseudo = size, augment, pseudo or PseudoConfig()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        cloudy, clear = read_rgb(row["cloudy"]), read_rgb(row["clear"])
        if cloudy.shape != clear.shape:
            raise ValueError(f"Mismatched pair: {row['id']}")
        mask = None
        if row.get("mask"):
            mask = read_mask(row["mask"])
            if mask.shape[-2:] != cloudy.shape[-2:]:
                raise ValueError(f"Mismatched mask dimensions: {row['id']}")
            mask = (mask > 0.5).float()
        items = [cloudy, clear] + ([mask] if mask is not None else [])
        if self.size:
            items = [F.interpolate(im[None], size=(self.size, self.size),
                     mode="nearest" if i == 2 else "bilinear",
                     **({} if i == 2 else {"align_corners": False, "antialias": True}))[0]
                     for i, im in enumerate(items)]
        if self.augment:
            for axis in (-1, -2):
                if random.random() < 0.5:
                    items = [im.flip(axis) for im in items]
            # Right-angle rotations avoid interpolating or misaligning supervision.
            turns = random.randrange(4)
            items = [torch.rot90(im, turns, (-2, -1)) for im in items]
        cloudy, clear = items[:2]
        severity, pseudo_mask = pseudo_labels(cloudy, clear, self.pseudo)
        return dict(
            cloudy=cloudy,
            clear=clear,
            severity=severity,
            mask=items[2] if mask is not None else pseudo_mask,
            mask_source="dataset" if mask is not None else "pseudo",
            stable_clear=(1 - (items[2] if mask is not None else pseudo_mask)) *
                         ((cloudy-clear).abs().mean(0,keepdim=True) <= self.pseudo.noise_floor).float(),
            id=row["id"],
            group=row.get("group",row["id"]),
            dataset=row["dataset"],
        )


def read_mask(path):
    """Only unambiguous binary masks. White/one means cloud; no silent palette conversion."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB") if im.mode == "P" else im).copy()
    if a.ndim == 3:
        if a.shape[2] != 3 or not (np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,0],a[...,2])):
            raise ValueError(f"Colored mask requires explicit class mapping: {path}")
        a = a[...,0]
    if a.ndim != 2 or not set(np.unique(a).tolist()).issubset({0,1,255,False,True}):
        raise ValueError(f"Expected binary 0/1 or 0/255 mask: {path}")
    return torch.from_numpy((a>0).astype(np.float32))[None]


def content_manifest_hash(path):
    """Location-independent hash includes decoded cloudy/clear/mask content and membership."""
    path=Path(path).resolve(); rows=json.loads(path.read_text())["records"]; canonical=[]
    for r in sorted(rows,key=lambda r:r["id"]):
        item={k:r.get(k) for k in ("id","dataset","split","group")}
        for field in ("cloudy","clear","mask"):
            if r.get(field):
                f=(path.parent/r[field]).resolve()
                t=read_mask(f) if field=="mask" else read_rgb(f)
                item[field]=hashlib.sha256(str(tuple(t.shape)).encode()+t.numpy().tobytes()).hexdigest()
            else: item[field]=None
        canonical.append(item)
    return hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()