import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import PairedDataset, PseudoConfig, read_rgb, validate_records, content_manifest_hash
from .losses import HybridLoss, LossConfig, Perceptual
from .metrics import (
    average,
    cloud_metrics,
    json_safe,
    region_metrics,
    restoration_metrics,
)
from .model import CloudRemovalNet, ModelConfig


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(data), indent=2, allow_nan=False))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(request):
    if request != "auto":
        return torch.device(request)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def image_of(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))


def save_panel(path, batch, out, index=0):
    entries = [("Cloudy", batch["cloudy"]), ("Mask target", batch["mask"])]
    if "mask" in out:
        entries.append(("Predicted mask", out["mask"]))
    entries.append(("Pseudo severity", batch["severity"]))
    if "severity" in out:
        entries.append(("Predicted severity", out["severity"]))
    if "restored" in out:
        entries.append(("Restored", out["restored"]))
    entries.append(("Clear reference", batch["clear"]))
    w, h = 180, 180
    panel = Image.new("RGB", (w * len(entries), h + 34), "#101d2b")
    draw = ImageDraw.Draw(panel)
    for i, (label, value) in enumerate(entries):
        panel.paste(image_of(value[index]).resize((w, h)), (i * w, 34))
        draw.text((i * w + 8, 10), label, fill="white")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def checked_manifest(path):
    p = Path(path).resolve()
    records = json.loads(p.read_text())["records"]
    for r in records:
        for key in ("cloudy", "clear", "mask"):
            if r.get(key):
                r[key] = str((p.parent / r[key]).resolve())
    validate_records(records)
    return content_manifest_hash(p)


def write_curves(path, history):
    # Dependency-free scalable figure; actual recorded values, no fabricated results.
    if not history:
        return
    values = [r["train_loss"] for r in history] + [r["validation_loss"] for r in history]
    hi, lo = max(values), min(values)

    def points(key):
        return " ".join(
            f"{50 + i * 600 / max(1, len(history) - 1):.1f},{260 - (r[key] - lo) * 210 / max(hi - lo, 1e-9):.1f}"
            for i, r in enumerate(history)
        )

    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="320" viewBox="0 0 720 320"><rect width="720" height="320" fill="white"/><text x="50" y="25" font-size="18">Training (blue) and validation (orange) loss</text><path d="M50 45 V270 H670" fill="none" stroke="#888"/><polyline points="{points("train_loss")}" fill="none" stroke="#2476ce" stroke-width="2"/><polyline points="{points("validation_loss")}" fill="none" stroke="#e48620" stroke-width="2"/><text x="50" y="300">Epoch 1 to {len(history)}; loss range {lo:.4f} to {hi:.4f}</text></svg>'
    )


def train(config, resume=None):
    cfg = dict(config)
    if cfg.get("epochs", 0) < 1 or cfg.get("batch_size", 1) < 1:
        raise ValueError("Positive epochs and batch_size required")
    seed_all(cfg.get("seed", 42))
    torch.set_num_threads(cfg.get("threads", 4))
    device = device_for(cfg.get("device", "auto"))
    manifest_hash = checked_manifest(cfg["manifest"])
    pseudo = PseudoConfig(**cfg.get("pseudo", {}))
    train_data = PairedDataset(cfg["manifest"], "train", cfg.get("size", 256), True, pseudo)
    val_data = PairedDataset(cfg["manifest"], "validation", cfg.get("size", 256), False, pseudo)
    outdir = Path(cfg["output"])
    if not resume and (outdir / "last.pt").exists():
        raise ValueError("Run already exists; use --resume or choose a new output directory")
    outdir.mkdir(parents=True, exist_ok=True)
    model = CloudRemovalNet(ModelConfig(**cfg.get("model", {}))).to(device)
    criterion = HybridLoss(LossConfig(**cfg.get("loss", {}))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 2e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    history, start, best = [], 0, -float("inf")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume manifest differs from the saved experiment")
        for key in (
            "model",
            "loss",
            "pseudo",
            "size",
            "seed",
            "batch_size",
            "accumulation_steps",
            "learning_rate",
            "weight_decay", "severity_bins", "code_sha256", "epochs", "data_protocol", "hsformer_provenance",
        ):
            if cfg.get(key) != state["config"].get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start, best, history = state["epoch"] + 1, state["best_psnr"], state["history"]
        if cfg["epochs"] <= start:
            raise ValueError("epochs must exceed completed epochs when resuming")
    save_json(outdir / "config.json", cfg)
    save_json(
        outdir / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            device=str(device),
            manifest_sha256=manifest_hash,
            parameters=sum(p.numel() for p in model.parameters()),
        ),
    )
    run_started = time.monotonic()
    session_epochs = cfg.get("epochs_per_session",cfg["epochs"])
    for epoch in range(start, min(cfg["epochs"],start+session_epochs)):
        epoch_started=time.monotonic()
        # Epoch-local seeds give reproducible shuffled batches and augmentation on resume.
        seed_all(cfg.get("seed", 42) + epoch)
        loader = DataLoader(
            train_data, batch_size=cfg.get("batch_size", 4), shuffle=True, num_workers=0
        )
        model.train()
        train_sum, count, term_sums = 0.0, 0, {}
        accumulation=int(cfg.get("accumulation_steps",1))
        if accumulation<1: raise ValueError("Positive accumulation_steps required")
        optimizer.zero_grad(set_to_none=True)
        group_count=0
        for batch_index,batch in enumerate(loader):
            batch = to_device(batch, device)
            prediction = model(batch["cloudy"])
            loss, terms = criterion(prediction, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            n = batch["cloudy"].shape[0]
            # Sample-weighted accumulation handles the last short batch/group.
            (loss if accumulation==1 else loss*n).backward()
            group_count+=n
            if (batch_index+1)%accumulation==0 or batch_index+1==len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None and accumulation>1: parameter.grad.div_(group_count)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); group_count=0
            train_sum += float(loss.detach()) * n
            count += n
            for k, v in terms.items():
                term_sums[k] = term_sums.get(k, 0.0) + float(v.detach()) * n
        model.eval()
        val_losses, val_scores = [], []
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(val_data, batch_size=1)):
                batch = to_device(batch, device)
                prediction = model(batch["cloudy"])
                loss, _ = criterion(prediction, batch)
                val_losses.append(float(loss))
                val_scores.append(
                    restoration_metrics(prediction["restored"][0].clamp(0, 1), batch["clear"][0])[
                        "psnr"
                    ]
                )
                if i == 0:
                    save_panel(outdir / "validation_latest.png", batch, prediction)
        row = dict(
            epoch=epoch + 1,
            train_loss=train_sum / count,
            validation_loss=sum(val_losses) / len(val_losses),
            validation_psnr=sum(val_scores) / len(val_scores),
            **{f"train_{k}": v / count for k, v in term_sums.items()},
        )
        row["epoch_seconds"]=time.monotonic()-epoch_started
        history.append(row)
        recent_seconds=sum(r["epoch_seconds"] for r in history[-3:])/len(history[-3:])
        row["estimated_remaining_model_hours"]=max(0,cfg["epochs"]-epoch-1)*recent_seconds/3600
        improved = row["validation_psnr"] > best
        best = max(best, row["validation_psnr"])
        checkpoint = dict(
            config=cfg,
            manifest_sha256=manifest_hash,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            best_psnr=best,
            history=history,
            content_records=json.loads(Path(cfg["manifest"]).read_text())["records"],
        )
        torch.save(checkpoint, outdir / "last.tmp")
        (outdir / "last.tmp").replace(outdir / "last.pt")
        if improved:
            torch.save(checkpoint, outdir / "best.tmp")
            (outdir / "best.tmp").replace(outdir / "best.pt")
        save_json(outdir / "history.json", history)
        write_curves(outdir / "training_curves.svg", history)
        print(json.dumps(json_safe(row)), flush=True)
        if time.monotonic()-run_started + row["epoch_seconds"] > cfg.get("session_seconds",1e20):
            break
    completed=history[-1]["epoch"] >= cfg["epochs"]
    save_json(outdir/"status.json",dict(completed=completed,epochs_done=history[-1]["epoch"],target=cfg["epochs"]))
    if not completed:
        return None
    return str(outdir / "best.pt")


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CloudRemovalNet(ModelConfig(**state["config"].get("model", {}))).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def evaluate(
    checkpoint,
    manifest,
    output,
    device="auto",
    split="test",
    cross_dataset=False,
    with_lpips=True,
    panels=8,
    predictions=None,
    method=None,
    size_override=None,
    quantize=False,
):
    device = device_for(device)
    torch.set_num_threads(4)
    model, state = load_model(checkpoint, device)
    manifest_hash = checked_manifest(manifest)
    if cross_dataset:
        # Also reject reusing training images under a different cross-dataset manifest.
        train_manifest = Path(state["config"]["manifest"])
        if not train_manifest.exists():
            raise ValueError(
                "Original training manifest required to verify cross-dataset isolation; restore it at the recorded path"
            )
        original = json.loads(train_manifest.read_text())["records"]
        external = json.loads(Path(manifest).read_text())["records"]
        mixed = []
        for prefix, rows, base in (
            ("original", original, train_manifest.resolve().parent),
            ("external", external, Path(manifest).resolve().parent),
        ):
            for r in rows:
                row = dict(
                    r, id=prefix + "/" + r["id"], split="train" if prefix == "original" else "test"
                )
                for key in ("cloudy", "clear", "mask"):
                    if row.get(key):
                        row[key] = str((base / row[key]).resolve())
                mixed.append(row)
        validate_records(mixed)
    elif manifest_hash != state["manifest_sha256"]:
        raise ValueError(
            "In-domain evaluation requires the original manifest. Use --cross-dataset for a different dataset"
        )
    cfg = state["config"]
    eval_size=cfg.get("size",256) if size_override is None else size_override
    dataset = PairedDataset(
        manifest, split, eval_size, False, PseudoConfig(**cfg.get("pseudo", {}))
    )
    perceptual = Perceptual().to(device) if with_lpips else None
    thresholds = cfg.get("severity_bins", [0.33, 0.66])
    if len(thresholds) != 2 or not 0 <= thresholds[0] < thresholds[1] <= 1:
        raise ValueError("severity_bins must be two increasing values in [0,1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i, batch in enumerate(DataLoader(dataset, batch_size=1)):
            batch = to_device(batch, device)
            if predictions is not None:
                prediction_path = Path(predictions) / (batch["id"][0] + ".png")
                restored_image = read_rgb(prediction_path)[None].to(device)
                if restored_image.shape != batch["clear"].shape:
                    raise ValueError(
                        f"External output must match evaluation RGB dimensions: {prediction_path}"
                    )
                pred = {"restored": restored_image}
                elapsed = None
            else:
                if i == 0:
                    model(batch["cloudy"])  # Warm-up excluded from latency.
                sync(device)
                started = time.perf_counter()
                pred = model(batch["cloudy"])
                sync(device)
                elapsed = (time.perf_counter() - started) * 1000
            restored, target, cloudy, mask = (
                pred["restored"][0].clamp(0, 1),
                batch["clear"][0],
                batch["cloudy"][0],
                batch["mask"][0],
            )
            if quantize:
                restored=(restored*255).round()/255
            evidence = batch["severity"][0]
            # Conditional mean severity avoids confusing cloud coverage with opacity.
            active = evidence > dataset.pseudo.threshold
            severity = float(evidence[active].mean()) if active.any() else 0.0
            group = (
                "low"
                if severity < thresholds[0]
                else "medium"
                if severity < thresholds[1]
                else "high"
            )
            row = dict(
                id=batch["id"][0], group=batch["group"][0],dataset=batch["dataset"][0],
                mask_source=batch["mask_source"][0],
                severity_group=group,
                mean_cloud_severity=severity,
                cloud_coverage=float(mask.mean()),
                inference_ms=elapsed,
            )
            row.update(restoration_metrics(restored, target))
            row.update({"input_" + k: v for k, v in restoration_metrics(cloudy, target).items()})
            row["lpips"] = (
                float(perceptual(restored[None], target[None])[0]) if perceptual else None
            )
            row.update(region_metrics(restored, target, cloudy, mask, "cloud_"))
            row.update(region_metrics(restored, target, cloudy, 1 - mask, "clear_"))
            row.update(region_metrics(restored,target,cloudy,batch["stable_clear"][0],"stable_clear_"))
            row["psnr_gain"]=row["psnr"]-row["input_psnr"]
            row["mae"]=float((restored-target).abs().mean())
            row["input_mae"]=float((cloudy-target).abs().mean())
            row["edge_mae"]=float(((restored[:,1:,:]-restored[:,:-1,:])-(target[:,1:,:]-target[:,:-1,:])).abs().mean()
                                    +((restored[:,:,1:]-restored[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])).abs().mean())/2
            row["identity_change_mae"]=float((model(target[None])["restored"][0]-target).abs().mean()) if predictions is None else None
            row["coverage_group"]="low" if float(mask.mean())<1/3 else "medium" if float(mask.mean())<2/3 else "high"
            if "preclip" in pred:
                row["preclip_fraction"]=float(((pred["preclip"]<0)|(pred["preclip"]>1)).float().mean())
                row["edit_bound_violation"]=float(((restored-cloudy).abs()-pred["confidence"][0]).clamp_min(0).max())
            if "mask" in pred:
                row.update(cloud_metrics(pred["mask"][0], mask))
            if "severity" in pred:
                row["pseudo_severity_mae"] = float((pred["severity"][0] - evidence).abs().mean())
            rows.append(row)
            if i < panels:
                save_panel(output / f"panel_{i:04d}.png", batch, pred)
    report = dict(
        protocol="cross-dataset generalization; no fine-tuning"
        if cross_dataset
        else f"in-domain {split}",
        checkpoint=str(Path(checkpoint).resolve()),
        model={"variant": method or "external_predictions"}
        if predictions is not None
        else model.config_dict(),
        parameters=None if predictions is not None else sum(p.numel() for p in model.parameters()),
        device=str(device),
        latency_ms_median=float(np.median([r["inference_ms"] for r in rows])) if predictions is None else None,
        latency_ms_p95=float(np.quantile([r["inference_ms"] for r in rows],.95)) if predictions is None else None,
        peak_cuda_memory_mb=float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else None,
        external_predictions=str(predictions) if predictions is not None else None,
        manifest_sha256=manifest_hash,
        size=eval_size,
        quantized=quantize,
        config=cfg,
        pseudo=asdict(dataset.pseudo),
        severity_bins=thresholds,
        lpips_enabled=with_lpips,
        metric_protocol="Image-macro; RGB [0,1] clipped restoration; SSIM Gaussian valid windows; empty regions null; infinite PSNR string Infinity. Severity is pseudo-supervision agreement.",
        overall=average(rows),
        by_severity={
            k: average([r for r in rows if r["severity_group"] == k])
            for k in ("low", "medium", "high")
        },
        by_dataset={k: average([r for r in rows if r["dataset"]==k]) for k in sorted({r["dataset"] for r in rows})},
        by_coverage={k: average([r for r in rows if r["coverage_group"]==k]) for k in ("low","medium","high")},
        by_mask_source={
            k: average([r for r in rows if r["mask_source"] == k]) for k in ("dataset", "pseudo")
        },
        worst_psnr_ids=[r["id"] for r in sorted(rows, key=lambda r: r["psnr"])[:10]],
        samples=rows,
    )
    save_json(output / "metrics.json", report)
    fields = sorted({k for r in rows for k in r})
    with (output / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    return report


def infer(checkpoint, input_path, output, device="auto"):
    device = device_for(device)
    model, _ = load_model(checkpoint, device)
    source = Path(input_path)
    from .data import EXTENSIONS

    paths = (
        [source]
        if source.is_file()
        else sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    )
    if not paths:
        raise ValueError("No input images found")
    with torch.no_grad():
        for path in paths:
            x = read_rgb(path)[None].to(device)
            pred = model(x)
            relative = Path(path.name) if source.is_file() else path.relative_to(source)
            folder = Path(output) / relative.parent / relative.name
            folder.mkdir(parents=True, exist_ok=True)
            for key in ("restored", "mask", "severity", "confidence"):
                if key in pred:
                    image_of(pred[key][0]).save(folder / f"{key}.png")
            np.save(folder / "residual.npy", pred["residual"][0].cpu().numpy())
    return len(paths)