"""Image-macro metrics with explicit empty-region semantics."""

import math

from .losses import ssim


def psnr(mse):
    return -10 * math.log10(mse) if mse > 0 else float("inf")


def restoration_metrics(pred, target):
    mse = float((pred - target).square().mean())
    return dict(psnr=psnr(mse), ssim=float(ssim(pred[None], target[None])[0]), rmse=math.sqrt(mse))


def region_metrics(pred, target, cloudy, mask, prefix):
    pixels = float(mask.sum())
    if pixels == 0:
        return {prefix + k: None for k in ("psnr", "rmse", "mae", "input_mae", "change_mae")}
    denom = pixels * pred.shape[0]
    mse = float(((pred - target).square() * mask).sum() / denom)
    return {
        prefix + "psnr": psnr(mse),
        prefix + "rmse": math.sqrt(mse),
        prefix + "mae": float(((pred - target).abs() * mask).sum() / denom),
        prefix + "input_mae": float(((cloudy - target).abs() * mask).sum() / denom),
        prefix + "change_mae": float(((pred - cloudy).abs() * mask).sum() / denom),
    }


def cloud_metrics(mask_prediction, mask):
    p, t = mask_prediction >= 0.5, mask >= 0.5
    tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
    return dict(
        mask_iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        mask_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        mask_precision=tp / (tp + fp) if tp + fp else None,
        mask_recall=tp / (tp + fn) if tp + fn else None,
    )


def average(rows):
    result = {"count": len(rows)}
    keys = sorted(
        {k for r in rows for k, v in r.items() if isinstance(v, (float, int)) or v is None}
    )
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (float, int))]
        result[key] = sum(values) / len(values) if values else None
        result[key + "_n"] = len(values)
    return result


def json_safe(value):
    # Strict JSON: exact-match infinite PSNR is represented explicitly, not capped.
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value