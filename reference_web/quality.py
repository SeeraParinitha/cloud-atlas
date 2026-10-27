"""Conservative RGB screens. These cannot establish truth beneath opaque cloud."""
import numpy as np
from scipy.ndimage import binary_erosion, label, sobel


def largest_region(mask):
    regions, count = label(mask)
    return int(np.bincount(regions.ravel())[1:].max()) if count else 0


def residual_screen(prediction, reference, valid):
    pred = np.clip(prediction, 0, 1)
    ref = np.clip(reference, 0, 1)
    delta = pred - ref
    # Compare every comparable pixel, including pixels missed by the target mask.
    display_delta = (pred**(1/2.2) - ref**(1/2.2)).mean(0)
    bright = valid & (delta.min(0) > .025) & (display_delta > .075)
    dark = valid & (delta.max(0) < -.025) & (display_delta < -.075)
    limit = max(9, int(valid.size*.0005))
    bright_size, dark_size = largest_region(bright), largest_region(dark)
    return {'accepted': bright_size < limit and dark_size < limit,
            'bright_region_pixels': bright_size, 'dark_region_pixels': dark_size,
            'region_limit_pixels': limit, 'notice': 'RGB difference alarms can also detect real surface changes.'}


def terrain_screen(target, reference):
    """Reject visible colour/structure changes; never warp or recolour a photo."""
    clear = target['usable'] & ~target['risk'] & ~reference['risk']
    support = binary_erosion(clear, iterations=3)
    count = int(support.sum())
    report = {'accepted': True, 'clear_overlap_pixels': count, 'registration_verified': False}
    if count < max(64, int(clear.size*.1)):
        return {**report, 'reason': 'insufficient_visible_terrain',
                'notice': 'Only the geographic grid can be checked where the target is obscured.'}
    a, b = target['rgb'][:, support], reference['rgb'][:, support]
    difference = np.abs(a - b).mean(0)
    report.update(median_rgb_difference=float(np.median(difference)),
                  p90_rgb_difference=float(np.quantile(difference, .9)))
    if report['median_rgb_difference'] > .025 or report['p90_rgb_difference'] > .06:
        return {**report, 'accepted': False, 'reason': 'visible_terrain_colour_changed'}
    def edges(rgb):
        gray = rgb.mean(0)
        return np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    x, y = edges(target['rgb']), edges(reference['rgb'])
    if x[support].std() < 1e-4 or y[support].std() < 1e-4:
        return {**report, 'reason': 'low_texture_grid_and_colour_only'}
    scores = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = np.roll(y, (dy, dx), (0, 1))[support]
            corr = float(np.corrcoef(x[support], shifted)[0, 1]) if shifted.std() > 1e-6 else -1.
            scores.append((corr, dy, dx))
    best = max(scores)
    zero = next(score for score, dy, dx in scores if dy == dx == 0)
    accepted = zero >= .5 and not (best[1:] != (0, 0) and best[0] > zero+.05)
    return {**report, 'accepted': accepted, 'reason': 'visible_structure_check',
            'zero_shift_correlation': zero, 'best_shift': list(best[1:]),
            'notice': 'Heuristic integer-shift screen, not verified subpixel registration.'}
