"""Run the original HSFormer/prior, then optionally show one full observation."""
import json
import shutil
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
from PIL import Image
from rasterio.io import MemoryFile

from sentinel_triplets.data import days
from sentinel_triplets.full_area import png
from sentinel_triplets.web_service import TemporalService
from temporal_prior.real_review import read_scene
from .quality import residual_screen, terrain_screen, largest_region
from .coverage import CoverageClient, CoverageError, check_coverage, MIN_COVERAGE, MAX_CHECKS


# Keep both model priors and displayed observations within the same 60-day season window.
SEARCH_WINDOWS = ((1, 21, 12), (22, 60, 12), (-30, -1, 12))
MAX_REFERENCE_AGE = 60


def assess_model(target, prediction, prior=None):
    """Conservative visual alarms, not a calibrated accuracy/confidence score."""
    affected = target['cloud'] & target['usable']
    report = {'method': 'Whole-image decision from coverage and RGB residual alarms',
              'verified_accuracy': False, 'reasons': [], 'thresholds': {
                  'mean_reference_error': .035, 'p95_reference_error': .08,
                  'bright_residual': .04, 'bright_residual_fraction': .03,
                  'unchanged_tolerance': .002, 'unchanged_fraction': .1}}
    reasons = report['reasons']
    white_region = largest_region(np.isfinite(prediction).all(0) & (prediction.min(0) > .75))
    report['near_white_region_pixels'] = white_region
    if white_region >= max(9, int(affected.size*.0005)):
        reasons.append('large_remaining_near_white_region')
    if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        reasons.append('invalid_model_values')
    if not target['usable'].all():
        reasons.append('invalid_or_unclassified_target_pixels')
    if prior is not None:
        surface = terrain_screen(target, prior)
        report['visible_terrain'] = surface
        if not surface['accepted']:
            reasons.append('prior_changes_visible_terrain')
        valid = target['usable'] & ~prior['risk']
        if valid.any() and np.isfinite(prediction[:, valid]).all():
            screen = residual_screen(prediction, prior['rgb'], valid)
            report['whole_frame_residuals'] = screen
            if not screen['accepted']:
                reasons.append('remaining_cloud_shadow_or_surface_disagreement')
    if not affected.any():
        report.update(fallback_requested=bool(reasons), affected_pixels=0)
        return report
    support = affected & ~prior['risk'] if prior is not None else np.zeros_like(affected)
    report['affected_pixels'] = int(affected.sum())
    report['unsupported_pixels'] = int((affected & ~support).sum())
    if (affected & ~support).any():
        reasons.append('clouds_without_reference_support')
    if support.any() and np.isfinite(prediction[:, support]).all():
        ref = prior['rgb'][:, support]
        pred = prediction[:, support]
        difference = np.abs(pred - ref).mean(0)
        mean_error = float(difference.mean())
        p95_error = float(np.quantile(difference, .95))
        bright = float(((pred - ref).mean(0) > .04).mean())
        unchanged = float(((np.abs(pred - target['rgb'][:, support]).max(0) <= .002)
                           & (np.abs(target['rgb'][:, support] - ref).mean(0) > .035)).mean())
        report.update(mean_reference_error=mean_error, p95_reference_error=p95_error,
                      bright_residual_fraction=bright, unchanged_cloud_fraction=unchanged)
        if mean_error > .035 or p95_error > .08:
            reasons.append('large_difference_from_clear_prior')
        if bright > .03:
            reasons.append('remaining_bright_cloud_or_haze')
        if unchanged > .1:
            reasons.append('cloudy_pixels_not_changed')
    report['fallback_requested'] = bool(reasons)
    return report


def inspect_full_reference(raw, target_grid, target=None):
    """Require the full requested grid and zero flagged/unknown/missing pixels."""
    scene = read_scene(raw)
    if scene['grid'] != target_grid:
        return scene, {'qualified': False, 'reason': 'grid_mismatch'}
    with MemoryFile(raw) as mem, mem.open() as ds:
        probability = ds.read(5)
        physical_valid = (ds.read(6) > .5) & np.isfinite(ds.read()).all(0)
    physical_valid &= (scene['rgb'] >= 0).all(0) & (scene['rgb'] <= 1).all(0)
    valid = scene['usable'] & np.isfinite(scene['rgb']).all(0)
    valid &= (scene['rgb'] >= 0).all(0) & (scene['rgb'] <= 1).all(0)
    # Stricter than the model prior: even a single risk or missing pixel rejects
    # a full-image replacement. The native SCL/CLD masks are still fallible.
    risk = scene['risk'] | (probability >= 20) | ~valid
    qualified = bool(valid.all() and not risk.any())
    surface = None
    if physical_valid.all() and target is not None:
        surface = terrain_screen(target, scene)
        if not surface['accepted']:
            return scene, {'qualified': False, 'reason': 'visible_terrain_changed', 'terrain': surface}
    # Keep the clearest complete observed frame as a labelled preview if strict
    # acceptance fails everywhere. Never use partial footprints or mostly-cloud frames.
    preview_usable = bool(physical_valid.all() and risk.mean() < .5)
    scene['risk'] = risk
    return scene, {'qualified': qualified, 'preview_usable': preview_usable,
                   'reason': 'accepted' if qualified else 'cloud_shadow_or_incomplete_coverage',
                   'visible_terrain': surface,
                   'valid_fraction': float(valid.mean()), 'risk_fraction': float(risk.mean()),
                   'maximum_cloud_probability': float(np.nan_to_num(probability, nan=100).max())}


class WholeReferenceService(TemporalService):
    """The inherited service retains its original model, weights and inference."""

    def __init__(self, parent, checkpoint, output='temporal_web_results', client=None):
        super().__init__(parent, checkpoint, output=output, client=client or CoverageClient())

    def search_available(self, bbox, start, end, progress=lambda message: None):
        catalog = self.client.search(bbox, start, end)
        scenes = list({s['datetime']: s for s in catalog['scenes']}.values())
        accepted, rejected, failed = [], [], []
        cache = getattr(self, '_coverage_cache', {})
        self._coverage_cache = cache
        for index, scene in enumerate(scenes[:MAX_CHECKS]):
            stamp = scene['datetime']
            progress(f'Checking image coverage {index+1}/{min(MAX_CHECKS, len(scenes))}')
            key = (tuple(bbox), stamp)
            try:
                cached = cache.get(key)
                if cached and time.monotonic()-cached[0] < 1200:
                    fraction = cached[1]
                else:
                    fraction = check_coverage(self.client, bbox, stamp)
                    if key not in cache and len(cache) >= 512:
                        cache.pop(next(iter(cache)))
                    cache[key] = (time.monotonic(), fraction)
            except (RuntimeError, ValueError, rasterio.errors.RasterioError):
                failed.append(stamp)
                continue  # Never list an observation whose coverage was not verified.
            if fraction >= MIN_COVERAGE:
                accepted.append({**scene, 'coverage_fraction': fraction})
            else:
                rejected.append({'datetime': stamp, 'coverage_fraction': fraction})
        return {'scenes': accepted, 'truncated': bool(catalog.get('truncated') or len(scenes)>MAX_CHECKS),
                'coverage_threshold': MIN_COVERAGE, 'coverage_checked': min(MAX_CHECKS, len(scenes)),
                'coverage_rejected': rejected, 'coverage_failed': failed}

    def restore(self, bbox, acquisition, progress=lambda message: None):
        progress('Verifying selected image coverage')
        # Recheck the selected acquisition even for direct requests or an old search.
        fraction = check_coverage(self.client, bbox, acquisition)
        if fraction < MIN_COVERAGE:
            raise CoverageError(f'This image covers only {fraction*100:.1f}% of the selected area. Choose another acquisition.')
        return super().restore(bbox, acquisition, progress)

    def _find_clear_reference(self, bbox, acquisition, target, folder, model_date, progress):
        checks = []
        catalogs = []
        seen = set()
        best_preview = None
        best_score = None
        model_reference = folder/'reference_source.tif'
        if model_date and 0 < days(acquisition, model_date) <= MAX_REFERENCE_AGE and model_reference.exists():
            raw = model_reference.read_bytes()
            scene, quality = inspect_full_reference(raw, target['grid'], target)
            seen.add(model_date)
            checks.append({'date': model_date, 'source': 'model_prior', **quality})
            if quality['qualified']:
                return (raw, scene, model_date), checks, catalogs
            if quality.get('preview_usable'):
                best_preview = (raw, scene, model_date)
                best_score = (quality['risk_fraction'], abs(days(acquisition, model_date)))
        dt = datetime.fromisoformat(acquisition.replace('Z', '+00:00'))
        for youngest, oldest, limit in SEARCH_WINDOWS:
            progress('Searching nearby observations of the same area')
            start = (dt-timedelta(days=oldest)).date().isoformat()
            end = min(dt-timedelta(days=youngest), datetime.now(timezone.utc)).date().isoformat()
            if start > end:
                continue
            try:
                catalog = self.client.search(bbox, start, end)
            except (RuntimeError, ValueError):
                catalogs.append({'start': start, 'end': end, 'state': 'search_failed'})
                continue
            candidates = []
            for item in catalog['scenes']:
                date = item['datetime']
                age = days(acquisition, date)
                if date not in seen and start <= date[:10] <= end and -30 <= age <= MAX_REFERENCE_AGE and age != 0:
                    candidates.append(item)
            candidates.sort(key=lambda s: abs(days(acquisition, s['datetime'])))
            catalogs.append({'start': start, 'end': end, 'state': 'searched',
                             'truncated': bool(catalog.get('truncated', False)),
                             'candidates_available': len(candidates), 'candidate_limit': limit})
            fetched = 0
            for item in candidates:
                date = item['datetime']
                if date in seen:
                    continue
                if fetched >= limit:
                    break
                seen.add(date)
                fetched += 1
                progress(f'Checking full-area cloud coverage: {date[:10]} ({fetched}/{limit})')
                try:
                    raw = self.client.fetch(bbox, date)
                    scene, quality = inspect_full_reference(raw, target['grid'], target)
                except (RuntimeError, ValueError, rasterio.errors.RasterioError):
                    checks.append({'date': date, 'qualified': False, 'reason': 'reference_unavailable'})
                    continue
                checks.append({'date': date, 'source': 'expanded_search', **quality})
                if quality['qualified']:
                    return (raw, scene, date), checks, catalogs
                if quality.get('preview_usable'):
                    score = (quality['risk_fraction'], abs(days(acquisition, date)))
                    if best_score is None or score < best_score:
                        best_score, best_preview = score, (raw, scene, date)
        if best_preview is not None:
            for check in checks:
                if check['date'] == best_preview[2]:
                    check['selected_for_preview'] = True
        return best_preview, checks, catalogs

    def _restore(self, bbox, acquisition, progress):
        started = time.perf_counter()
        # This is exactly the original HSFormer + bounded temporal-prior flow.
        report, folder = super()._restore(bbox, acquisition, progress)
        target = read_scene((folder/'source_reflectance.tif').read_bytes())
        # Preserve the selected acquisition's API classification, never a reference mask.
        with rasterio.open(folder/'source_reflectance.tif') as source:
            scl, valid = source.read(4), source.read(6) > .5
        mask = np.zeros((*scl.shape, 3), dtype=np.uint8)
        mask[np.isin(scl, [8, 9, 10]) & valid] = [255, 255, 255]
        mask[(scl == 3) & valid] = [130, 100, 230]
        mask[(~valid) | (scl == 0)] = [100, 100, 100]
        Image.fromarray(mask).save(folder/'api_cloud_mask.png')
        # The model's original mask is untouched. The post-check also considers
        # moderate provider cloud probability and gaps outside that model mask.
        with rasterio.open(folder/'source_reflectance.tif') as source:
            moderate_cloud = source.read(5) >= 20
        target['risk'] |= moderate_cloud
        target['cloud'] |= moderate_cloud
        prior_path = folder/'reference_source.tif'
        prior = read_scene(prior_path.read_bytes()) if prior_path.exists() else None
        with rasterio.open(folder/'restored.tif') as ds:
            prediction = ds.read()
            profile = ds.profile.copy()
        progress('Checking the HSFormer/prior result for incomplete cloud removal')
        review = assess_model(target, prediction, prior)
        for source, dest in [('restored.png', 'model.png'), ('restored.tif', 'model.tif'),
                             ('edit_mask.png', 'model_edit_mask.png'),
                             ('unprocessed_cloud_mask.png', 'model_unprocessed_cloud_mask.png')]:
            shutil.copyfile(folder/source, folder/dest)
        (folder/'model_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        report.update(model_quality_review=review, model_state=report['state'],
                      display_mode='original' if report['state']=='no_cloud_flagged' else 'model',
                      displayed_acquisition=acquisition, fallback_reference_date=None,
                      full_reference_checks=[], full_reference_catalogs=[],
                      result_available=True, quality_status='preview' if review['fallback_requested'] else 'checks_passed',
                      fallback_policy={'maximum_age_days': MAX_REFERENCE_AGE, 'maximum_later_days': 30,
                                       'maximum_additional_images': 36, 'preview_risk_limit': .5,
                                       'maximum_risk_pixels': 0, 'cloud_probability_cutoff': 20,
                                       'same_grid_required': True, 'whole_image_only': True})
        if review['fallback_requested']:
            match, checks, catalogs = self._find_clear_reference(
                bbox, acquisition, target, folder, report['reference_date'], progress)
            report.update(full_reference_checks=checks, full_reference_catalogs=catalogs)
            if match is not None:
                raw, scene, date = match
                selected_quality = next(c for c in checks if c['date']==date)
                strict = selected_quality['qualified']
                image = scene['rgb'].astype(np.float32)
                # Whole observed RGB frame. No spatial patching, blending or
                # per-image recolouring or transition animation.
                png(image, folder/'clear_reference.png')
                shutil.copyfile(folder/'clear_reference.png', folder/'restored.png')
                (folder/'clear_reference_source.tif').write_bytes(raw)
                with rasterio.open(folder/'restored.tif', 'w', **profile) as dst:
                    dst.write(image)
                    dst.write_mask(np.full(image.shape[1:], 255, dtype='uint8'))
                    dst.update_tags(description='Full reference observation; not model restoration of the selected date',
                                    target_date=acquisition, observation_date=date, reference_date=date,
                                    display_mode='full_reference')
                Image.fromarray(scene['risk'].astype('uint8')*255).save(folder/'unprocessed_cloud_mask.png')
                Image.fromarray(np.full(image.shape[1:], 255, dtype='uint8')).save(folder/'edit_mask.png')
                report.update(state='full_reference_shown' if strict else 'reference_preview_shown', display_mode='full_reference',
                              result_available=True, quality_status='checks_passed' if strict else 'reference_preview',
                              displayed_risk_fraction=selected_quality['risk_fraction'],
                              displayed_acquisition=date, fallback_reference_date=date,
                              fallback_reference_age_days=days(acquisition, date),
                              edited_fraction=1.0, unprocessed_cloud_fraction=float(scene['risk'].mean()),
                              unchanged_outside_edit_mask=True,
                              notice=('Showing one full reference observation of this area from the displayed date. '
                                      + ('All pixels passed provider-mask checks; masks can miss haze.' if strict else
                                         'This is the clearest usable reference found; some clouds, shadows or uncertain pixels remain.')))
            else:
                report.update(state='full_reference_unavailable',
                              result_available=True, quality_status='preview',
                              display_mode='original' if report['model_state'] in ('no_usable_reference', 'no_cloud_flagged') else 'model',
                              notice='The model result did not pass the visual checks, and the bounded search found no fully clear '
                                     'recent reference consistent with this area. The best available model preview is displayed; clouds or shadows may remain.')
        report['seconds'] = round(time.perf_counter()-started, 2)
        (folder/'metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return report, folder
