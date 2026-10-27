# Cloud Atlas
### Cloud Density-Aware Cloud Removal for Optical Remote Sensing Images

Cloud Atlas is a final-year research project exploring cloud and cloud shadow removal from optical satellite imagery. It combines model comparison, adaptation to Sentinel-2 imagery, and learning from earlier observations with a local web application.

The application provides Google sign-in, location search, satellite acquisition selection, image comparison, downloads, and account-based result history.

## Project Workflow

1. Prepare paired cloudy and clear RGB images from RICE.
2. Train and compare a cloud-guided attention U-Net and an RGB-adapted HSFormer.
3. Fine-tune both models on a prepared WHUS2-CRv subset.
4. Train a fusion component using frozen HSFormer predictions and earlier observations of the same area.
5. Evaluate the models and provide an interface for satellite image processing.

## Models

### Cloud-Guided Attention U-Net

A four-scale encoder–decoder model with skip connections. It predicts a cloud mask, a corruption-severity map, and an RGB correction.

The mask and severity guide feature processing. The predicted mask controls how much correction is applied to the input image.

### RGB-Adapted HSFormer

An adaptation of the published HSFormer architecture for three-channel RGB imagery. It combines convolutional feature extraction, transformer attention, multi-scale reconstruction, and refinement.

The original architecture is credited to its authors. This project's RGB training setup is not a reproduction of every experiment in the original paper.

### Learned Prior Fusion

A small convolutional network combines:

- The current cloudy image.
- The frozen HSFormer prediction.
- An earlier observation of the same location.
- Cloud/shadow and validity masks.
- The time gap between observations.

The network predicts pixelwise mixing weights and an additional RGB candidate. Valid input pixels outside the supplied affected-region mask are preserved.

## Datasets

The counts below describe the subsets prepared for this project.

| Dataset | Purpose | Training | Validation | Test |
|---|---|---:|---:|---:|
| RICE | Initial model comparison | 865 | 185 | 186 |
| WHUS2-CRv subset | Sentinel-2 fine-tuning | 512 | 64 | — |
| Satellite image triplets | Learned prior fusion | 69 | 24 | 24 |

Images were processed as 256 × 256 RGB patches.

The triplet dataset contains 117 examples across 12 locations, with separate locations used for training, validation, and testing. Earlier observations were selected within 60 days before the input acquisition.

Datasets are not included in this repository.

## Experimental Results

### RICE Test Partition

Evaluation on 186 held-out image pairs.

| Model | PSNR ↑ | SSIM ↑ | RMSE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Cloud-guided attention U-Net | 32.081 | 0.9295 | 0.03065 | 0.13127 |
| RGB-adapted HSFormer | 30.011 | 0.9204 | 0.03779 | 0.14419 |

Attention U-Net performed better across the four reported metrics on this partition.

### Sentinel-2 Validation Subset

Evaluation on 64 WHUS2-CRv validation pairs after fine-tuning.

| Model | PSNR ↑ | SSIM ↑ | RMSE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Fine-tuned attention U-Net | 33.262 | 0.9104 | 0.02683 | 0.10954 |
| Fine-tuned HSFormer | 33.477 | 0.9166 | 0.02652 | 0.10874 |

HSFormer achieved a small numerical advantage across all four metrics and was selected as the backbone for prior-fusion experiments.

### Prior-Fusion Evaluation

Results are averaged across held-out test locations.

| Method | Proxy PSNR ↑ | Proxy cloud MAE ↓ |
|---|---:|---:|
| Single-image HSFormer | 17.818 | 0.21752 |
| Learned fusion | 31.750 | 0.02206 |
| Earlier-observation baseline | 34.028 | 0.01495 |

The learned fusion model improved over single-image HSFormer but did not outperform the earlier-observation baseline.

These are proxy measurements because the targets were captured on different dates. They do not verify the exact surface conditions beneath clouds on the selected date.

PSNR is measured in decibels. Scores from different datasets and evaluation settings should not be directly ranked.

## Web Application Features

- Google account sign-in.
- Location search and interactive map selection.
- Satellite acquisition search.
- Original/result image comparison.
- PNG downloads.
- Saved results with location labels.
- Account-based result access and deletion.

The application runs locally on the user's computer. Internet access is required for Google sign-in, location search, and satellite image retrieval.

The experimental tables above describe evaluated model outputs, rather than establishing the accuracy of every image displayed by the application.

## Technology Stack

| Component | Technology |
|---|---|
| Model implementation | PyTorch |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Interactive map | Leaflet |
| Raster processing | Rasterio |
| Array processing | NumPy |
| Image handling | Pillow |
| User authentication | Google OAuth |
| Satellite services | Copernicus Data Space Ecosystem / Sentinel Hub |

## Repository Structure

```text
cloud-atlas/
├── cloud_severity/       # Models, losses, metrics, and dataset utilities
├── sentinel_triplets/    # Triplet preparation, fusion, and evaluation
├── temporal_prior/      # Supporting prior-image processing modules
├── reference_web/       # Application backend and frontend
├── web/                 # Supporting web assets
├── scripts/
│   ├── prepare_hsformer.py
│   └── run_local_web.py
├── docs/
│   └── LOCAL_WEBSITE.md
├── requirements*.txt
├── run_website.py
└── Start Cloud Atlas.cmd
