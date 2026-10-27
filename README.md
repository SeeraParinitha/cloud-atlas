# Cloud Atlas: cloud and shadow restoration

The original HSFormer-with-prior reference website now has a **full-photo fallback**:
when the model fails visual checks, it can show one complete clear reference
observation of the same area in an original/result slider. Use
`notebooks/reference_web_full_fallback.ipynb` and the same original weight datasets.
See [the reference website guide](docs/REFERENCE_WEB_FULL_FALLBACK.md).

## Use the already-trained model

Training is complete for the supplied output archive. Use `notebooks/cloud_shadow_inference.ipynb` to launch the map website without training. See `docs/INFERENCE.md` for exact Kaggle steps. City/town search, a draggable centre marker, and two-corner rectangle selection are included. Start on CPU to save GPU quota.

## Run on Kaggle

Upload `notebooks/cloud_shadow_complete.ipynb`. Select a T4 GPU and enable Internet. Add Copernicus Sentinel Hub OAuth credentials as Kaggle Secrets named `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`, granting this notebook access. These are required only for satellite retrieval. An optional `APP_PASSWORD` sets the demo login password. Then Run All.

The notebook contains the project source: no manual source upload or dataset-path edits are needed for RICE. It downloads the official RICE ZIP, checks it, prepares paired images, trains both models, compares validation results, exports a winner, evaluates the separate test split, generates reports and creates `cloud_shadow_complete.zip`. The final cell starts a password-protected temporary web demo if credentials are present and both models finished.

The faster default is 30 epochs per model (reduced from 100). This cuts the epoch count by 70%, but can reduce final accuracy and does not guarantee completion within 20 GPU hours. Each epoch logs an estimated remaining time for that model. Even 30 epochs per model can exceed one Kaggle session. The code stops around its 10.5-hour budget, saves checkpoints and reports incomplete training. Download the output ZIP and attach it as a Kaggle input to the next session; Run All resumes. It does not lower the training schedule or select from incomplete runs. GPU quota, downloads and provider availability cannot be guaranteed by code. Kaggle's two T4 option uses one GPU here; this is not distributed training.

## Models and original-code preservation

- Model 1 (`full`): the original residual CNN encoder, mask/severity guidance, attention decoder and gated restoration. Its model weights/layout, outputs and loss are preserved; regression tests compare it with the original extracted source. `notebooks/original_reference.ipynb` is the unmodified archive of your provided notebook. The new notebook reorganizes execution and adds orchestration; it does not reproduce every old display cell.
- Model 2 (`hsformer`): official HSFormer RGB architecture pinned to commit `8d59013b7f9ecaf3e048b1d9dc7eac85805b0b38`, adapted to the same final-output restoration loss. This is **not** an exact reproduction of the paper's multiscale/uncertainty training. Its predicted variance is not presented as confidence. It uses 256Ã—256 patches and batch 1 with four-step gradient accumulation to reduce GPU memory.
- One seed (123), no cross-validation. Fixed 70/15/15 train/validation/test split, seed 42 for partitioning. Image-level split: geographic independence is not established. Pseudo-supervision calibration uses training data only.

Each model's checkpoint uses its highest validation PSNR. Across models, the winner has the lowest mean rank over validation PSNR, SSIM, LPIPS and stable-clear change MAE. Ties use PSNR then latency. Test images never select the winner. Reports include RMSE, clear/affected-region metrics, per-image results, runtime, training curves and example panels. A relative winner can still perform poorly in absolute terms: inspect held-out results before making scientific claims.

The external HSFormer repository has no license file in the pinned source checked here. Review upstream rights before redistribution or commercial deployment. Source is fetched at runtime; it is not included in the source-delivery ZIP.

## WHUS2-CR

External evaluation support is included, but no WHUS2-CR archive or verified prepared test manifest was supplied. The default run clearly records `whus2cr: not run`; RICE completion is not a completed WHUS2-CR benchmark. See `docs/DATA.md` for the manifest/protocol requirements. Set `whus2_manifest` and `whus2_protocol` in `configs/run_all.json` after attaching prepared official test data. Both models are evaluated without fine-tuning; those results do not choose the winner.

## Satellite API and website

The backend uses Copernicus Data Space Sentinel Hub OAuth, Catalog and Process APIs. Create an account and OAuth client in the Sentinel Hub dashboard: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html . Credentials stay on the backend. Access is subject to provider quotas.

A user clicks a map or enters latitude/longitude, selects an area and date range, chooses an acquisition, and requests restoration. Outputs are original/restored PNG, georeferenced restored RGB TIFF, original reflectance TIFF and processing metadata. The original model also exposes mask/severity maps through result endpoints. Acquisition date is shown: this is on-demand processing of available satellite observations, not a live satellite video feed.

Sentinel-2 L1C RGB bands B04/B03/B02 are converted with fixed `clip(reflectance / 0.4, 0, 1)` scaling. The service restores overlapping 256-pixel tiles and retains no-data masks/georeferencing. This display-RGB adaptation needs separate satellite-domain validation. RICE metrics do not prove API imagery accuracy. A single image cannot establish the true content beneath opaque cloud; outputs are explicitly labeled estimates, not measured cloud-free ground truth.

### Persistent website

Extract the Kaggle output ZIP. Install a CUDA-compatible PyTorch build if using a GPU, then:

```
pip install -r requirements.txt -r requirements-web.txt
```

Set environment variables `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`, and a strong `APP_PASSWORD` on the server. `.env.example` lists names; files are not automatically loaded. The login username is `admin`.

```
python -m uvicorn cloud_severity.api:app --host 0.0.0.0 --port 8000
```

Use one worker (the bounded in-process queue shares one model). Put the service behind HTTPS before remote access. `Dockerfile` is included and requires `deployment/model.pt` plus the prepared HSFormer runtime from the Kaggle output. This is a single-user research application; job history is in memory and image files persist in `web_results`. Apply hosting storage/retention controls for sustained use.

The Kaggle Gradio link is temporary and requires an active interactive kernel. A background saved run will not provide permanent hosting. The persistent FastAPI website includes a clickable map and before/after comparison slider; the Kaggle demo uses coordinates with a map preview.

## Local verification

```
python scripts/prepare_hsformer.py
python -m pytest -q
python -m cloud_severity.pilot
```

Tests include synthetic training/resume/evaluation, original-model parity, validation-only selection, tile boundaries, authenticated API validation and georeferenced output with a mocked provider. CPU verification is not a completed T4 training benchmark or a live satellite-account test. No trained accuracy numbers are invented.

Build the self-contained notebook after source edits with `python scripts/build_notebook.py`.

Sources: [RICE](https://github.com/BUPTLdy/RICE_DATASET), [HSFormer](https://github.com/qi657/HSFormer-CR), [Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html).
