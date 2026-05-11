# LepMuse Segmentation Model Guide

A complete reference for training, evaluating, running inference, and extending the segmentation backend used in the LepMuse butterfly measurement pipeline.

---

| § | Section | Description |
|---|---|---|
| 1 | [Architecture Overview](#architecture-overview) | Model class, backbone, segmentation classes, loss, metric, and export format |
| 2 | [Environment Setup](#environment-setup) | Conda environment activation, version verification, and dependency notes |
| 3 | [Dataset Layout](#dataset-layout) | Directory structure, `codes.txt` format, pixel-value mapping, label validation |
| 4 | [Training](#training) | Config reference, GPU budget, smoke test, full training run, ResNet34 variant |
| 5 | [Evaluation](#evaluation) | Mask-level accuracy, end-to-end measurement accuracy, stage isolation |
| 6 | [Inference & Pipeline Integration](#inference--pipeline-integration) | Python API, all CLI input modes, flags, and JSON config pattern |
| 7 | [Troubleshooting](#troubleshooting) | Common error messages with root causes and fixes |
| 8 | [Plugging in a New Model (SAM2 example)](#plugging-in-a-new-model-sam2-example) | Protocol interface, files to create, factory registration, SAM2 strategies |

---

## Architecture Overview

The segmenter is a **UNet with a ResNet18 encoder** trained via [fastai 2.8.x](https://docs.fast.ai/) for 4-class semantic segmentation.

| Property | Value |
|---|---|
| Task | 4-class semantic segmentation |
| Classes | `background` · `lepidopteran` · `tags` · `ruler` |
| Encoder | ResNet18 (ImageNet pretrained) |
| Decoder | UNet transposed-conv decoder + self-attention |
| Input resolution | 800 × 1200 px |
| Output | Per-pixel class probability maps → binary masks |
| Loss | Cross-entropy (fastai default for segmentation) |
| Metric | `acc_camvid` — pixel accuracy over non-background pixels |
| Export format | FastAI `.pkl` — cloudpickle serialised `Learner` (fastai ≥ 2.8) |

The pipeline stages downstream of segmentation are ruler scale detection (Fourier tick analysis) and landmark-based wing measurement. The segmenter interface is a simple protocol — any backend that returns a `SegmentationResult` with `lepidopteran_mask`, `ruler_mask`, and `tag_mask` can be dropped in without touching the rest of the pipeline.

---

## Environment Setup

```bash
conda activate lepmuse
cd /home/rahul/workspace/vision/lepmuse
```

Verify the key packages (expected versions in the `lepmuse` environment):

```bash
python -c "
import fastai, fastcore, fasttransform, torch, torchvision
print('fastai       ', fastai.__version__)        # 2.8.7
print('fastcore     ', fastcore.__version__)      # 1.12.46
print('fasttransform', fasttransform.__version__) # 0.0.2
print('torch        ', torch.__version__)         # 2.4.1+cu118
print('cuda         ', torch.cuda.is_available())
print('devices      ', torch.cuda.device_count())
"
```

> **fastai 2.8 / fastcore 1.8+ note** — `fastcore.transform` and `fastcore.dispatch` are now empty stubs; all transform classes (`Transform`, `Pipeline`, `TypeDispatch`, etc.) live in the [`fasttransform`](https://github.com/AnswerDotAI/fasttransform) package. fastai 2.8 requires and depends on `fasttransform` directly. Models exported with fastai ≤ 2.7 **cannot** be loaded with fastai ≥ 2.8 — they must be retrained and re-exported.

Install or refresh dependencies:

```bash
pip install -r requirements.txt
```

> `opencv-python-headless` is the only OpenCV variant listed in `requirements.txt`.
> Do **not** install `opencv-python` alongside it — they share the same `cv2` module and will conflict.

---

## Dataset Layout

Both training and validation datasets follow the same directory structure:

```
datasets/battus100/
├── training_images/
│   ├── codes.txt          # class names, space-separated on one line
│   ├── images/            # input images  (.JPG / .PNG)
│   └── labels/            # segmentation masks (.PNG, pixel values 0–3)
└── val_images/
    ├── codes.txt
    ├── images/
    ├── labels/             # ground-truth masks for segmentation evaluation
    └── manual_measurements_gt.csv   # ground-truth wing measurements (mm)
```

`codes.txt` content (must match this exact order — class index 0 is background):

```
background lepidopteran tags ruler
```

Label mask pixel values:

| Value | Class |
|---|---|
| 0 | background |
| 1 | lepidopteran |
| 2 | tags |
| 3 | ruler |

Verify label range before training:

```bash
python -c "
import numpy as np
from pathlib import Path
from PIL import Image

labels = sorted(Path('datasets/battus100/training_images/labels').glob('*.png'))
vals = set()
for p in labels:
    vals |= set(np.unique(np.array(Image.open(p))))
print('Unique pixel values:', sorted(vals))  # expected: [0, 1, 2, 3]
"
```

---

## Training

### Configuration

Training is driven by `TrainConfig` in [segmentation/unet/train.py](segmentation/unet/train.py). The default JSON config lives at [configs/unet_train_battus100.json](configs/unet_train_battus100.json):

```json
{
  "train_data_path": "datasets/battus100/training_images",
  "output": "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl",
  "epochs": 50,
  "batch_size": 8,
  "image_height": 800,
  "image_width": 1200,
  "architecture": "resnet18",
  "cuda_visible_devices": null,
  "num_workers": 4,
  "self_attention": false,
  "mixed_precision": true,
  "valid_pct": 0.2,
  "seed": 42,
  "runs_dir": "runs",
  "run_name": null,
  "tensorboard": true,
  "log_preds": true,
  "csv_log": true,
  "checkpoint_metric": "acc_camvid",
  "early_stopping_patience": null,
  "aug_oversample": 1.15,
  "aug_min_zoom": 0.75,
  "aug_max_zoom": 1.35,
  "aug_max_rotate": 90.0,
  "aug_max_warp": 0.35,
  "aug_max_lighting": 0.35,
  "aug_p_affine": 0.75,
  "aug_p_lighting": 0.75,
  "aug_random_erasing": true,
  "aug_random_erasing_p": 0.3
}
```

`valid_pct` and `seed` control the reproducible train/val split — at `valid_pct=0.2` the 80 training images split 64 train / 16 validation each run.

### Augmentation strategy

The dataset has three spatially-variable elements — the butterfly specimen, a measurement ruler, and specimen tags — that can appear at different positions, scales, and orientations depending on how the tray was loaded and photographed. The augmentation pipeline is designed to make the model invariant to all of these variations.

| Augmentation | Config key | Purpose |
|---|---|---|
| Oversample resize → RandomCrop | `aug_oversample=1.15` | Resize to 115% of target then randomly crop back. Forces the model to learn objects at different frame positions — ruler is not always at the bottom, tags not always at fixed corners |
| Zoom out / in | `aug_min_zoom=0.75`, `aug_max_zoom=1.35` | Specimens vary in body size; museum batches use different camera distances. 25% zoom-out to 35% zoom-in covers the realistic range |
| Rotation | `aug_max_rotate=90.0` | ±90° rotation; combined with horizontal and vertical flips gives full dihedral-8 coverage, so orientation of the ruler strip carries no positional meaning |
| Perspective warp | `aug_max_warp=0.35` | Camera angle varies across light-box shots; warp simulates mild keystoning |
| Lighting / contrast | `aug_max_lighting=0.35` | Museum lamp aging and positioning causes batch-to-batch brightness and contrast shifts |
| RandomErasing | `aug_random_erasing=true`, `aug_random_erasing_p=0.3` | Erases random rectangular patches from the **input image only** (not the mask). Simulates: tags partially overlapping the specimen, ruler partially outside the frame, glare patches. Trains the model to use surrounding context rather than fixed local texture |

**Validation receives no augmentation.** `item_tfms` resizes validation images to the exact target size; the `aug_transforms`, `RandomCrop`, and `RandomErasing` transforms all carry `split_idx=0` in fastai's pipeline and are skipped during validation.

**To ablate a technique** (e.g. compare with and without RandomErasing):

```bash
# With erasing (default)
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --run-name resnet18_b8_erasing

# Without erasing
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --no-random-erasing \
  --run-name resnet18_b8_no_erasing

tensorboard --logdir runs
```

### Run naming

When `run_name` is `null`, the folder name is auto-generated from the actual config fields at launch time:

```
{timestamp}_{architecture}_b{batch_size}_e{epochs}_s{image_height}x{image_width}
```

Example: `20240411_103002_resnet18_b8_e50_s800x1200`

This means overriding `--batch-size 16` on the CLI is immediately visible in the run folder name — it will read `_b16_` — rather than inheriting whatever string was baked into the `output` filename. Use `--run-name` to add a semantic label on top:

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --run-name resnet34_strong_aug \
  --architecture resnet34 \
  --cuda-visible-devices 0
```

**GPU memory budget** (resnet18, 800 × 1200, `self_attention=false`, `mixed_precision=true`):

| VRAM | Recommended `batch_size` |
|---|---|
| 8 GB | 2–4 |
| 16 GB | 8 (default) |
| 24 GB+ | 16, or switch to `resnet34` |

> **`self_attention` is disabled by default.** At 800 × 1200 the UNet self-attention block operates on 200 × 300 feature maps (60,000 tokens). The resulting N × N attention matrix requires **14.4 GB fp32 / 7.2 GB fp16** per batch — more than a full A100 can spare once the rest of the model and activations are loaded. `self_attention=true` will always OOM at this resolution; only enable it if you reduce `image_height` / `image_width` to at most 512 × 512.
>
> **Mixed precision** (`mixed_precision=true`, default) halves activation memory with no accuracy loss on modern GPUs (Volta/Turing/Ampere). Disable with `"mixed_precision": false` only if you see NaN losses, which can happen on very old GPUs without native fp16 support.

### Step 1 — Smoke test (battus10 subset, 2 epochs)

Run a quick end-to-end check before committing to the full training run:

```bash
python -m segmentation.unet.train \
  --train-data-path datasets/battus10/training_images \
  --output /tmp/lepmuse_smoke.pkl \
  --epochs 2 \
  --batch-size 2 \
  --cuda-visible-devices 0
```

Expected console output shows the fastai training table with `train_loss`, `valid_loss`, and `acc_camvid` columns, plus paths for TensorBoard and the CSV log:

```log
2024-04-11 10:30:00   Unique labels: {'background': 0, 'lepidopteran': 1, 'tags': 2, 'ruler': 3}
2024-04-11 10:30:01   Loading segmentation masks data for images: 8
2024-04-11 10:30:01   Run directory   → /workspace/lepmuse/runs/20240411_103001_lepmuse_smoke_b2
2024-04-11 10:30:01     tensorboard --logdir /workspace/lepmuse/runs
2024-04-11 10:30:02   CSV log         → runs/20240411_103001_lepmuse_smoke_b2/history.csv
2024-04-11 10:30:02   TensorBoard     → runs/20240411_103001_lepmuse_smoke_b2
epoch  train_loss  valid_loss  acc_camvid  time
0      0.4821      0.3912      0.7234      00:45
Better model found at epoch 0 with acc_camvid value: 0.7234.
1      0.3105      0.2871      0.8012      00:43
Better model found at epoch 1 with acc_camvid value: 0.8012.
2024-04-11 10:32:31   Exported model  → runs/20240411_103001_lepmuse_smoke_b2/lepmuse_smoke.pkl
```

`acc_camvid` should be a float (not `nan`) from epoch 0 onward. If it shows `nan`, verify label masks contain foreground pixels (§3).

### Step 2 — Full training on battus100

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --cuda-visible-devices 0
```

Override individual parameters on the CLI without editing the JSON:

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --cuda-visible-devices 1 \
  --batch-size 16 \
  --epochs 50 \
  --early-stopping-patience 10
```

Label a run for easy identification in TensorBoard:

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --run-name resnet18_b10_e50_lr1e3 \
  --cuda-visible-devices 1
```

Disable TensorBoard for a headless run (other artifacts still land in the run directory):

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --no-tensorboard \
  --cuda-visible-devices 0
```

### Step 3 — (Optional) ResNet34 variant

ResNet34 typically gains 1–2% on `acc_camvid` at the cost of roughly 2× memory and 1.4× training time:

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --architecture resnet34 \
  --output segmentation/unet/models/battus100_segm_c4_resnet34_b8_e50_s1200x800.pkl \
  --run-name resnet34_b8_e50 \
  --cuda-visible-devices 0
```

### What to watch during training

`fine_tune(50)` uses fastai's 1-cycle policy with progressive layer unfreezing:

- **Epoch 0** — backbone frozen; only the UNet decoder trains. `acc_camvid` typically starts between 0.65–0.75.
- **Epochs 1–49** — full network trains with discriminative learning rates (lower LR for early encoder layers). `acc_camvid` should climb toward 0.88–0.93 by epoch 30 and plateau.
- A `Better model found` log line prints each time `SaveModelCallback` saves a new best checkpoint.
- If `acc_camvid` stays below 0.60 after 10 epochs, verify label mask pixel values match the `codes.txt` order.

### Tracking with TensorBoard

Launch TensorBoard in a separate terminal from the repo root while training is running (or after):

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006` in a browser. Three tabs are populated:

| Tab | What you see |
|---|---|
| **Scalars** | `train_loss` per batch; `valid_loss` and `acc_camvid` per epoch; learning rates per layer group |
| **Images** | 4 side-by-side tiles per epoch — input image, ground-truth mask, predicted mask, overlay — sampled from the validation set |
| **Graphs** | UNet model graph (disabled by default; enable with `trace_model=true`) |

Every training run writes all its artifacts to a single timestamped subdirectory under `runs_dir` (default `runs/`). Runs never overwrite each other and can be compared side-by-side in TensorBoard by pointing `--logdir` at the base directory:

```
runs/
├── 20240411_103002_battus100_segm_c4_resnet18_b8_e50_s1200x800_b8/
│   ├── events.out.tfevents.*                           ← TensorBoard scalars + images
│   ├── history.csv                                     ← per-epoch metrics (CSV)
│   ├── train.log                                       ← full stdout + stderr transcript
│   ├── best_ckpt.pth                                   ← best checkpoint weights
│   └── battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl ← exported learner (inference)
├── resnet34_b8_e50/                                    ← custom --run-name
│   ├── events.out.tfevents.*
│   ├── history.csv
│   ├── train.log
│   ├── best_ckpt.pth
│   └── battus100_segm_c4_resnet34_b8_e50_s1200x800.pkl
└── ...
```

### Tracking with the CSV log

`history.csv` is written to the run directory alongside the TensorBoard events:

```
runs/<run_name>/history.csv
```

It contains one row per epoch and is suitable for plotting learning curves with pandas or any spreadsheet tool:

```
epoch,train_loss,valid_loss,acc_camvid,time
0,0.4821,0.3912,0.7234,00:45
1,0.3105,0.2871,0.8012,00:43
...
```

### Checkpointing and best-model export

`SaveModelCallback` monitors `acc_camvid` (higher = better) and writes `best_ckpt.pth` into the run directory whenever a new best validation score is achieved. When `fine_tune()` finishes, fastai **automatically reloads** `best_ckpt.pth` before returning — so `learner.export()` always serialises the best-validation weights, not the final-epoch weights.

All artifacts land in the same run directory:

```
runs/<run_name>/
├── events.out.tfevents.*                           ← TensorBoard events
├── history.csv                                     ← per-epoch metrics log
├── train.log                                       ← full stdout + stderr transcript
├── best_ckpt.pth                                   ← best checkpoint (reloadable with learner.load())
└── battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl ← exported learner (best weights, used for inference)
```

To load the exported model for inference or to reload the checkpoint for further fine-tuning:

```python
from fastai.vision.all import load_learner
learner = load_learner("runs/<run_name>/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl")
# or to reload checkpoint weights into a live learner:
learner.load("best_ckpt")
```

### Early stopping

Set `early_stopping_patience` to a positive integer to stop training automatically when `acc_camvid` has not improved for that many consecutive validation epochs. The `SaveModelCallback` still reloads the best weights before export.

```bash
python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json \
  --early-stopping-patience 10 \
  --epochs 200
```

A patience of 10 on a 50-epoch budget typically fires around epoch 35–40 once the model has converged, saving roughly 30% of training time.

### Output

The trained model is exported to `runs/<run_name>/<output filename>.pkl`. The `output` config field specifies only the filename; the run directory is always the parent. The `.pkl` is a cloudpickle-serialised fastai `Learner` (fastai 2.8+ default) — it embeds the architecture, all transform parameters, and class vocabulary inline, so inference requires no separate config file.

> **Migrating from an older `.pkl`** — models exported with fastai ≤ 2.7 reference `fastcore.dispatch.TypeDispatch` and `fastcore.transform.*`, which no longer exist in fastcore 1.8+. fastai 2.8 will raise a `RuntimeError` on load. The only migration path is to **retrain and re-export** with `python -m segmentation.unet.train`.

---

## Evaluation

Two evaluation levels are available: segmentation mask accuracy and end-to-end wing measurement accuracy.

### Segmentation accuracy (mask-level)

Uses the 20 held-out images in `datasets/battus100/val_images/` and reports per-image foreground accuracy against ground-truth label masks.

```bash
python -m segmentation.unet.evaluate \
  --config configs/unet_eval_battus100.json
```

Config ([configs/unet_eval_battus100.json](configs/unet_eval_battus100.json)):

```json
{
  "weights": "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl",
  "image_dir": "datasets/battus100/val_images/images",
  "mask_dir":  "datasets/battus100/val_images/labels",
  "output_csv": "datasets/battus100/results/segmentation_eval.csv"
}
```

Output CSV columns: `image_id · mask_path · foreground_accuracy · status · error`

CLI overrides:

```bash
python -m segmentation.unet.evaluate \
  --weights segmentation/unet/models/battus100_segm_c4_resnet34_b8_e50_s1200x800.pkl \
  --image-dir datasets/battus100/val_images/images \
  --mask-dir  datasets/battus100/val_images/labels \
  --output-csv /tmp/eval_resnet34.csv
```

### End-to-end measurement accuracy (pipeline-level)

Runs the full pipeline (segmentation → ruler scale → landmark detection → measurements) and produces a CSV of wing measurements that can be compared against `manual_measurements_gt.csv`.

```bash
python -m lepmuse.cli \
  --config configs/pipeline_battus100_val_unet.json
```

Config ([configs/pipeline_battus100_val_unet.json](configs/pipeline_battus100_val_unet.json)):

```json
{
  "input":          "datasets/battus100/val_images/images",
  "output_folder":  "datasets/battus100/results/val_images",
  "path_csv":       "datasets/battus100/results/val_images/results.csv",
  "stage":          "measurements",
  "segmenter":      "unet",
  "weights":        "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl",
  "continue_on_error": true,
  "write_failures":    true,
  "min_pixels_per_mm": 1.0,
  "max_pixels_per_mm": 500.0,
  "min_wing_mm":       5.0,
  "max_wing_mm":       120.0,
  "min_shoulder_mm":   1.0,
  "max_wing_asymmetry_mm": 30.0
}
```

Output CSV columns (from [lepmuse/results.py](lepmuse/results.py)):

```
image_id · left_wing (mm) · right_wing (mm) · left_wing_center (mm) ·
right_wing_center (mm) · wing_span (mm) · wing_shoulder (mm) ·
image_path · view · specimen_id · stage · segmenter · pixels_per_mm · status · error
```

Compare to ground truth:

```python
import pandas as pd

pred = pd.read_csv("datasets/battus100/results/val_images/results.csv")
gt   = pd.read_csv("datasets/battus100/val_images/manual_measurements_gt.csv")
merged = pred.merge(gt, on="image_id", suffixes=("_pred", "_gt"))
mae = (merged["left_wing (mm)_pred"] - merged["left_wing (mm)_gt"]).abs().mean()
print(f"Left wing MAE: {mae:.2f} mm")
```

### Stopping at an intermediate stage

The `stage` parameter controls how far through the pipeline each image is processed. This is useful to isolate which stage is failing on a problematic image:

| `stage` | Output |
|---|---|
| `binarization` | Segmentation masks only |
| `ruler_detection` | Masks + `pixels_per_mm` scale |
| `measurements` | Full wing measurements (default) |

```bash
python -m lepmuse.cli \
  --config configs/pipeline_battus100_val_unet.json \
  --stage binarization
```

---

## Inference & Pipeline Integration

### Loading the model

```python
from segmentation.unet.infer import UNetSegmenter

seg = UNetSegmenter("segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl")
```

### Segmenting a single image

```python
import numpy as np
from skimage.io import imread
from lepmuse.types import ImageRecord

image_rgb = imread("path/to/image.jpg")
record = ImageRecord(image_id="image.jpg", path="path/to/image.jpg")

result = seg.segment(image_rgb, record)
# result.lepidopteran_mask  — boolean ndarray, shape (H, W)
# result.ruler_mask         — boolean ndarray, shape (H, W)
# result.tag_mask           — boolean ndarray, shape (H, W)
```

### Running the full CLI pipeline

Single image:

```bash
python -m lepmuse.cli \
  --input path/to/image.jpg \
  --path_csv results.csv \
  --weights segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl
```

Folder of images:

```bash
python -m lepmuse.cli \
  --input path/to/images/ \
  --path_csv results.csv \
  --output_folder path/to/outputs/
```

CSV manifest (requires an `image_path`, `image_paths`, or `path` column):

```bash
python -m lepmuse.cli \
  --input manifest.csv \
  --path_csv results.csv
```

Text file manifest (one absolute path per line):

```bash
python -m lepmuse.cli \
  --input image_list.txt \
  --path_csv results.csv
```

Optional flags:

| Flag | Effect |
|---|---|
| `-p` / `--plot` | Save a summary plot per image |
| `-pp` / `--detailed_plot` | Save a detailed multi-panel plot |
| `-ar` / `--auto_rotate` | Apply EXIF-based rotation before processing |
| `--cache` | Cache expensive computation steps with joblib |
| `--stop-on-error` | Halt on first failure instead of continuing |
| `--skip-failures` | Do not write failed images to the output CSV |

### Using a JSON config (recommended for batch runs)

```bash
python -m lepmuse.cli --config configs/pipeline_battus100_val_unet.json
```

CLI arguments override the config file when both are provided.

---

## Troubleshooting

**`RuntimeError: Loading model … attempted to import from fastcore.dispatch and/or fastcore.transform`**
The `.pkl` was exported with fastai ≤ 2.7 and cannot be loaded by fastai ≥ 2.8. Both `fastcore.dispatch` and `fastcore.transform` are now empty stubs — all transform types have moved to `fasttransform`. The only fix is to retrain and re-export with the current fastai 2.8.7 stack: `python -m segmentation.unet.train --config configs/unet_train_battus100.json`.

**`acc_camvid` is `nan` during training**
All pixels in a batch are background. This can happen with very small batch sizes combined with images that are mostly background. Increase `batch_size` or verify that label masks contain foreground pixels (see the label validation script in §3).

**`torch.OutOfMemoryError` — `beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)`**
This is the self-attention OOM. The UNet decoder's self-attention block creates a 60,000 × 60,000 attention matrix at 800 × 1200 (1/4-resolution feature map = 200 × 300 = 60,000 tokens), requiring **14.4 GB fp32 / 7.2 GB fp16** per sample. No GPU can fit this at training time when full-network gradients are retained. Fix: ensure `self_attention` is `false` in the config (the default). Only enable it at resolutions ≤ 512 × 512.

**`torch.OutOfMemoryError` — generic CUDA out of memory**
If self-attention is already disabled, the OOM is from batch size or image size. Reduce `batch_size` (try halving it), or reduce `image_height` / `image_width`. Enabling `mixed_precision=true` (the default) typically frees 30–40% of activation memory. You can also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m segmentation.unet.train \
  --config configs/unet_train_battus100.json
```

**`FileNotFoundError: UNet weights not found`**
The default weights path is relative to the working directory. Always run commands from the repo root (`/home/rahul/workspace/vision/lepmuse`), or pass an absolute path with `--weights`.

**`ValueError: Expected at least 4 segmentation classes`**
The loaded `.pkl` was trained with fewer than 4 classes (e.g., an older 3-class model). Retrain or point `--weights` to the correct 4-class model.

**`No lepidopteran region found in segmentation mask`**
The model predicted no foreground for the butterfly class. Common causes: image is a crop with no specimen, image resolution is very different from training (800 × 1200), or EXIF rotation is needed (try `--auto_rotate`).

**`foreground_accuracy` is 0.0 for all images in segmentation eval**
The mask files in `--mask-dir` are likely RGB-encoded rather than indexed (pixel values 0–3). Use the `labels/` subdirectory (indexed PNGs), not `labels_rgb/`.

**Model loads but inference produces wrong class order**
Check `datasets/.../codes.txt`. It must read `background lepidopteran tags ruler` (space-separated, in that order). If the order differs, the class index→mask unpacking in [segmentation/unet/infer.py:46](segmentation/unet/infer.py#L46) will assign the wrong mask to the wrong class.

---

## Plugging in a New Model (SAM2 example)

The segmenter is abstracted behind a `Segmenter` protocol defined in [lepmuse/types.py:45](lepmuse/types.py#L45):

```python
class Segmenter(Protocol):
    name: str
    def segment(self, image_rgb: np.ndarray, record: ImageRecord) -> SegmentationResult: ...
```

Any class that satisfies this interface works with the pipeline and CLI without further changes. To add SAM2:

### Step 1 — Create `segmentation/sam2/infer.py`

```python
from __future__ import annotations

import numpy as np
from lepmuse.types import ImageRecord, SegmentationResult


class Sam2Segmenter:
    name = "sam2"

    def __init__(self, weights: str | None = None, device: str = "cuda"):
        # load SAM2 checkpoint here
        ...

    def segment(self, image_rgb: np.ndarray, record: ImageRecord) -> SegmentationResult:
        # generate prompts (e.g., whole-image box) and run SAM2 predictor
        # map SAM2 instance masks → lepidopteran_mask, ruler_mask, tag_mask
        ...
        return SegmentationResult(
            lepidopteran_mask=lepidop_bin,
            ruler_mask=ruler_bin,
            tag_mask=tags_bin,
            model_name=self.name,
        )
```

### Step 2 — Register in the factory

In [lepmuse/segmentation.py](lepmuse/segmentation.py):

```python
def build_segmenter(name: str, weights: str | Path | None = None) -> Segmenter:
    if name == "unet":
        from segmentation.unet.infer import UNetSegmenter
        return UNetSegmenter(weights=weights)
    if name == "sam2":
        from segmentation.sam2.infer import Sam2Segmenter
        return Sam2Segmenter(weights=weights)
    raise ValueError(f"Unsupported segmenter: {name}")
```

### Step 3 — Add `"sam2"` as a valid CLI choice

In [lepmuse/cli.py:19](lepmuse/cli.py#L19):

```python
parser.add_argument("--segmenter", choices=["unet", "sam2"], ...)
```

### Step 4 — Run with the new backend

```bash
python -m lepmuse.cli \
  --config configs/pipeline_battus100_val_unet.json \
  --segmenter sam2 \
  --weights path/to/sam2_checkpoint.pt
```

No changes are needed to the evaluation, measurement, scale detection, or result-writing code.

### SAM2 design considerations

SAM2 produces single-instance masks from prompts rather than a single multi-class prediction. Three workable strategies:

| Strategy | How | Trade-off |
|---|---|---|
| **Box prompt per class** | Run SAM2 once per class with a bounding box derived from a lightweight detector | Most accurate; 4× inference cost |
| **SAM2 for specimen + CV for ruler/tags** | SAM2 for `lepidopteran`, traditional thresholding for ruler and tag regions | Fast; ruler/tag quality limited by heuristics |
| **Fine-tuned SAM2 decoder head** | Train a multi-class head on top of the SAM2 image encoder | Best long-term path; requires retraining infrastructure |

The existing `sam2` conda environment at `/home/rahul/miniconda3/envs/sam2` is already available for development.
