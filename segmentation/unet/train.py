from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    train_data_path: str = "datasets/battus100/training_images"
    # Provides the exported .pkl filename; all run artifacts live under run_dir.
    output: str = "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl"
    epochs: int = 50
    batch_size: int = 8
    image_height: int = 800
    image_width: int = 1200
    architecture: str = "resnet18"
    cuda_visible_devices: str | None = None
    num_workers: int = 4
    self_attention: bool = False    # True requires O(H²W²) GPU memory — infeasible at 800×1200
    mixed_precision: bool = True    # fp16 training; halves activation memory, no accuracy loss
    valid_pct: float = 0.2
    seed: int = 42
    # ── Experiment organisation ───────────────────────────────────────────────
    # Every run writes all artifacts to: <runs_dir>/<run_name>/
    #   events.out.tfevents.*   — TensorBoard scalars + prediction images
    #   history.csv             — per-epoch metrics
    #   best_ckpt.pth           — best checkpoint weights
    #   train.log               — full stdout + stderr transcript
    #   <output stem>.pkl       — exported inference learner
    runs_dir: str = "runs"
    run_name: str | None = None     # None → auto-generated from config fields
    # ── Logging ──────────────────────────────────────────────────────────────
    tensorboard: bool = True
    log_preds: bool = True          # write prediction tiles to TensorBoard Images tab
    csv_log: bool = True
    # ── Checkpointing & early stopping ───────────────────────────────────────
    checkpoint_metric: str = "acc_camvid"
    early_stopping_patience: int | None = None
    # ── Augmentation ─────────────────────────────────────────────────────────
    # Position invariance is achieved through min_zoom=0.75 (zoom-out places the
    # specimen at different positions within the frame) rather than an oversample+crop
    # stage, which would leave validation batches at a different resolution and cause
    # GPU allocator fragmentation between the frozen (val only) and unfrozen epochs.
    aug_oversample: float = 1.0     # keep at 1.0; zoom handles position variation
    aug_min_zoom: float = 0.75      # zoom out up to 25% — scale invariance for specimen size
    aug_max_zoom: float = 1.35      # zoom in up to 35%
    aug_max_rotate: float = 90.0    # ±90° — ruler/specimen can appear in any orientation
    aug_max_warp: float = 0.35      # perspective warp — camera angle variation
    aug_max_lighting: float = 0.35  # brightness + contrast — lamp aging between museum batches
    aug_p_affine: float = 0.75      # probability of applying any affine transform per image
    aug_p_lighting: float = 0.75    # probability of applying any lighting transform per image
    aug_random_erasing: bool = True # erase random patches (p=0.3) — partial ruler/tag occlusion
    aug_random_erasing_p: float = 0.3


def load_config(path: str | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    return TrainConfig(**json.loads(Path(path).read_text()))


def _build_run_name(config: TrainConfig) -> str:
    """Auto-generated name encodes the key config dimensions for easy identification."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{ts}"
        f"_{config.architecture}"
        f"_b{config.batch_size}"
        f"_e{config.epochs}"
        f"_s{config.image_height}x{config.image_width}"
    )


class _Tee:
    """Mirror writes to multiple streams (e.g. stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def fileno(self) -> int:
        return self._streams[0].fileno()

    def isatty(self) -> bool:
        return False


def train(config: TrainConfig) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # headless backend — must be set before any other matplotlib import

    import numpy as np
    from fastai.callback.progress import CSVLogger
    from fastai.callback.tensorboard import TensorBoardCallback
    from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
    from fastai.vision.all import (
        RandomErasing,
        Resize,
        SegmentationDataLoaders,
        aug_transforms,
        get_image_files,
        resnet18,
        resnet34,
        unet_learner,
    )

    if config.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    # Expandable segments prevent address-space fragmentation between epochs with
    # different active tensor sizes (e.g. frozen vs unfrozen forward passes).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Run directory ─────────────────────────────────────────────────────────
    # resolve() → absolute path. fastai computes learner.path / model_dir when
    # saving checkpoints; an absolute model_dir is used as-is, keeping all
    # artifacts in run_dir rather than under the dataloader root.
    run_name = config.run_name or _build_run_name(config)
    run_dir = (Path(config.runs_dir) / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Transcript log ────────────────────────────────────────────────────────
    log_path = run_dir / "train.log"
    log_file = open(log_path, "w", buffering=1)   # line-buffered: visible live via tail -f
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)

    try:
        _train_inner(
            config, run_dir, log_path,
            resnet18, resnet34, unet_learner,
            RandomErasing, Resize,
            SegmentationDataLoaders, aug_transforms, get_image_files,
            CSVLogger, TensorBoardCallback, EarlyStoppingCallback, SaveModelCallback,
            np,
        )
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()

    return run_dir / Path(config.output).name


def _train_inner(
    config, run_dir, log_path,
    resnet18, resnet34, unet_learner,
    RandomErasing, Resize,
    SegmentationDataLoaders, aug_transforms, get_image_files,
    CSVLogger, TensorBoardCallback, EarlyStoppingCallback, SaveModelCallback,
    np,
):
    print(datetime.now(), "\t Run directory   →", run_dir)
    print(datetime.now(), "\t Transcript log  →", log_path)
    print(datetime.now(), "\t   tensorboard --logdir", run_dir.parent)

    train_path = Path(config.train_data_path)
    codes = np.loadtxt(train_path / "codes.txt", dtype="str")
    name2id = {value: key for key, value in enumerate(codes)}
    background_code = name2id["background"]
    print(datetime.now(), "\t Unique labels:", name2id)

    def acc_camvid(inp, targ):
        targ = targ.squeeze(1)
        mask = targ != background_code
        if not mask.any():
            return inp.new_tensor(float("nan"))
        return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()

    def label_func(image):
        return train_path / "labels" / f"{image.stem}.png"

    files = get_image_files(train_path / "images")
    if not files:
        raise FileNotFoundError(f"No training images found in {train_path / 'images'}")
    print(datetime.now(), "\t Loading segmentation masks data for images:", len(files))

    # ── Augmentation pipeline ─────────────────────────────────────────────────
    # item_tfms resize to the EXACT target size for both train and val.
    # Using an oversample+RandomCrop scheme would leave validation batches at the
    # larger oversampled resolution (RandomCrop is split_idx=0, train only), which
    # causes train/val size mismatch → allocator fragmentation → OOM at epoch 1
    # when all encoder layers are unfrozen and activation memory peaks.
    # Position invariance is instead provided by min_zoom=0.75: at 75% zoom the
    # specimen occupies a smaller portion of the frame and is effectively shifted.
    item_tfms = Resize((config.image_height, config.image_width))

    # aug_transforms, RandomErasing: all have split_idx=0 — training batches only.
    batch_tfms_list = []
    batch_tfms_list += aug_transforms(
        mult=1.0,
        do_flip=True,
        flip_vert=True,
        max_rotate=config.aug_max_rotate,
        min_zoom=config.aug_min_zoom,
        max_zoom=config.aug_max_zoom,
        max_lighting=config.aug_max_lighting,
        max_warp=config.aug_max_warp,
        p_affine=config.aug_p_affine,
        p_lighting=config.aug_p_lighting,
    )

    if config.aug_random_erasing:
        # Applied only to the input image (x), not the segmentation mask (y).
        # Simulates partial ruler occlusion, tags overlapping the specimen, or
        # image regions lost to glare — trains the model to use context not just
        # local texture.
        batch_tfms_list.append(
            RandomErasing(p=config.aug_random_erasing_p, sl=0.02, sh=0.2, min_aspect=0.3)
        )

    print(
        datetime.now(),
        f"\t Augmentation    →"
        f"  zoom=[{config.aug_min_zoom}, {config.aug_max_zoom}]"
        f"  rotate=±{config.aug_max_rotate}°"
        f"  warp={config.aug_max_warp}"
        f"  lighting={config.aug_max_lighting}"
        f"  erasing={config.aug_random_erasing}(p={config.aug_random_erasing_p})",
    )

    dls = SegmentationDataLoaders.from_label_func(
        train_path,
        bs=config.batch_size,
        fnames=files,
        label_func=label_func,
        valid_pct=config.valid_pct,
        seed=config.seed,
        codes=codes,
        num_workers=config.num_workers,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms_list,
    )

    arch = {"resnet18": resnet18, "resnet34": resnet34}[config.architecture]

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cbs = []

    # Saves run_dir/best_ckpt.pth whenever checkpoint_metric improves.
    # fastai reloads the best checkpoint automatically after fine_tune() completes.
    cbs.append(
        SaveModelCallback(
            monitor=config.checkpoint_metric,
            comp=np.greater,
            fname="best_ckpt",
            reset_on_fit=True,
        )
    )

    if config.csv_log:
        history_path = run_dir / "history.csv"
        cbs.append(CSVLogger(fname=history_path, append=False))
        print(datetime.now(), "\t CSV log         →", history_path)

    if config.tensorboard:
        cbs.append(
            TensorBoardCallback(
                log_dir=run_dir,
                trace_model=False,
                log_preds=config.log_preds,
                n_preds=4,
            )
        )
        print(datetime.now(), "\t TensorBoard     →", run_dir)

    if config.early_stopping_patience is not None:
        cbs.append(
            EarlyStoppingCallback(
                monitor=config.checkpoint_metric,
                comp=np.greater,
                patience=config.early_stopping_patience,
            )
        )
        print(datetime.now(), "\t Early stopping patience =", config.early_stopping_patience)

    # ── Training ──────────────────────────────────────────────────────────────
    learner = unet_learner(
        dls,
        arch,
        pretrained=True,
        metrics=acc_camvid,
        self_attention=config.self_attention,
        model_dir=run_dir,   # absolute path — prevents fastai prepending learner.path
    )
    if config.mixed_precision:
        learner = learner.to_fp16()
        print(datetime.now(), "\t Mixed precision (fp16) enabled")
    print(datetime.now(), "\t Train model for epochs=", config.epochs)
    learner.fine_tune(config.epochs, cbs=cbs)

    # Export: SaveModelCallback has already reloaded the best checkpoint.
    export_path = run_dir / Path(config.output).name
    learner.export(export_path)
    print(datetime.now(), "\t Exported model  →", export_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Battus UNet segmentation model.")
    parser.add_argument("--config")
    parser.add_argument("--train-data-path")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--runs-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--no-log-preds", dest="log_preds", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--checkpoint-metric")
    # Augmentation overrides
    parser.add_argument("--aug-oversample", type=float)
    parser.add_argument("--aug-min-zoom", type=float)
    parser.add_argument("--aug-max-zoom", type=float)
    parser.add_argument("--aug-max-rotate", type=float)
    parser.add_argument("--aug-max-warp", type=float)
    parser.add_argument("--aug-max-lighting", type=float)
    parser.add_argument("--no-random-erasing", dest="aug_random_erasing", action="store_false")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    updates = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    if updates:
        config = replace(config, **updates)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
