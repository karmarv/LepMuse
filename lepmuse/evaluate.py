from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResultEvalConfig:
    actual: str = "datasets/battus100/val_images/manual_measurements_gt.csv"
    predicted: str = "datasets/battus100/results/val_images/results.csv"
    image_name_column: str = "image_id"
    actual_left_column: str = "left_wing (mm)"
    actual_right_column: str = "right_wing (mm)"
    predicted_left_column: str = "left_wing (mm)"
    predicted_right_column: str = "right_wing (mm)"
    sd_threshold: float = 2.0
    output_plot: str = "datasets/battus100/results/result_plot.png"
    output_comparison_csv: str = "datasets/battus100/results/comparison.csv"
    output_outliers_csv: str = "datasets/battus100/results/outliers.csv"
    copy_outliers_from: str | None = None
    copy_outliers_to: str | None = None


def load_config(path: str | None) -> ResultEvalConfig:
    if path is None:
        return ResultEvalConfig()
    return ResultEvalConfig(**json.loads(Path(path).read_text()))


def evaluate_results(config: ResultEvalConfig) -> dict[str, float | int]:
    actual = _read_table(config.actual)
    actual = actual[[config.image_name_column, config.actual_left_column, config.actual_right_column]]
    actual = actual.rename(
        columns={
            config.image_name_column: "image_id",
            config.actual_left_column: "actual_left",
            config.actual_right_column: "actual_right",
        }
    )

    predicted = pd.read_csv(config.predicted)
    predicted = predicted.rename(
        columns={
            config.predicted_left_column: "predicted_left",
            config.predicted_right_column: "predicted_right",
        }
    )
    both = pd.merge(actual, predicted, on="image_id", how="inner")
    both["left_diff"] = both["predicted_left"] - both["actual_left"]
    both["right_diff"] = both["predicted_right"] - both["actual_right"]

    all_diffs = pd.concat([both["right_diff"], both["left_diff"]]).dropna()
    mean = float(np.mean(all_diffs))
    sd = float(np.std(all_diffs))
    lower = mean - config.sd_threshold * sd
    upper = mean + config.sd_threshold * sd

    both["left_SD"] = (both["left_diff"] - mean) / sd if sd else 0
    both["right_SD"] = (both["right_diff"] - mean) / sd if sd else 0
    both["is_outlier"] = (abs(both["left_SD"]) > config.sd_threshold) | (abs(both["right_SD"]) > config.sd_threshold)

    inlier_diffs = all_diffs[(all_diffs >= lower) & (all_diffs <= upper)]

    stats = {
        "mean_difference": mean,
        "difference_sd": sd,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_measurements": int(len(all_diffs[(all_diffs < lower) | (all_diffs > upper)])),
        "outlier_images": int(np.count_nonzero(both["is_outlier"])),
        "matched_images": int(len(both)),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")

    _write_plot(inlier_diffs, both, stats, config.output_plot)
    _write_csv(both, config.output_comparison_csv)
    _write_csv(both[both["is_outlier"]].drop(columns=["is_outlier"]), config.output_outliers_csv)
    _copy_outliers(both, config)
    return stats


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_plot(
    inlier_diffs: pd.Series,
    both: pd.DataFrame,
    stats: dict,
    output_plot: str | Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    output_plot = Path(output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.38, wspace=0.32, top=0.93)

    stat_str = (
        f"n={stats['matched_images']}  "
        f"mean={stats['mean_difference']:+.2f} mm  "
        f"sd={stats['difference_sd']:.2f} mm  "
        f"outliers={stats['outlier_images']} img"
    )
    fig.suptitle(
        f"Measurement Evaluation Report  [{stat_str}]",
        fontsize=12, fontweight="bold",
    )

    # ── [0,0] Error histogram (inliers only) ─────────────────────────────────
    ax1 = axes[0, 0]
    inlier_diffs.hist(bins="auto", ax=ax1, color="steelblue", edgecolor="white")
    ax1.axvline(stats["mean_difference"], color="orange", linestyle="--",
                linewidth=1.5, label=f'mean={stats["mean_difference"]:+.2f}')
    ax1.set_xlabel("Predicted − Actual (mm)")
    ax1.set_ylabel("Number of images")
    ax1.set_title("Error distribution (inliers)")
    ax1.legend(fontsize=8)

    # ── [0,1] Scatter predicted vs actual — left wing ────────────────────────
    inlier_mask = ~both["is_outlier"]
    _scatter_wing(axes[0, 1], both[inlier_mask], both[~inlier_mask],
                  "actual_left", "predicted_left", "Left wing: predicted vs actual")

    # ── [1,0] Scatter predicted vs actual — right wing ───────────────────────
    _scatter_wing(axes[1, 0], both[inlier_mask], both[~inlier_mask],
                  "actual_right", "predicted_right", "Right wing: predicted vs actual")

    # ── [1,1] Per-side error box plot ────────────────────────────────────────
    ax4 = axes[1, 1]
    box_data = [both["left_diff"].dropna().values, both["right_diff"].dropna().values]
    bp = ax4.boxplot(box_data, labels=["Left wing", "Right wing"], patch_artist=True,
                     medianprops={"color": "orange", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["#4c8cbe", "#e07b54"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax4.set_ylabel("Predicted − Actual (mm)")
    ax4.set_title("Error by wing side")

    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_wing(ax, inliers, outliers, col_actual, col_pred, title):
    all_vals = pd.concat([inliers[col_actual], inliers[col_pred],
                          outliers[col_actual], outliers[col_pred]]).dropna()
    lo, hi = all_vals.min(), all_vals.max()
    pad = (hi - lo) * 0.05
    ref = [lo - pad, hi + pad]
    ax.plot(ref, ref, "k--", linewidth=0.8, label="1:1")
    if len(inliers):
        ax.scatter(inliers[col_actual], inliers[col_pred], s=18, alpha=0.7,
                   color="steelblue", label="inlier")
    if len(outliers):
        ax.scatter(outliers[col_actual], outliers[col_pred], s=30, alpha=0.9,
                   color="tomato", marker="x", linewidths=1.5, label="outlier")
    ax.set_xlim(ref)
    ax.set_ylim(ref)
    ax.set_xlabel("Actual (mm)")
    ax.set_ylabel("Predicted (mm)")
    ax.set_title(f"{title}: predicted vs actual")
    ax.legend(fontsize=7)


def _write_csv(df: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _copy_outliers(df: pd.DataFrame, config: ResultEvalConfig) -> None:
    if not config.copy_outliers_from or not config.copy_outliers_to:
        return
    source = Path(config.copy_outliers_from)
    target = Path(config.copy_outliers_to)
    target.mkdir(parents=True, exist_ok=True)
    for image_name in df[df["is_outlier"]]["image_id"]:
        image_path = source / image_name
        if image_path.exists():
            shutil.copy(image_path, target / image_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted measurements against manual labels.")
    parser.add_argument("--config")
    parser.add_argument("-a", "--actual")
    parser.add_argument("-p", "--predicted")
    parser.add_argument("-n", "--image-name-column")
    parser.add_argument("-l", "--actual-left-column")
    parser.add_argument("-r", "--actual-right-column")
    parser.add_argument("--output-plot")
    parser.add_argument("--output-comparison-csv")
    parser.add_argument("--output-outliers-csv")
    parser.add_argument("-sd", "--sd-threshold", type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    updates = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    if updates:
        config = replace(config, **updates)
    evaluate_results(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
