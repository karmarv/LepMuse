from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Force non-interactive Agg backend before any matplotlib import.
# Must be set before pyplot is first imported anywhere in the process.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from .config import load_config, merge_config
from .inputs import read_records
from .results import finalize_csv
from .segmentation import build_segmenter

# Matches ANSI escape sequences (colours, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b.")


def _clean_for_log(text: str) -> str:
    """Strip ANSI codes and collapse carriage-return overwrites for log files.

    Progress bars write partial lines ending with \\r to overwrite themselves
    in a terminal.  In a log file that produces unreadable noise — we keep
    only the last segment after the final \\r so each logical line appears once.
    """
    text = _ANSI_RE.sub("", text)
    if "\r" in text:
        parts = text.split("\r")
        trailing_nl = "\n" if text.endswith("\n") else ""
        last = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else "")
        text = last.rstrip("\n") + trailing_nl
    return text


class _Tee:
    """Mirror writes to the terminal and a clean log file."""

    def __init__(self, terminal, logfile):
        self._terminal = terminal
        self._logfile = logfile

    def write(self, data: str) -> int:
        self._terminal.write(data)
        cleaned = _clean_for_log(data)
        if cleaned:
            self._logfile.write(cleaned)
        return len(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._logfile.flush()

    def fileno(self) -> int:
        return self._terminal.fileno()

    def isatty(self) -> bool:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LepMuse UNet measurement inference.")
    parser.add_argument("--config", help="JSON config file. CLI arguments override config values.")
    parser.add_argument("-i", "--input", dest="input", help="Image, folder, text manifest, or CSV manifest.")
    parser.add_argument("-o", "--output_folder", help="Folder for optional artifact outputs.")
    parser.add_argument("-csv", "--path_csv", help="Output CSV path.")
    parser.add_argument("-s", "--stage", choices=["binarization", "ruler_detection", "measurements"])
    parser.add_argument("--segmenter", choices=["unet"], help="Segmentation backend.")
    parser.add_argument("--weights", help="UNet FastAI exported model path.")
    parser.add_argument("-p", "--plot", action="store_true", default=None)
    parser.add_argument("-pp", "--detailed_plot", action="store_true", default=None)
    parser.add_argument("-ar", "--auto_rotate", action="store_true", default=None)
    parser.add_argument("-dpi", "--dpi", type=int)
    parser.add_argument("--cache", action="store_true", default=None)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false", default=None)
    parser.add_argument("--skip-failures", dest="write_failures", action="store_false", default=None)
    parser.add_argument("--eval-config", dest="eval_config", help="ResultEvalConfig JSON for post-run evaluation.")
    parser.add_argument("--workers", dest="num_workers", type=int, help="Number of parallel workers (default: 1).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = merge_config(load_config(args.config), args)

    # Mirror stdout + stderr to pipeline.log inside the output folder.
    output_dir = Path(config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    log_file = open(log_path, "w", buffering=1)  # line-buffered: visible live via tail -f
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    print(f"Pipeline log: {log_path}")

    try:
        if config.cache:
            import joblib
            from . import cache

            cache.memory = joblib.Memory("./cachedir", verbose=0)
        records = read_records(config.input)
        segmenter = build_segmenter(config.segmenter, weights=config.weights)
        from .pipeline import PipelineRunner

        PipelineRunner(segmenter=segmenter, config=config).run(records)
        finalize_csv(config.path_csv)

        if config.eval_config:
            from dataclasses import replace as dc_replace
            from .evaluate import load_config as load_eval_config, evaluate_results
            eval_cfg = load_eval_config(config.eval_config)
            # Always point predicted at the pipeline's own output CSV.
            eval_cfg = dc_replace(eval_cfg, predicted=config.path_csv)
            print(f"\n--- Evaluation ---")
            evaluate_results(eval_cfg)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
