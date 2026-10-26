"""Run the reviewed preprocessing notebook without Jupyter.

The notebook remains the source of signal processing and metadata rules.
Builds a separate 0.67–50 Hz dataset; never filters the existing 40 Hz arrays
again and never overwrites an existing artifact with different settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "PTBXL_preprocessing.ipynb"


def run(*, data_dir, output_dir, limit_per_split=None, progress_every=500):
    import matplotlib
    matplotlib.use("Agg")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "__preprocessing_cli__"}
    old_cwd = Path.cwd()
    try:
        os.chdir(HERE.parent)
        for number in [4, 6, 8, 10, 13, 15, 17, 19, 21, 26, 28, 30]:
            cell = notebook["cells"][number-1]
            if cell["cell_type"] != "code":
                raise RuntimeError(f"Notebook cell {number} changed; review CLI interface")
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell{number}", "exec"), ns)
            if number == 4:
                ns["display"] = lambda *args, **kwargs: None
            if number == 6:
                ns.update(DATA_DIR=Path(data_dir).resolve(), OUTPUT_DIR=Path(output_dir).resolve(),
                          LOW_HZ=.67, HIGH_HZ=50., SOURCE_FS=500, TARGET_FS=250,
                          NORMALIZATION="per-record", LIMIT_PER_SPLIT=limit_per_split,
                          PROGRESS_EVERY=progress_every)
                print(json.dumps({"data_dir": str(ns["DATA_DIR"]), "output_dir": str(ns["OUTPUT_DIR"]),
                                  "bandpass_hz": [.67, 50.], "sampling_rate_hz": 250,
                                  "limit_per_split": limit_per_split}, indent=2), flush=True)
        manifest = ns["manifest"]
        if manifest["config"]["bandpass_high"] != 50.:
            raise RuntimeError("Expected new 50 Hz dataset")
        print("Validated artifact:", ns["OUTPUT_DIR"], flush=True)
        return manifest
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parents[1] / "ptb-xl/ptb-xl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-per-split", type=int, help="Small pipeline check; use a separate output directory")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.output_dir is None:
        suffix = "_smoke" if args.limit_per_split is not None else ""
        args.output_dir = HERE / f"artifacts/ptbxl_250hz_bp067_50{suffix}"
    run(**vars(args))


if __name__ == "__main__":
    main()
