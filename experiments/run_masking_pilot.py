"""Execute the self-contained notebook without Jupyter (e.g. on a GPU node).

Default: visualization/contracts only. --train explicitly enables training.
This runner keeps the notebook as the single source of model/training code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("Masking_pilot.ipynb")


def run_notebook(*, mode="smoke", train=False, seed=42,
                 artifact_dir=None, output_root=None, quiet=False):
    import matplotlib
    matplotlib.use("Agg")
    from IPython.utils.capture import capture_output

    namespace = {"__name__": "__masking_pilot__"}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original_cwd = Path.cwd()
    try:
        os.chdir(NOTEBOOK.parent.parent)
        for number, cell in enumerate(notebook["cells"], 1):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if 'MODE = "smoke"' in source:
                source = source.replace('MODE = "smoke"', f"MODE = {mode!r}", 1)
                source = source.replace("RUN_TRAINING = False", f"RUN_TRAINING = {train!r}", 1)
                source = source.replace("SEED = 42", f"SEED = {seed!r}", 1)
                source = source.replace("ARTIFACT_DIR_OVERRIDE = None",
                                        f"ARTIFACT_DIR_OVERRIDE = {str(artifact_dir)!r}" if artifact_dir else
                                        "ARTIFACT_DIR_OVERRIDE = None", 1)
                source = source.replace("OUTPUT_ROOT_OVERRIDE = None",
                                        f"OUTPUT_ROOT_OVERRIDE = {str(output_root)!r}" if output_root else
                                        "OUTPUT_ROOT_OVERRIDE = None", 1)
            if not quiet:
                print(f"Executing notebook cell {number}", flush=True)
            # Suppress rich tables outside Jupyter but show progress text immediately.
            with capture_output(stdout=quiet, stderr=quiet, display=True):
                exec(compile(source, f"{NOTEBOOK.name}:cell{number}", "exec"), namespace)
        return namespace
    finally:
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = run_notebook(mode=args.mode, train=args.train, seed=args.seed,
                          artifact_dir=args.artifact_dir.resolve() if args.artifact_dir else None,
                          output_root=args.output_root.resolve() if args.output_root else None)
    print("Completed. Output:", result.get("RUN_DIR"), flush=True)


if __name__ == "__main__":
    main()
