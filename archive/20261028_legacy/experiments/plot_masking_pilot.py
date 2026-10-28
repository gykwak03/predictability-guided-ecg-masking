"""Plot saved MAE training histories without running training again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="Saved run directory; defaults to the latest pilot run.")
    parser.add_argument("--window", type=int, default=50,
                        help="Trailing moving-average window in updates.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")

    run_dir = args.run_dir
    if run_dir is None:
        root = Path(__file__).resolve().parent / "artifacts/masking_pilot"
        candidates = []
        for config_path in root.glob("*/run_config.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("mode") == "pilot" and list(config_path.parent.glob("*/pretrain_history.csv")):
                candidates.append(config_path.parent)
        if not candidates:
            parser.error("No pilot histories found. Pass the saved run directory explicitly.")
        run_dir = max(candidates, key=lambda path: path.name)
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    histories = []
    summary = []
    for path in sorted(run_dir.glob("*/pretrain_history.csv")):
        patch, ratio = path.parent.name.split("_")
        patch, ratio = int(patch[1:]), int(ratio[1:])
        history = pd.read_csv(path).sort_values("step")
        if history.empty or not np.isfinite(history[["step", "masked_mse"]].to_numpy()).all():
            raise ValueError(f"Empty or non-finite history: {path}")
        width = min(args.window, len(history))
        history["smoothed"] = history.masked_mse.rolling(width, min_periods=1).mean()
        histories.append((patch, ratio, history))
        summary.append(dict(patch_samples=patch, mask_ratio=ratio / 100,
                            updates=len(history), window_updates=width,
                            first_window_mse=history.masked_mse.iloc[:width].mean(),
                            last_window_mse=history.masked_mse.iloc[-width:].mean()))
    if not histories:
        parser.error(f"No pretrain_history.csv files in {run_dir}")

    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patches = sorted({patch for patch, _, _ in histories})
    colors = {ratio: f"C{i}" for i, ratio in enumerate(sorted({ratio for _, ratio, _ in histories}))}
    expected_steps = config.get("profile", {}).get("steps", "unknown")
    print(f"Run: {run_dir}\nMode: {config['mode']}; configured updates: {expected_steps}")
    print(pd.DataFrame(summary).to_string(index=False))

    for tail in (False, True):
        fig, axes = plt.subplots(1, len(patches), figsize=(5 * len(patches), 4.5),
                                 sharey=True, squeeze=False)
        for ax, patch in zip(axes[0], patches):
            for current_patch, ratio, history in histories:
                if current_patch != patch:
                    continue
                selected = history
                if tail:
                    cutoff = history.step.min() + 0.8 * (history.step.max() - history.step.min())
                    selected = history[history.step >= cutoff]
                ax.plot(selected.step, selected.masked_mse, color=colors[ratio], alpha=0.18,
                        linewidth=0.7)
                ax.plot(selected.step, selected.smoothed, color=colors[ratio],
                        label=f"Mask {ratio}%")
            ax.set_title(f"Patch {patch} samples")
            ax.set_xlabel("Training update")
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0, 0].set_ylabel("Training masked MSE")
        region = "last 20% of updates" if tail else "all updates"
        fig.suptitle(f"{config['mode']} | {region} | trailing mean up to {args.window} updates")
        fig.tight_layout()
        destination = output_dir / ("training_curves_tail.png" if tail else "training_curves.png")
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        print(f"Saved: {destination}")
    print("Faint lines: raw batch losses; solid lines: moving averages.")
    print("Validation MSE is evaluated only after training; no validation curve was logged.")


if __name__ == "__main__":
    main()
