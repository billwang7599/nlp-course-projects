"""Grid-search the LSTM and Transformer configs, one trial at a time (not part of the submission).

Usage: uv run python sweep.py [--grid size|depth] [--part 1|2|both] [--epochs 8] [--mirror DIR]

Each trial gets sweep/<part>/<name>/ with its own config.yaml, checkpoint and metrics.csv.
Results are appended to sweep/summary.csv (size grid) or sweep/summary_depth.csv (depth grid)
after every trial.

size grid:  hidden/d_model x lr, all trials run --epochs epochs. Dev PPL at the shorter epoch
            count is read from the same run (valid because Adam's learning rate is constant).
--mirror:   (depth grid) after each trial, copy only its small files (config, metrics.csv, log, plot)
            and the summary CSV to DIR, e.g. on Google Drive. Checkpoints and W&B logs stay local.
            On restart, finished trials found in DIR are restored and skipped, so a Colab
            disconnect costs at most the trial that was running.
depth grid: trades layers against epochs. Each trial runs the epoch count that should take about
            35 minutes on a T4, then reports the best dev PPL reachable within 35 and 39 minutes.
"""

import argparse
import csv
import itertools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

SHORT_EPOCHS = 4
BUDGETS = (35, 39)

SIZE_GRIDS = {
    1: {
        "base": "configs/lstm.yaml",
        "script": "run_part1.py",
        "axes": {
            "hidden": ("model.hidden_dim", [512, 768]),
            "lr": ("optimizer.learning_rate", [1e-3, 2e-3]),
        },
    },
    2: {
        "base": "configs/transformer.yaml",
        "script": "run_part2.py",
        "axes": {
            "d": ("model.d_model", [256, 384]),
            "lr": ("optimizer.learning_rate", [3e-4, 1e-3]),
        },
    },
}

# (layers, hidden_dim or d_model, epochs); epochs are guessed from the size-grid timings to land
# near 35 minutes. Learning rate comes from the base config (2e-3 LSTM, 1e-3 Transformer).
DEPTH_TRIALS = {
    1: {
        "base": "configs/lstm.yaml",
        "script": "run_part1.py",
        "trials": [(1, 1024, 10), (3, 512, 14), (3, 640, 9), (4, 512, 10), (4, 384, 18)],
    },
    2: {
        "base": "configs/transformer.yaml",
        "script": "run_part2.py",
        "trials": [(6, 384, 7), (6, 320, 10), (8, 256, 11), (12, 256, 7), (2, 512, 11)],
    },
}


def set_path(cfg, dotted, value):
    *parents, leaf = dotted.split(".")
    for key in parents:
        cfg = cfg[key]
    cfg[leaf] = value


def write_trial(part, name, cfg, epochs, root):
    """Point all of a trial's outputs next to its config.yaml, so trials never overwrite each other."""
    if part == 2:
        cfg["model"]["dim_feedforward"] = 4 * cfg["model"]["d_model"]
    cfg["training"]["epochs"] = epochs
    cfg["wandb"]["name"] = f"part{part}-{name}"
    cfg["wandb"]["dir"] = "logs"
    cfg["output"].update(
        artifact_dir="out", metrics_file="metrics.csv", summary_file="summary.json", plot_file="loss.png"
    )
    trial_dir = root / f"part{part}" / name
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def make_size_trial(part, names, values, epochs, root):
    spec = SIZE_GRIDS[part]
    with open(spec["base"]) as f:
        cfg = yaml.safe_load(f)
    name = "_".join(f"{n}{v}" for n, v in zip(names, values))
    for (_, (dotted, _)), value in zip(spec["axes"].items(), values):
        set_path(cfg, dotted, value)
    return name, write_trial(part, name, cfg, epochs, root)


def make_depth_trial(part, layers, width, epochs, root):
    with open(DEPTH_TRIALS[part]["base"]) as f:
        cfg = yaml.safe_load(f)
    width_key = "hidden_dim" if part == 1 else "d_model"
    cfg["model"]["num_layers"] = layers
    cfg["model"][width_key] = width
    name = f"L{layers}_w{width}_ep{epochs}"
    return name, write_trial(part, name, cfg, epochs, root)


def dev_ppls(metrics_file):
    with open(metrics_file) as f:
        return [float(r["ppl"]) for r in csv.DictReader(f) if r["split"] == "dev"]


def run_trial(part, script, name, config_path):
    trial_dir = config_path.parent
    print(f"=== part{part} {name} ===", flush=True)
    start = time.time()
    with open(trial_dir / "train.log", "w") as log:
        result = subprocess.run(
            [sys.executable, script, "--config", str(config_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MPLBACKEND": "agg"},  # Colab sets a notebook-only backend
        )
    minutes = (time.time() - start) / 60
    if result.returncode != 0:
        print(f"FAILED {name}, see {trial_dir / 'train.log'}", flush=True)
        return None, minutes
    return dev_ppls(trial_dir / "metrics.csv"), minutes


MIRRORED_FILES = ("config.yaml", "metrics.csv", "train.log", "loss.png")


def restore_from_mirror(mirror, root, part, name):
    """Copy a finished trial's small files back from the mirror so the skip check sees it."""
    source = mirror / f"part{part}" / name
    if (source / "metrics.csv").exists():
        target = root / f"part{part}" / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in MIRRORED_FILES:
            if (source / filename).exists():
                shutil.copy2(source / filename, target / filename)


def save_to_mirror(mirror, root, part, name, summary_name):
    """Summary first, trial files last: a cut-off here repeats a trial rather than losing its row."""
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / summary_name, mirror / summary_name)
    target = mirror / f"part{part}" / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in MIRRORED_FILES:
        source = root / f"part{part}" / name / filename
        if source.exists():
            shutil.copy2(source, target / filename)


def append_row(summary_path, row):
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


def run_size_grid(parts, epochs, root):
    for part in parts:
        spec = SIZE_GRIDS[part]
        names = list(spec["axes"])
        for values in itertools.product(*(vals for _, vals in spec["axes"].values())):
            name, config_path = make_size_trial(part, names, values, epochs, root)
            if (config_path.parent / "metrics.csv").exists():
                print(f"skip {name} (already done)", flush=True)
                continue
            ppls, minutes = run_trial(part, spec["script"], name, config_path)
            if ppls is None:
                continue
            append_row(
                root / "summary.csv",
                {
                    "part": part,
                    "trial": name,
                    "epochs": epochs,
                    "minutes": round(minutes, 1),
                    "est_minutes_short": round(minutes * SHORT_EPOCHS / epochs, 1),
                    "best_ppl": round(min(ppls), 3),
                    f"ppl_ep{SHORT_EPOCHS}": round(ppls[SHORT_EPOCHS - 1], 3) if len(ppls) >= SHORT_EPOCHS else "",
                },
            )


def run_depth_grid(parts, root, mirror):
    if mirror and (mirror / "summary_depth.csv").exists():
        shutil.copy2(mirror / "summary_depth.csv", root / "summary_depth.csv")
    for part in parts:
        spec = DEPTH_TRIALS[part]
        for layers, width, epochs in spec["trials"]:
            name, config_path = make_depth_trial(part, layers, width, epochs, root)
            if mirror:
                restore_from_mirror(mirror, root, part, name)
            if (config_path.parent / "metrics.csv").exists():
                print(f"skip {name} (already done)", flush=True)
                continue
            ppls, minutes = run_trial(part, spec["script"], name, config_path)
            if ppls is None:
                continue
            row = {
                "part": part,
                "trial": name,
                "layers": layers,
                "width": width,
                "epochs": epochs,
                "minutes": round(minutes, 1),
                "best_ppl": round(min(ppls), 3),
                "best_epoch": ppls.index(min(ppls)) + 1,
            }
            for budget in BUDGETS:
                # Epochs that fit in the budget, assuming time grows in proportion to epochs.
                k = max(1, min(epochs, int(budget * epochs / minutes)))
                row[f"epochs_in_{budget}min"] = k
                row[f"ppl_in_{budget}min"] = round(min(ppls[:k]), 3)
            append_row(root / "summary_depth.csv", row)
            if mirror:
                save_to_mirror(mirror, root, part, name, "summary_depth.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="size", choices=["size", "depth"])
    parser.add_argument("--part", default="both", choices=["1", "2", "both"])
    parser.add_argument("--epochs", type=int, default=8, help="size grid only")
    parser.add_argument("--mirror", type=Path, default=None, help="depth grid only; see module docstring")
    args = parser.parse_args()

    root = Path("sweep")
    root.mkdir(exist_ok=True)
    parts = [1, 2] if args.part == "both" else [int(args.part)]

    if args.grid == "size":
        run_size_grid(parts, args.epochs, root)
        summary_path = root / "summary.csv"
    else:
        run_depth_grid(parts, root, args.mirror)
        summary_path = root / "summary_depth.csv"

    if summary_path.exists():
        print("\n" + summary_path.read_text())


if __name__ == "__main__":
    main()
