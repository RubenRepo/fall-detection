"""Minimal experiment logger """
import csv
import datetime
import subprocess
from pathlib import Path

FIELDS = ["date", "goal", "data_split", "model", "seed", "metrics", "git_commit", "notes"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"


def log_experiment(goal, data_split, model, seed, metrics, notes="", path="experiments/log.csv"):
    row = {
        "date": datetime.date.today().isoformat(),
        "goal": goal,
        "data_split": data_split,
        "model": model,
        "seed": seed,
        "metrics": metrics,
        "git_commit": _git_commit(),
        "notes": notes,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)