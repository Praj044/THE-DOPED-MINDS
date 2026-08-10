"""
Builds a train/validation split that keeps whole capture "bursts" (screenshots
taken seconds apart, almost certainly showing near-identical or highly similar
frames of the same defect instance) together on one side of the split. This
prevents the leakage we found in the plain random 80/20 split, where
near-duplicate frames from the same burst landed on both sides.

Evidence for the grouping signal (verified against the real DATASET folder,
not assumed):
  - All 324 filenames match the pattern "Screenshot YYYY-MM-DD HHMMSS.png"
    (0 unmatched files across all 8 classes) -- this is a real timestamp,
    not a guess.
  - The distribution of gaps between consecutive same-class timestamps is
    cleanly bimodal: 276 gaps <=10s (within-burst), a completely EMPTY zone
    from 30-60s (zero gaps fall there), then the next cluster starts at 73s+
    (between-burst breaks). Any threshold in [30, 73) seconds produces an
    identical grouping, so 60s is a safe, non-arbitrary choice.

This script does NOT touch the original dataset folder -- it only reads from
it and writes copies into a new output directory.

Usage (Linux/Mac):
    python make_burst_split.py --data_dir /path/to/DATASET --output /path/to/burst_split \
        --val_split 0.2 --gap_seconds 60 --seed 42

Usage (Windows, cmd.exe):
    python make_burst_split.py --data_dir C:\\path\\to\\DATASET --output C:\\path\\to\\burst_split --val_split 0.2 --gap_seconds 60 --seed 42
"""
import argparse
import os
import re
import shutil
import random
from datetime import datetime
from collections import defaultdict

FILENAME_PATTERN = re.compile(r"Screenshot (\d{4}-\d{2}-\d{2}) (\d{6})\.\w+$")


def parse_timestamp(filename):
    m = FILENAME_PATTERN.search(filename)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(date_str + time_str, "%Y-%m-%d%H%M%S")


def group_into_bursts(files_with_ts, gap_seconds):
    """files_with_ts: list of (datetime, filepath), assumed same class.
    Returns list of bursts, each a list of filepaths, ordered by time."""
    items = sorted(files_with_ts, key=lambda x: x[0])
    bursts = []
    current = [items[0]]
    for i in range(1, len(items)):
        gap = (items[i][0] - items[i - 1][0]).total_seconds()
        if gap > gap_seconds:
            bursts.append(current)
            current = []
        current.append(items[i])
    bursts.append(current)
    return [[fp for _, fp in burst] for burst in bursts]


def assign_bursts_to_split(bursts, val_split, seed, class_name):
    """Greedily assigns whole bursts to val until the target fraction is
    roughly reached, guaranteeing (if >=2 bursts exist) at least one burst
    on each side so the class is represented in both splits."""
    rng = random.Random(seed)
    order = list(range(len(bursts)))
    rng.shuffle(order)
    shuffled = [bursts[i] for i in order]

    total = sum(len(b) for b in bursts)
    target_val = total * val_split

    val_bursts, train_bursts = [], []
    val_count = 0
    for idx, burst in enumerate(shuffled):
        remaining_bursts = len(shuffled) - idx
        if val_count < target_val and remaining_bursts > 1:
            val_bursts.append(burst)
            val_count += len(burst)
        else:
            train_bursts.append(burst)

    # Guarantee representation on both sides if at all possible.
    if not val_bursts and len(train_bursts) >= 2:
        val_bursts.append(train_bursts.pop())
    if not train_bursts and len(val_bursts) >= 2:
        train_bursts.append(val_bursts.pop())

    warning = None
    if not val_bursts:
        warning = f"Class '{class_name}' has only 1 burst total -- cannot appear in both splits."

    return train_bursts, val_bursts, warning


def build_split(data_dir, output_dir, val_split, gap_seconds, seed, mode):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    report = {
        "total": 0, "train_total": 0, "val_total": 0,
        "per_class": {}, "warnings": [], "burst_counts": {},
    }

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files_with_ts = []
        unmatched = []
        for f in os.listdir(cls_dir):
            fp = os.path.join(cls_dir, f)
            if not os.path.isfile(fp):
                continue
            ts = parse_timestamp(f)
            if ts is None:
                unmatched.append(fp)
            else:
                files_with_ts.append((ts, fp))

        if unmatched:
            report["warnings"].append(
                f"Class '{cls}': {len(unmatched)} file(s) did not match the timestamp "
                f"pattern and were assigned individually to train (no grouping signal): "
                f"{[os.path.basename(u) for u in unmatched]}"
            )

        bursts = group_into_bursts(files_with_ts, gap_seconds) if files_with_ts else []
        report["burst_counts"][cls] = len(bursts)

        train_bursts, val_bursts, warning = assign_bursts_to_split(bursts, val_split, seed, cls)
        if warning:
            report["warnings"].append(warning)

        train_files = [fp for burst in train_bursts for fp in burst] + unmatched
        val_files = [fp for burst in val_bursts for fp in burst]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        if val_files:
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for fp in train_files:
            dst = os.path.join(train_dir, cls, os.path.basename(fp))
            _place_file(fp, dst, mode)
        for fp in val_files:
            dst = os.path.join(val_dir, cls, os.path.basename(fp))
            _place_file(fp, dst, mode)

        report["per_class"][cls] = {"train": len(train_files), "val": len(val_files)}
        report["train_total"] += len(train_files)
        report["val_total"] += len(val_files)
        report["total"] += len(train_files) + len(val_files)

    return report


def _place_file(src, dst, mode):
    if mode == "symlink":
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def verify_no_burst_crosses_boundary(data_dir, output_dir, gap_seconds):
    """Independent post-hoc check: re-derive bursts from the ORIGINAL data_dir
    and confirm no single burst has members in both output train/ and val/."""
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    problems = []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files_with_ts = []
        for f in os.listdir(cls_dir):
            fp = os.path.join(cls_dir, f)
            if not os.path.isfile(fp):
                continue
            ts = parse_timestamp(f)
            if ts is not None:
                files_with_ts.append((ts, f))
        if not files_with_ts:
            continue
        bursts = group_into_bursts([(ts, f) for ts, f in files_with_ts], gap_seconds)

        train_names = set(os.listdir(os.path.join(train_dir, cls))) if os.path.exists(os.path.join(train_dir, cls)) else set()
        val_names = set(os.listdir(os.path.join(val_dir, cls))) if os.path.exists(os.path.join(val_dir, cls)) else set()

        for burst in bursts:
            names = set(os.path.basename(f) for f in burst)
            in_train = names & train_names
            in_val = names & val_names
            if in_train and in_val:
                problems.append((cls, list(names)))

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a burst-aware train/val split for the SEM dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Original DATASET folder (read-only)")
    parser.add_argument("--output", type=str, required=True, help="Output folder to create (train/ and val/ subfolders)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--gap_seconds", type=int, default=60,
                         help="Gap (seconds) above which two consecutive same-class shots are treated as different bursts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink"])
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Path not found: {args.data_dir}")

    report = build_split(args.data_dir, args.output, args.val_split, args.gap_seconds, args.seed, args.mode)

    print(f"Total images: {report['total']}")
    print(f"Train images: {report['train_total']}")
    print(f"Val images:   {report['val_total']}")
    print("\nBursts detected per class:")
    for cls, n in report["burst_counts"].items():
        print(f"  {cls}: {n} bursts")
    print("\nPer-class split counts:")
    for cls, counts in report["per_class"].items():
        print(f"  {cls:<16} train={counts['train']:<4} val={counts['val']}")

    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  - {w}")

    print("\nVerifying no burst crosses the train/val boundary...")
    problems = verify_no_burst_crosses_boundary(args.data_dir, args.output, args.gap_seconds)
    if problems:
        print("PROBLEM: the following bursts were split across train/val:")
        for cls, names in problems:
            print(f"  {cls}: {names}")
    else:
        print("OK: no burst has members in both train/ and val/.")
