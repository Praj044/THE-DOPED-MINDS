"""
Reconstructs the exact validation split that get_dataloaders() would have
produced for a given data_dir/seed/val_split, and copies those files into a
new folder (preserving class subfolders) so evaluate.py can be run against
a genuinely held-out set.

IMPORTANT: this recovers the val split used during training (images not
used for gradient updates), which is the best held-out set we can get
without retraining. It is NOT a fully independent test set, because these
images WERE used for checkpoint selection ("save best model on val_acc").
For a rigorous final number, hold out a third split before your next
training run instead (see README "Recommended fix" note printed below).

Usage:
    python make_val_split.py --data_dir /path/to/DATASET --output /path/to/held_out_val \
        --seed 42 --val_split 0.2
"""
import argparse
import os
import shutil
import torch

from data_loader import WM811KDataset


def build_val_split(data_dir, output_dir, seed=42, val_split=0.2, mode="copy"):
    base_dataset = WM811KDataset(data_dir, transform=None)
    n = len(base_dataset)
    val_size = int(n * val_split)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[train_size:]

    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for idx in val_indices:
        src_path = base_dataset.samples[idx]
        class_name = os.path.basename(os.path.dirname(src_path))
        dst_dir = os.path.join(output_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        if mode == "symlink":
            if not os.path.exists(dst_path):
                os.symlink(os.path.abspath(src_path), dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Wrote {copied} held-out images ({len(val_indices)} val indices, "
          f"seed={seed}, val_split={val_split}) to {output_dir}")
    print(f"Total dataset size was {n}; train_size={train_size}, val_size={val_size}")
    print(
        "\nNOTE: these images were NOT used for gradient updates, but WERE used "
        "for 'save best model on val_acc' during training -- so this is a decent "
        "held-out check, not a fully independent test set. For a rigorous final "
        "number before deployment, carve out a third split (e.g. 70/15/15 "
        "train/val/test) before your next training run and never touch the test "
        "folder until the very end."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct the held-out val split used during training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Original full dataset folder")
    parser.add_argument("--output", type=str, required=True, help="Where to write the held-out split")
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used in train_sem.py")
    parser.add_argument("--val_split", type=float, default=0.2, help="Must match the val_split used in training")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink"],
                         help="copy duplicates files; symlink saves disk space (Linux/Mac only)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Path not found: {args.data_dir}")

    build_val_split(args.data_dir, args.output, args.seed, args.val_split, args.mode)
