"""
AgriVisionAI — Dataset Splitter
Splits filtered dataset into train/val/test with stratified sampling.
"""

import random
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.config import (
    FILTERED_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
)


def split_dataset():
    """Split filtered dataset into train/val/test with stratified sampling."""
    random.seed(SEED)

    if not FILTERED_DATA_DIR.exists():
        print(f"ERROR: Filtered data not found at {FILTERED_DATA_DIR}")
        print("Run filter_dataset.py first.")
        sys.exit(1)

    class_dirs = sorted([d for d in FILTERED_DATA_DIR.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"ERROR: No class directories found in {FILTERED_DATA_DIR}")
        sys.exit(1)

    # Validate ratios
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}"

    # Clean existing splits
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    stats = {}

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted(list(class_dir.glob("*")))
        images = [img for img in images if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Rest goes to test
        n_test = n - n_train - n_val

        splits = {
            "train": (TRAIN_DIR, images[:n_train]),
            "val": (VAL_DIR, images[n_train:n_train + n_val]),
            "test": (TEST_DIR, images[n_train + n_val:]),
        }

        for split_name, (split_dir, split_images) in splits.items():
            dest = split_dir / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest / img.name)

        stats[class_name] = {"total": n, "train": n_train, "val": n_val, "test": n_test}
        print(f"  {class_name}: {n_train} train / {n_val} val / {n_test} test (total: {n})")

    # Summary
    total = sum(s["total"] for s in stats.values())
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total_test = sum(s["test"] for s in stats.values())

    print(f"\n✅ Split complete: {total} images → {total_train} train / {total_val} val / {total_test} test")
    print(f"   Directories: {TRAIN_DIR}, {VAL_DIR}, {TEST_DIR}")

    return stats


if __name__ == "__main__":
    split_dataset()
