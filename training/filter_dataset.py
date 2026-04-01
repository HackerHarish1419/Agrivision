"""
AgriVisionAI — Dataset Filter
Extracts only Apple, Grape, and Tomato classes from the full PlantVillage dataset.
"""

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.config import RAW_DATA_DIR, FILTERED_DATA_DIR, TARGET_CROPS


def find_dataset_root(raw_dir: Path) -> Path:
    """
    Auto-detect the actual image root inside raw_dir.
    PlantVillage downloads can have nested structures like:
      raw/plantvillage/color/Apple___Apple_scab/...
      raw/PlantVillage/Apple___Apple_scab/...
      raw/Apple___Apple_scab/...
    This function walks until it finds class folders.
    """
    # Check if raw_dir itself contains class folders
    for child in raw_dir.iterdir():
        if child.is_dir() and "___" in child.name:
            return raw_dir

    # Recurse one or two levels
    for child in raw_dir.rglob("*"):
        if child.is_dir() and "___" in child.name:
            return child.parent

    raise FileNotFoundError(
        f"Could not find PlantVillage class folders (with '___' separator) "
        f"inside {raw_dir}. Please check your dataset structure."
    )


def filter_dataset():
    """Filter PlantVillage dataset to only target crops."""
    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("Please download the PlantVillage dataset and extract it there.")
        print("Download: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        sys.exit(1)

    dataset_root = find_dataset_root(RAW_DATA_DIR)
    print(f"Found dataset root: {dataset_root}")

    FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    filtered_count = 0
    total_images = 0

    for class_dir in class_dirs:
        # Check if this class belongs to one of our target crops
        crop_name = class_dir.name.split("___")[0]
        if crop_name not in TARGET_CROPS:
            continue

        dest = FILTERED_DATA_DIR / class_dir.name
        if dest.exists():
            print(f"  [SKIP] {class_dir.name} (already exists)")
            image_count = len(list(dest.glob("*")))
        else:
            shutil.copytree(class_dir, dest)
            image_count = len(list(dest.glob("*")))
            print(f"  [COPY] {class_dir.name} → {image_count} images")

        filtered_count += 1
        total_images += image_count

    print(f"\n✅ Filtered {filtered_count} classes, {total_images} total images")
    print(f"   Output: {FILTERED_DATA_DIR}")
    return filtered_count


if __name__ == "__main__":
    filter_dataset()
