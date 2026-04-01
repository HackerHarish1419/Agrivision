"""
AgriVisionAI — Training Configuration
All hyperparameters and paths are configurable here.
"""

import os
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FILTERED_DATA_DIR = DATA_DIR / "filtered"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"

# === Target Crops (filter from PlantVillage) ===
TARGET_CROPS = ["Apple", "Grape", "Tomato"]

# === Model ===
MODEL_NAME = os.getenv("MODEL_ARCH", "efficientnet_b0")
NUM_CLASSES = 18  # Will be dynamically set from dataset
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
PRETRAINED = True

# === Training Hyperparameters ===
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
EPOCHS = int(os.getenv("EPOCHS", 20))

# Phase 1: Feature extraction (frozen backbone)
PHASE1_EPOCHS = int(os.getenv("PHASE1_EPOCHS", 8))
PHASE1_LR = float(os.getenv("PHASE1_LR", 1e-3))

# Phase 2: Fine-tuning (full network)
PHASE2_LR_BACKBONE = float(os.getenv("PHASE2_LR_BACKBONE", 1e-5))
PHASE2_LR_HEAD = float(os.getenv("PHASE2_LR_HEAD", 1e-4))

WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))

# === Data Split ===
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", 0.80))
VAL_RATIO = float(os.getenv("VAL_RATIO", 0.10))
TEST_RATIO = float(os.getenv("TEST_RATIO", 0.10))

# === Checkpoints ===
CHECKPOINT_SAVE_EPOCHS = [5, 10, 15, 20]  # Save at these epochs
BEST_MODEL_FILENAME = "best_model.pth"
CLASS_NAMES_FILENAME = "class_names.json"

# === Augmentation ===
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
RANDOM_ROTATION_DEGREES = 15

# === Reproducibility ===
SEED = int(os.getenv("SEED", 42))
