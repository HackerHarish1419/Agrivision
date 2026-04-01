"""
╔══════════════════════════════════════════════════════════════╗
║           AgriVisionAI — Google Colab Training Script        ║
║      Matrix Fusion 4.0 · Yenepoya AI Hackathon 2026         ║
║                                                              ║
║  HOW TO USE:                                                 ║
║  1. Open Google Colab (colab.research.google.com)            ║
║  2. Upload this file OR paste into cells                     ║
║  3. Set Runtime → GPU (T4)                                   ║
║  4. Upload your kaggle.json for dataset download             ║
║  5. Run all cells — takes ~25 min on T4                      ║
║  6. Download best_model.pth + class_names.json at the end    ║
╚══════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# CELL 1: Install Dependencies & Setup
# =============================================================================
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("torch")
install("torchvision")
install("scikit-learn")
install("matplotlib")
install("seaborn")
install("Pillow")

import os
import json
import random
import shutil
import time
from pathlib import Path
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms, datasets, models
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# CELL 2: Configuration — EDIT THESE AS NEEDED
# =============================================================================
class Config:
    # Paths
    DATA_ROOT = Path("/content/data")
    RAW_DIR = DATA_ROOT / "raw"
    FILTERED_DIR = DATA_ROOT / "filtered"
    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    TEST_DIR = DATA_ROOT / "test"
    OUTPUT_DIR = Path("/content/output")

    # Target crops
    TARGET_CROPS = ["Apple", "Grape", "Tomato"]

    # Model
    MODEL_ARCH = "efficientnet_b0"  # Options: efficientnet_b0, resnet50, resnet34
    IMAGE_SIZE = 224
    PRETRAINED = True

    # Training
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    SEED = 42

    # Phase 1: Feature extraction (frozen backbone)
    PHASE1_EPOCHS = 8
    PHASE1_LR = 1e-3

    # Phase 2: Fine-tuning (full network)
    PHASE2_EPOCHS = 12
    PHASE2_LR_BACKBONE = 1e-5
    PHASE2_LR_HEAD = 1e-4

    WEIGHT_DECAY = 1e-4

    # Data split
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    # Checkpoints to save
    CHECKPOINT_EPOCHS = [5, 10, 15, 20]

    # Augmentation
    COLOR_JITTER = (0.2, 0.2, 0.2, 0.1)
    ROTATION_DEGREES = 15

cfg = Config()

# Reproducibility
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# CELL 3: Download Dataset from Kaggle
# =============================================================================
def download_dataset():
    """Download PlantVillage dataset from Kaggle."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_dirs = [d for d in cfg.RAW_DIR.rglob("*") if d.is_dir() and "___" in d.name]
    if len(existing_dirs) > 10:
        print(f"✅ Dataset already exists ({len(existing_dirs)} class folders found)")
        return

    print("📥 Downloading PlantVillage dataset from Kaggle...")
    print("   If this fails, upload your kaggle.json first:")
    print("   from google.colab import files; files.upload()")

    # Setup Kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    # Try to find kaggle.json
    if not (kaggle_dir / "kaggle.json").exists():
        # Check if uploaded to /content
        if Path("/content/kaggle.json").exists():
            shutil.copy("/content/kaggle.json", kaggle_dir / "kaggle.json")
            os.chmod(kaggle_dir / "kaggle.json", 0o600)
        else:
            print("\n⚠️  kaggle.json not found!")
            print("Option 1: Upload kaggle.json to Colab and re-run this cell")
            print("Option 2: Manually download from Kaggle and upload to /content/data/raw/")
            return

    install("kaggle")

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "abdallahalidev/plantvillage-dataset",
        "-p", str(cfg.RAW_DIR),
        "--unzip"
    ], check=True)

    print("✅ Dataset downloaded successfully!")

download_dataset()

# =============================================================================
# CELL 4: Filter Dataset to Target Crops
# =============================================================================
def find_dataset_root(raw_dir: Path) -> Path:
    """Auto-detect the image root — handles nested folder structures."""
    for child in raw_dir.iterdir():
        if child.is_dir() and "___" in child.name:
            return raw_dir

    for child in raw_dir.rglob("*"):
        if child.is_dir() and "___" in child.name:
            return child.parent

    raise FileNotFoundError(
        f"Could not find PlantVillage class folders inside {raw_dir}. "
        "Make sure dataset is extracted correctly."
    )


def filter_dataset():
    """Keep only Apple, Grape, Tomato classes."""
    dataset_root = find_dataset_root(cfg.RAW_DIR)
    print(f"📂 Found dataset root: {dataset_root}")

    cfg.FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    filtered = 0
    total = 0

    for class_dir in class_dirs:
        crop = class_dir.name.split("___")[0]
        if crop not in cfg.TARGET_CROPS:
            continue

        dest = cfg.FILTERED_DIR / class_dir.name
        if not dest.exists():
            shutil.copytree(class_dir, dest)

        count = len(list(dest.glob("*")))
        print(f"  ✓ {class_dir.name}: {count} images")
        filtered += 1
        total += count

    print(f"\n✅ Filtered to {filtered} classes, {total} total images")

filter_dataset()

# =============================================================================
# CELL 5: Split Dataset
# =============================================================================
def split_dataset():
    """Stratified train/val/test split."""
    random.seed(cfg.SEED)

    for split_dir in [cfg.TRAIN_DIR, cfg.VAL_DIR, cfg.TEST_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    class_dirs = sorted([d for d in cfg.FILTERED_DIR.iterdir() if d.is_dir()])
    stats = {}

    for class_dir in class_dirs:
        name = class_dir.name
        images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG"}
        ])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * cfg.TRAIN_RATIO)
        n_val = int(n * cfg.VAL_RATIO)

        splits = {
            "train": (cfg.TRAIN_DIR, images[:n_train]),
            "val": (cfg.VAL_DIR, images[n_train:n_train + n_val]),
            "test": (cfg.TEST_DIR, images[n_train + n_val:]),
        }

        for split_name, (split_dir, split_imgs) in splits.items():
            dest = split_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        stats[name] = {"total": n, "train": n_train, "val": n_val, "test": n - n_train - n_val}

    # Print summary
    print("\n📊 Dataset Split Summary:")
    print(f"{'Class':<55} {'Train':>6} {'Val':>5} {'Test':>5} {'Total':>6}")
    print("─" * 80)
    for name, s in sorted(stats.items()):
        print(f"  {name:<53} {s['train']:>6} {s['val']:>5} {s['test']:>5} {s['total']:>6}")
    print("─" * 80)
    t_train = sum(s["train"] for s in stats.values())
    t_val = sum(s["val"] for s in stats.values())
    t_test = sum(s["test"] for s in stats.values())
    t_total = sum(s["total"] for s in stats.values())
    print(f"  {'TOTAL':<53} {t_train:>6} {t_val:>5} {t_test:>5} {t_total:>6}")

    return stats

split_stats = split_dataset()

# =============================================================================
# CELL 6: Data Transforms & Loaders
# =============================================================================
def get_transforms():
    """Get train/val transforms — configurable augmentations."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(cfg.ROTATION_DEGREES),
        transforms.ColorJitter(*cfg.COLOR_JITTER),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.1),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE + 32),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, val_transform


def get_dataloaders():
    """Create data loaders with class-weighted sampling for imbalance."""
    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(str(cfg.TRAIN_DIR), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(cfg.VAL_DIR), transform=val_tf)
    test_dataset = datasets.ImageFolder(str(cfg.TEST_DIR), transform=val_tf)

    # Class-weighted sampler for training (handles imbalance)
    class_counts = Counter(train_dataset.targets)
    num_samples = len(train_dataset)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        sampler=sampler, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True
    )

    # Save class names mapping
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    print(f"\n📦 DataLoaders ready:")
    print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"   Classes: {len(class_names)}")

    # Show class distribution
    print("\n📊 Training class distribution (with weighted sampling):")
    for cls_idx, cls_name in enumerate(class_names):
        count = class_counts.get(cls_idx, 0)
        bar = "█" * (count // 20)
        print(f"   {cls_name:<55} {count:>5} {bar}")

    return train_loader, val_loader, test_loader, class_names, class_to_idx

train_loader, val_loader, test_loader, CLASS_NAMES, class_to_idx = get_dataloaders()
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n✅ NUM_CLASSES dynamically set to: {NUM_CLASSES}")

# =============================================================================
# CELL 7: Build Model
# =============================================================================
def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained: bool = True):
    """Build model with configurable architecture."""
    arch = arch.lower()

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    elif arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
        head_params = list(model.fc.parameters())

    elif arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
        head_params = list(model.fc.parameters())

    else:
        raise ValueError(f"Unsupported architecture: {arch}. Use efficientnet_b0, efficientnet_b2, resnet50, resnet34")

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Model: {arch}")
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Output classes:   {num_classes}")

    return model, backbone_params, head_params


model, backbone_params, head_params = build_model(
    NUM_CLASSES, cfg.MODEL_ARCH, cfg.PRETRAINED
)

# =============================================================================
# CELL 8: Training Functions
# =============================================================================
def compute_class_weights(dataset_dir: Path, num_classes: int):
    """Compute inverse class weights for loss function."""
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    counts = []
    for d in class_dirs:
        n = len(list(d.glob("*")))
        counts.append(n)

    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0 for c in counts]
    return torch.FloatTensor(weights).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}", end="\r")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return epoch_loss, epoch_f1, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_f1, epoch_acc, epoch_precision, epoch_recall, all_preds, all_labels


# =============================================================================
# CELL 9: Training Loop — Phase 1 (Frozen Backbone) + Phase 2 (Fine-tune)
# =============================================================================
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class weights for imbalanced loss
class_weights = compute_class_weights(cfg.TRAIN_DIR, NUM_CLASSES)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# History
history = {
    "epoch": [], "phase": [],
    "train_loss": [], "train_f1": [], "train_acc": [],
    "val_loss": [], "val_f1": [], "val_acc": [],
    "val_precision": [], "val_recall": [], "lr": []
}

best_val_f1 = 0.0
total_epochs = cfg.PHASE1_EPOCHS + cfg.PHASE2_EPOCHS

print("=" * 70)
print("  PHASE 1: Feature Extraction (Backbone Frozen)")
print("=" * 70)

# Freeze backbone
for p in backbone_params:
    p.requires_grad = False

optimizer = optim.AdamW(head_params, lr=cfg.PHASE1_LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PHASE1_EPOCHS)

for epoch in range(1, cfg.PHASE1_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_f1, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, epoch, total_epochs
    )
    val_loss, val_f1, val_acc, val_prec, val_rec, _, _ = evaluate(
        model, val_loader, criterion
    )
    scheduler.step()

    elapsed = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"  Epoch [{epoch:>2}/{total_epochs}] "
        f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.2e} | {elapsed:.1f}s"
    )

    # Log
    history["epoch"].append(epoch)
    history["phase"].append(1)
    history["train_loss"].append(train_loss)
    history["train_f1"].append(train_f1)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)
    history["val_precision"].append(val_prec)
    history["val_recall"].append(val_rec)
    history["lr"].append(current_lr)

    # Save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "arch": cfg.MODEL_ARCH,
            "image_size": cfg.IMAGE_SIZE,
        }, cfg.OUTPUT_DIR / "best_model.pth")
        print(f"    ★ New best model! Val F1: {val_f1:.4f}")

    # Checkpoint saves
    if epoch in cfg.CHECKPOINT_EPOCHS:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "arch": cfg.MODEL_ARCH,
        }, cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pth")


print("\n" + "=" * 70)
print("  PHASE 2: Fine-Tuning (Full Network Unfrozen)")
print("=" * 70)

# Unfreeze backbone
for p in backbone_params:
    p.requires_grad = True

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": cfg.PHASE2_LR_BACKBONE},
    {"params": head_params, "lr": cfg.PHASE2_LR_HEAD},
], weight_decay=cfg.WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PHASE2_EPOCHS)

for epoch in range(cfg.PHASE1_EPOCHS + 1, total_epochs + 1):
    t0 = time.time()

    train_loss, train_f1, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, epoch, total_epochs
    )
    val_loss, val_f1, val_acc, val_prec, val_rec, _, _ = evaluate(
        model, val_loader, criterion
    )
    scheduler.step()

    elapsed = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"  Epoch [{epoch:>2}/{total_epochs}] "
        f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.2e} | {elapsed:.1f}s"
    )

    # Log
    history["epoch"].append(epoch)
    history["phase"].append(2)
    history["train_loss"].append(train_loss)
    history["train_f1"].append(train_f1)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)
    history["val_precision"].append(val_prec)
    history["val_recall"].append(val_rec)
    history["lr"].append(current_lr)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "arch": cfg.MODEL_ARCH,
            "image_size": cfg.IMAGE_SIZE,
        }, cfg.OUTPUT_DIR / "best_model.pth")
        print(f"    ★ New best model! Val F1: {val_f1:.4f}")

    if epoch in cfg.CHECKPOINT_EPOCHS:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "arch": cfg.MODEL_ARCH,
        }, cfg.OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pth")

print(f"\n🏆 Training complete! Best Val F1: {best_val_f1:.4f}")

# =============================================================================
# CELL 10: Evaluation on Test Set
# =============================================================================
print("\n" + "=" * 70)
print("  FINAL EVALUATION ON TEST SET")
print("=" * 70)

# Load best model
checkpoint = torch.load(cfg.OUTPUT_DIR / "best_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded best model from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")

test_loss, test_f1, test_acc, test_prec, test_rec, test_preds, test_labels = evaluate(
    model, test_loader, criterion
)

print(f"\n📊 Test Results:")
print(f"   Macro F1:    {test_f1:.4f}")
print(f"   Accuracy:    {test_acc:.4f}")
print(f"   Precision:   {test_prec:.4f}")
print(f"   Recall:      {test_rec:.4f}")

# Full classification report
print(f"\n📋 Per-Class Classification Report:")
report = classification_report(
    test_labels, test_preds,
    target_names=CLASS_NAMES,
    digits=4
)
print(report)

# Save report
with open(cfg.OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(f"AgriVisionAI — Test Set Classification Report\n")
    f.write(f"Model: {cfg.MODEL_ARCH}\n")
    f.write(f"Best epoch: {checkpoint['epoch']}\n")
    f.write(f"Macro F1: {test_f1:.4f}\n")
    f.write(f"Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# =============================================================================
# CELL 11: Confusion Matrix
# =============================================================================
cm = confusion_matrix(test_labels, test_preds)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=[c.split("___")[-1] for c in CLASS_NAMES],
    yticklabels=[c.split("___")[-1] for c in CLASS_NAMES],
    ax=ax
)
ax.set_title(f"AgriVisionAI — Confusion Matrix (Test Set)\nMacro F1: {test_f1:.4f}", fontsize=14)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(cfg.OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Confusion matrix saved")

# =============================================================================
# CELL 12: Training Curves
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history["epoch"], history["train_loss"], "b-o", markersize=3, label="Train")
axes[0, 0].plot(history["epoch"], history["val_loss"], "r-o", markersize=3, label="Val")
axes[0, 0].axvline(x=cfg.PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Phase 2 Start")
axes[0, 0].set_title("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# F1
axes[0, 1].plot(history["epoch"], history["train_f1"], "b-o", markersize=3, label="Train")
axes[0, 1].plot(history["epoch"], history["val_f1"], "r-o", markersize=3, label="Val")
axes[0, 1].axvline(x=cfg.PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Phase 2 Start")
axes[0, 1].set_title("Macro F1 Score")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision & Recall
axes[1, 0].plot(history["epoch"], history["val_precision"], "g-o", markersize=3, label="Precision")
axes[1, 0].plot(history["epoch"], history["val_recall"], "m-o", markersize=3, label="Recall")
axes[1, 0].axvline(x=cfg.PHASE1_EPOCHS, color="gray", linestyle="--", alpha=0.5)
axes[1, 0].set_title("Validation Precision & Recall")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history["epoch"], history["lr"], "k-o", markersize=3)
axes[1, 1].set_title("Learning Rate")
axes[1, 1].set_yscale("log")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle("AgriVisionAI Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(cfg.OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Training curves saved")

# =============================================================================
# CELL 13: Checkpoint Comparison Table
# =============================================================================
print("\n📊 Checkpoint Comparison:")
print(f"{'Epoch':>6} {'Val F1':>8} {'Val Acc':>8} {'Val Prec':>9} {'Val Rec':>8}")
print("─" * 42)
for i, ep in enumerate(history["epoch"]):
    if ep in cfg.CHECKPOINT_EPOCHS or ep == checkpoint["epoch"]:
        marker = " ★" if ep == checkpoint["epoch"] else ""
        print(
            f"  {ep:>4} {history['val_f1'][i]:>8.4f} {history['val_acc'][i]:>8.4f} "
            f"{history['val_precision'][i]:>9.4f} {history['val_recall'][i]:>8.4f}{marker}"
        )

# =============================================================================
# CELL 14: Misclassification Analysis
# =============================================================================
print("\n🔍 Top Misclassification Pairs:")
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)

# Find top confused pairs
top_k = 10
flat_indices = np.argsort(cm_off_diag.ravel())[::-1][:top_k]
for idx in flat_indices:
    true_cls = idx // NUM_CLASSES
    pred_cls = idx % NUM_CLASSES
    count = cm_off_diag[true_cls, pred_cls]
    if count == 0:
        break
    true_name = CLASS_NAMES[true_cls].split("___")[-1]
    pred_name = CLASS_NAMES[pred_cls].split("___")[-1]
    print(f"   {true_name:<40} → {pred_name:<40} ({count} times)")

# =============================================================================
# CELL 15: Save class_names.json & Export for Local Deployment
# =============================================================================
class_names_map = {str(i): name for i, name in enumerate(CLASS_NAMES)}
with open(cfg.OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names_map, f, indent=2)
print(f"\n✅ class_names.json saved with {len(class_names_map)} classes")

# Save training history
with open(cfg.OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("✅ training_history.json saved")

# =============================================================================
# CELL 16: Download Files for Local Deployment
# =============================================================================
print("\n" + "=" * 70)
print("  📦 FILES TO DOWNLOAD FOR LOCAL DEPLOYMENT")
print("=" * 70)
print(f"\n  1. best_model.pth         → Place in: models/best_model.pth")
print(f"  2. class_names.json       → Place in: models/class_names.json")
print(f"  3. confusion_matrix.png   → For documentation")
print(f"  4. training_curves.png    → For documentation")
print(f"  5. classification_report  → For judging")
print(f"\n  Output directory: {cfg.OUTPUT_DIR}")

# Auto-download in Colab
try:
    from google.colab import files as colab_files
    print("\n📥 Downloading files...")
    for f in cfg.OUTPUT_DIR.iterdir():
        print(f"   Downloading: {f.name}")
        colab_files.download(str(f))
except ImportError:
    print("\n   (Not running in Colab — manually copy files from output directory)")

print("\n🎉 All done! Copy best_model.pth and class_names.json to your project's models/ folder.")
