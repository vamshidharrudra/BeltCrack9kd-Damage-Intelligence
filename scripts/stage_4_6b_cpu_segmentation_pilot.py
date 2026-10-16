
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STAGE 4.6B - CPU SEGMENTATION PILOT (CORRECTED)
# ============================================================

ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)
DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

MANIFEST_PATH = ROOT / "canonical_dataset" / "canonical_manifest.csv"
MASK_ROOT = ROOT / "canonical_dataset" / "masks"
OUTPUT_ROOT = ROOT / "audit_outputs" / "stage_4_6b_cpu_pilot"
MODEL_ROOT = ROOT / "models" / "segmentation"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 2
TRAIN_LIMIT = 400
VAL_LIMIT = 100
EPOCHS = 2
LEARNING_RATE = 0.001
CPU_THREADS = 4

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(CPU_THREADS)


# ============================================================
# PATH HELPERS
# ============================================================

def extract_frame_number(filename):
    stem = Path(str(filename)).stem

    if stem.isdigit():
        return int(stem)

    match = re.search(r"(\d+)$", stem)

    return int(match.group(1)) if match else None


def resolve_image(row):
    source = str(row["source_image"]).strip()

    if source and source.lower() != "nan":
        path = Path(source)

        if path.is_file():
            return path

        path = DATASET_ROOT / source

        if path.is_file():
            return path

    split = str(row["split"]).strip()
    sequence_folder = (
        str(row["sequence"]).replace("\\", "/").split("/")[-1]
    )
    filename = str(row["original_image_filename"])

    path = (
        DATASET_ROOT / "BeltCrack9kd1" / "images"
        / split / sequence_folder / filename
    )

    return path if path.is_file() else None


def resolve_mask(row):
    split = str(row["split"]).strip()
    sequence_folder = (
        str(row["sequence"]).replace("\\", "/").split("/")[-1]
    )

    # Stage 4.5 writes masks using the numeric frame stem.
    # Try the canonical name first, then the original image name.
    filenames = [
        row["canonical_image_filename"],
        row["original_image_filename"],
    ]

    frame = None

    for filename in filenames:
        frame = extract_frame_number(filename)
        if frame is not None:
            break

    if frame is None:
        return None

    path = MASK_ROOT / split / sequence_folder / f"{frame}.png"

    return path if path.is_file() else None


def resize_and_pad(image, size, is_mask=False):
    width, height = image.size
    scale = min(size / width, size / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resampling = (
        Image.Resampling.NEAREST
        if is_mask else Image.Resampling.BILINEAR
    )

    image = image.resize((new_width, new_height), resampling)

    mode = "L" if is_mask else "RGB"
    fill = 0 if is_mask else (0, 0, 0)
    canvas = Image.new(mode, (size, size), fill)

    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(image, (left, top))

    return canvas


# ============================================================
# DATASET
# ============================================================

class CrackDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["_image"]) as img:
            image = img.convert("RGB")

        with Image.open(row["_mask"]) as img:
            mask = img.convert("L")

        if image.size != mask.size:
            raise RuntimeError(
                f"Image/mask dimension mismatch: "
                f"{row['_image']} vs {row['_mask']}"
            )

        image = resize_and_pad(image, IMAGE_SIZE)
        mask = resize_and_pad(mask, IMAGE_SIZE, is_mask=True)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()

        y = np.asarray(mask, dtype=np.uint8)
        y = (y > 0).astype(np.float32)
        y = torch.from_numpy(y.copy()).unsqueeze(0)

        return x, y


# ============================================================
# SMALL U-NET
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.e1 = ConvBlock(3, 16)
        self.e2 = ConvBlock(16, 32)
        self.e3 = ConvBlock(32, 64)
        self.e4 = ConvBlock(64, 128)
        self.mid = ConvBlock(128, 256)

        self.u4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d4 = ConvBlock(256, 128)
        self.u3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d3 = ConvBlock(128, 64)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = ConvBlock(64, 32)
        self.u1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))

        x = self.d4(torch.cat([self.u4(z), d], dim=1))
        x = self.d3(torch.cat([self.u3(x), c], dim=1))
        x = self.d2(torch.cat([self.u2(x), b], dim=1))
        x = self.d1(torch.cat([self.u1(x), a], dim=1))

        return self.out(x)


# ============================================================
# LOSS AND EVALUATION
# ============================================================

def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)

    p = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (p * target).sum(dims)
    denominator = p.sum(dims) + target.sum(dims)

    dice = (2 * intersection + 1e-7) / (denominator + 1e-7)

    return 0.5 * bce + 0.5 * (1 - dice.mean())


def run_epoch(model, loader, optimizer=None, label=""):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    tp = fp = fn = 0.0
    samples = 0

    for batch_index, (images, masks) in enumerate(loader, 1):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = combined_loss(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        prediction = (torch.sigmoid(logits) >= 0.5).float()
        target = (masks >= 0.5).float()

        tp += (prediction * target).sum().item()
        fp += (prediction * (1 - target)).sum().item()
        fn += ((1 - prediction) * target).sum().item()

        n = len(images)
        samples += n
        total_loss += loss.item() * n

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"{label}: batch {batch_index}/{len(loader)}",
                flush=True
            )

    eps = 1e-7

    return {
        "loss": total_loss / max(samples, 1),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


# ============================================================
# LOAD AND VERIFY MANIFEST
# ============================================================

print("=" * 72)
print("STAGE 4.6B - CORRECTED CPU SEGMENTATION PILOT")
print("=" * 72)
print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("CPU threads:", CPU_THREADS)

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")

if not MASK_ROOT.is_dir():
    raise FileNotFoundError(f"Mask root missing: {MASK_ROOT}")

manifest = pd.read_csv(MANIFEST_PATH)

required = [
    "split", "component", "sequence",
    "canonical_image_filename",
    "original_image_filename",
    "source_image",
]

missing = [col for col in required if col not in manifest.columns]

if missing:
    raise ValueError(f"Manifest schema mismatch: {missing}")

print("\nManifest records:", len(manifest))

component_conflicts = (
    manifest.groupby("component")["split"].nunique() > 1
).sum()

sequence_conflicts = (
    manifest.groupby("sequence")["split"].nunique() > 1
).sum()

print("Component split conflicts:", int(component_conflicts))
print("Sequence split conflicts:", int(sequence_conflicts))

if component_conflicts or sequence_conflicts:
    raise RuntimeError("Leakage check failed.")


# ============================================================
# SELECT TRAIN AND VALIDATION SUBSETS
# ============================================================

parts = []

for split, limit in [("train", TRAIN_LIMIT), ("val", VAL_LIMIT)]:
    source = manifest[manifest["split"] == split].copy()

    if len(source) < limit:
        raise RuntimeError(
            f"Not enough {split} records: {len(source)}"
        )

    part = source.sample(n=limit, random_state=SEED).copy()

    # IMPORTANT: no pandas integer-column assignment.
    # Compute frame numbers as a normal Python list.
    frame_values = []

    for _, row in part.iterrows():
        frame = extract_frame_number(
            row["canonical_image_filename"]
        )

        if frame is None:
            frame = extract_frame_number(
                row["original_image_filename"]
            )

        frame_values.append(frame)

    part["_frame"] = pd.Series(
        frame_values,
        index=part.index,
        dtype="object",
    )

    part["_image"] = [
        resolve_image(row)
        for _, row in part.iterrows()
    ]

    part["_mask"] = [
        resolve_mask(row)
        for _, row in part.iterrows()
    ]

    missing_image_rows = part[
        part["_image"].apply(lambda p: p is None)
    ]

    missing_mask_rows = part[
        part["_mask"].apply(lambda p: p is None)
    ]

    print(f"\n{split.upper()} records selected: {len(part)}")
    print("Unresolved frame numbers:", sum(v is None for v in frame_values))
    print("Missing images:", len(missing_image_rows))
    print("Missing masks:", len(missing_mask_rows))

    if len(missing_image_rows):
        print(
            "Image examples:\n",
            missing_image_rows[
                ["sequence", "original_image_filename", "source_image"]
            ].head(5).to_string(index=False),
        )

    if len(missing_mask_rows):
        print(
            "Mask examples:\n",
            missing_mask_rows[
                ["sequence", "canonical_image_filename", "_frame"]
            ].head(10).to_string(index=False),
        )

    if any(v is None for v in frame_values):
        raise RuntimeError(f"Could not extract frame numbers in {split}.")

    if len(missing_image_rows) or len(missing_mask_rows):
        raise RuntimeError(
            f"{split} path resolution failed. "
            "Training was not started."
        )

    # Validate actual files and paired dimensions.
    for _, row in part.head(5).iterrows():
        with Image.open(row["_image"]) as im:
            image_size = im.size

        with Image.open(row["_mask"]) as mk:
            mask_size = mk.size
            mk.load()
            foreground = int(np.count_nonzero(np.asarray(mk)))

        if image_size != mask_size:
            raise RuntimeError(
                f"Dimension mismatch: {row['_image']} / {row['_mask']}"
            )

        if foreground == 0:
            raise RuntimeError(f"Empty mask found: {row['_mask']}")

    print(f"{split.upper()} sample pair checks: PASS")
    parts.append(part)

train_df, val_df = parts

for name, frame in [("train", train_df), ("val", val_df)]:
    duplicates = frame.duplicated(["sequence", "_frame"]).sum()
    print(f"{name.upper()} duplicate sequence/frame keys:", int(duplicates))

    if duplicates:
        raise RuntimeError(f"Duplicate frame keys in {name} pilot.")

if set(train_df["sequence"]) & set(val_df["sequence"]):
    raise RuntimeError("Sequence leakage between pilot train and validation.")


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    CrackDataset(train_df, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    CrackDataset(val_df, augment=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# TRAIN
# ============================================================

model = SmallUNet().to(DEVICE)

parameter_count = sum(p.numel() for p in model.parameters())

print("\nTrainable parameters:", f"{parameter_count:,}")
print("Train records:", len(train_df))
print("Validation records:", len(val_df))
print("TEST SET USED: NO")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

best_dice = -1.0
best_epoch = 0
history = []

checkpoint_path = MODEL_ROOT / "beltcrack9kd_cpu_pilot_best.pt"
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

    train_metrics = run_epoch(
        model, train_loader, optimizer, "TRAIN"
    )

    val_metrics = run_epoch(
        model, val_loader, None, "VAL"
    )

    record = {"epoch": epoch}
    record.update({f"train_{k}": v for k, v in train_metrics.items()})
    record.update({f"val_{k}": v for k, v in val_metrics.items()})
    record["epoch_seconds"] = time.time() - epoch_start
    history.append(record)

    print("Train:", json.dumps(train_metrics), flush=True)
    print("Validation:", json.dumps(val_metrics), flush=True)
    print("Epoch seconds:", round(record["epoch_seconds"], 1))

    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": "SmallUNet",
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_validation_dice": best_dice,
            },
            checkpoint_path,
        )

        print("Saved best validation checkpoint:", checkpoint_path)

history_path = OUTPUT_ROOT / "stage_4_6b_pilot_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

summary = {
    "stage": "4.6B",
    "status": "PILOT_COMPLETE",
    "device": str(DEVICE),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "train_records": len(train_df),
    "validation_records": len(val_df),
    "test_used": False,
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "best_validation_dice": best_dice,
    "parameters": parameter_count,
    "total_seconds": time.time() - start_time,
    "checkpoint": str(checkpoint_path),
    "history_csv": str(history_path),
}

summary_path = OUTPUT_ROOT / "stage_4_6b_pilot_summary.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print("\n" + "=" * 72)
print("STAGE 4.6B COMPLETE")
print("Best validation Dice:", round(best_dice, 6))
print("Checkpoint:", checkpoint_path)
print("History:", history_path)
print("Summary:", summary_path)
print("TEST SET WAS NOT USED.")
print("=" * 72)
