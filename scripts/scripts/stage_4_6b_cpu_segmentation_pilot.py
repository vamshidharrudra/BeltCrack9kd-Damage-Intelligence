
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STAGE 4.6B - CPU FEASIBILITY PILOT
# ============================================================

ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)
DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

MANIFEST = ROOT / "canonical_dataset" / "canonical_manifest.csv"
MASK_ROOT = ROOT / "canonical_dataset" / "masks"

OUT = ROOT / "audit_outputs" / "stage_4_6b_cpu_pilot"
MODELS = ROOT / "models" / "segmentation"

OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
SIZE = 256
BATCH = 2
TRAIN_LIMIT = 400
VAL_LIMIT = 100
EPOCHS = 2
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Limit CPU thread pressure during this pilot.
torch.set_num_threads(4)
DEVICE = torch.device("cpu")


def frame_number(filename):
    stem = Path(str(filename)).stem
    return int(stem) if stem.isdigit() else None


def resolve_image(row):
    source = str(row["source_image"]).strip()
    p = Path(source)

    if p.is_file():
        return p

    p = DATASET_ROOT / source
    if p.is_file():
        return p

    return (
        DATASET_ROOT / "BeltCrack9kd1" / "images"
        / str(row["split"])
        / str(row["sequence"]).split("/")[-1]
        / str(row["original_image_filename"])
    )


def resolve_mask(row):
    frame = frame_number(row["canonical_image_filename"])
    if frame is None:
        return None

    return (
        MASK_ROOT / str(row["split"])
        / str(row["sequence"]).split("/")[-1]
        / f"{frame}.png"
    )


def fit_square(image, is_mask=False):
    resampling = (
        Image.Resampling.NEAREST if is_mask
        else Image.Resampling.BILINEAR
    )

    w, h = image.size
    scale = min(SIZE / w, SIZE / h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    image = image.resize((nw, nh), resampling)
    mode = "L" if is_mask else "RGB"
    fill = 0 if is_mask else (0, 0, 0)
    canvas = Image.new(mode, (SIZE, SIZE), fill)
    canvas.paste(image, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas


class CrackDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["_image"]).convert("RGB")
        mask = Image.open(row["_mask"]).convert("L")

        image = fit_square(image)
        mask = fit_square(mask, is_mask=True)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()

        y = np.asarray(mask, dtype=np.float32) / 255.0
        y = torch.from_numpy((y >= 0.5).astype(np.float32)).unsqueeze(0)

        return x, y


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


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (p * target).sum(dims)
    dice = (2 * intersection + 1e-7) / (
        p.sum(dims) + target.sum(dims) + 1e-7
    )
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def run_epoch(model, loader, optimizer=None, name=""):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    tp = fp = fn = 0.0
    seen = 0

    for batch_idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        pred = (torch.sigmoid(logits) >= 0.5).float()
        target = (y >= 0.5).float()

        tp += (pred * target).sum().item()
        fp += (pred * (1 - target)).sum().item()
        fn += ((1 - pred) * target).sum().item()
        total_loss += loss.item() * len(x)
        seen += len(x)

        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(
                f"{name}: batch {batch_idx}/{len(loader)}",
                flush=True
            )

    eps = 1e-7
    return {
        "loss": total_loss / max(seen, 1),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


print("=" * 72)
print("STAGE 4.6B - CPU SEGMENTATION FEASIBILITY PILOT")
print("=" * 72)
print("Device:", DEVICE)
print("Image size:", SIZE)
print("Training subset:", TRAIN_LIMIT)
print("Validation subset:", VAL_LIMIT)
print("Epochs:", EPOCHS)

df = pd.read_csv(MANIFEST)

required = {
    "split", "component", "sequence",
    "canonical_image_filename", "original_image_filename",
    "source_image"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Manifest missing columns: {sorted(missing)}")

# Check leakage constraints against the complete canonical manifest.
if (df.groupby("component")["split"].nunique() > 1).any():
    raise RuntimeError("Component leakage detected.")
if (df.groupby("sequence")["split"].nunique() > 1).any():
    raise RuntimeError("Sequence leakage detected.")

# Select only TRAIN and VAL. Do not resolve or load test files.
parts = []
for split, limit in [("train", TRAIN_LIMIT), ("val", VAL_LIMIT)]:
    part = df[df["split"] == split].sample(
        n=min(limit, int((df["split"] == split).sum())),
        random_state=SEED
    ).copy()

    part["_image"] = part.apply(resolve_image, axis=1)
    part["_mask"] = part.apply(resolve_mask, axis=1)

    missing_files = [
        str(p) for p in part["_image"]
        if p is None or not Path(p).is_file()
    ]
    missing_masks = [
        str(p) for p in part["_mask"]
        if p is None or not Path(p).is_file()
    ]

    if missing_files or missing_masks:
        raise RuntimeError(
            f"{split}: missing images={len(missing_files)}, "
            f"missing masks={len(missing_masks)}"
        )

    parts.append(part)
    print(f"{split.upper()} pilot records: {len(part)}")

train_df, val_df = parts

train_loader = DataLoader(
    CrackDataset(train_df, augment=True),
    batch_size=BATCH, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    CrackDataset(val_df, augment=False),
    batch_size=BATCH, shuffle=False, num_workers=0
)

model = SmallUNet().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)

best_dice = -1.0
best_epoch = 0
history = []
checkpoint_path = MODELS / "beltcrack9kd_cpu_pilot_best.pt"

for epoch in range(1, EPOCHS + 1):
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
    history.append(record)

    print("Train:", json.dumps(train_metrics), flush=True)
    print("Val:  ", json.dumps(val_metrics), flush=True)

    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]
        best_epoch = epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": "SmallUNet",
            "image_size": SIZE,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_validation_dice": best_dice,
        }, checkpoint_path)
        print("Saved best validation checkpoint:", checkpoint_path, flush=True)

history_path = OUT / "stage_4_6b_pilot_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

summary = {
    "stage": "4.6B",
    "status": "PILOT_COMPLETE",
    "purpose": "CPU feasibility; not final model evaluation",
    "device": str(DEVICE),
    "image_size": SIZE,
    "batch_size": BATCH,
    "train_records": len(train_df),
    "validation_records": len(val_df),
    "test_used": False,
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "best_validation_dice": best_dice,
    "parameters": params,
    "checkpoint": str(checkpoint_path),
    "history_csv": str(history_path),
}
with open(OUT / "stage_4_6b_pilot_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 72)
print("STAGE 4.6B COMPLETE")
print("Best validation Dice:", round(best_dice, 6))
print("Checkpoint:", checkpoint_path)
print("History:", history_path)
print("Summary:", OUT / "stage_4_6b_pilot_summary.json")
print("TEST SET WAS NOT USED.")
print("=" * 72)
