
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# STAGE 4.6D - VALIDATION THRESHOLD DIAGNOSTIC
# No training. Validation only. Test set remains untouched.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")
MANIFEST = ROOT / "canonical_dataset" / "canonical_manifest.csv"
MASK_ROOT = ROOT / "canonical_dataset" / "masks"
CHECKPOINT = ROOT / "models" / "segmentation" / "beltcrack9kd_cpu_pilot_best.pt"
OUT = ROOT / "audit_outputs" / "stage_4_6d_threshold_diagnostic"

SIZE = 256
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
              0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

print("=" * 68)
print("STAGE 4.6D - THRESHOLD DIAGNOSTIC")
print("=" * 68)
print("Validation only: YES")
print("Training: NO")
print("Test set used: NO")

# ------------------------------------------------------------
# 1. Reconstruct the checkpoint architecture
# ------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
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
        self.u4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d4 = ConvBlock(256, 128)
        self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d3 = ConvBlock(128, 64)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d2 = ConvBlock(64, 32)
        self.u1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.d1 = ConvBlock(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        mid = self.mid(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(mid), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
if ckpt.get("model_type") != "SmallUNet" or int(ckpt.get("image_size", -1)) != SIZE:
    raise ValueError("Checkpoint architecture or image size does not match.")

model = SmallUNet()
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

print("Checkpoint loaded; best epoch:", ckpt.get("best_epoch"))

# ------------------------------------------------------------
# 2. Read validation manifest and resolve paths
# ------------------------------------------------------------

df = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
required = [
    "split", "sequence", "canonical_image_filename",
    "original_image_filename", "canonical_image_path", "source_image"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Required manifest columns missing: {missing}")

val = df[df["split"].str.strip().str.lower() == "val"].copy()
if val.empty:
    raise ValueError("No validation records found.")

def frame_number(filename):
    if not filename:
        return None
    stem = Path(str(filename).replace("\\", "/")).stem
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None

def existing_path(value):
    if not value:
        return None
    p = Path(str(value).strip().strip('"'))
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([ROOT / p, DATASET_ROOT / p])
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except (OSError, ValueError):
            pass
    return None

def resolve_image(row):
    for col in ["canonical_image_path", "source_image"]:
        p = existing_path(row[col])
        if p is not None:
            return p
    for col in ["canonical_image_filename", "original_image_filename"]:
        value = row[col]
        if value:
            p = Path(str(value).replace("\\", "/"))
            for candidate in [DATASET_ROOT / p, ROOT / p]:
                if candidate.is_file():
                    return candidate.resolve()
    return None

def resolve_mask(row):
    frame = frame_number(row["canonical_image_filename"])
    if frame is None:
        frame = frame_number(row["original_image_filename"])
    if frame is None:
        return None
    folder = Path(str(row["sequence"]).replace("\\", "/")).name
    candidate = MASK_ROOT / "val" / folder / f"{frame}.png"
    return candidate.resolve() if candidate.is_file() else None

records = []
missing_images, missing_masks, missing_frames = [], [], []

for idx, (_, row) in enumerate(val.iterrows()):
    image_path = resolve_image(row)
    mask_path = resolve_mask(row)
    frame = frame_number(row["canonical_image_filename"])
    if frame is None:
        frame = frame_number(row["original_image_filename"])

    if image_path is None:
        missing_images.append(idx)
    if mask_path is None:
        missing_masks.append(idx)
    if frame is None:
        missing_frames.append(idx)

    if image_path and mask_path and frame is not None:
        records.append({
            "sequence": row["sequence"],
            "frame": frame,
            "image_path": image_path,
            "mask_path": mask_path,
        })

print("Validation records:", len(val))
print("Resolved pairs:", len(records))
print("Missing images:", len(missing_images))
print("Missing masks:", len(missing_masks))
print("Unresolved frames:", len(missing_frames))

if missing_images or missing_masks or missing_frames or len(records) != len(val):
    raise RuntimeError("Path resolution failed. Do not continue until corrected.")

# ------------------------------------------------------------
# 3. Accumulators for per-image and global pixel metrics
# ------------------------------------------------------------

per_image = {t: [] for t in THRESHOLDS}
global_counts = {
    t: {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "truth": 0}
    for t in THRESHOLDS
}

start_time = time.time()

# ------------------------------------------------------------
# 4. Inference once per image; evaluate all thresholds
# ------------------------------------------------------------

for i, rec in enumerate(records):
    with Image.open(rec["image_path"]) as im:
        im = im.convert("RGB").resize(
            (SIZE, SIZE), Image.Resampling.BILINEAR
        )
        arr = np.asarray(im, dtype=np.float32) / 255.0

    with Image.open(rec["mask_path"]) as im:
        im = im.convert("L").resize(
            (SIZE, SIZE), Image.Resampling.NEAREST
        )
        truth = np.asarray(im, dtype=np.uint8) > 0

    tensor = torch.from_numpy(
        np.transpose(arr, (2, 0, 1)).copy()
    ).unsqueeze(0).float()

    with torch.inference_mode():
        prob = torch.sigmoid(model(tensor))[0, 0].numpy()

    for threshold in THRESHOLDS:
        pred = prob >= threshold
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())

        dice_den = 2 * tp + fp + fn
        union = tp + fp + fn

        dice = 2 * tp / dice_den if dice_den else 1.0
        iou = tp / union if union else 1.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        per_image[threshold].append({
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "predicted_coverage": float(pred.mean()),
            "ground_truth_coverage": float(truth.mean()),
        })

        counts = global_counts[threshold]
        counts["tp"] += tp
        counts["fp"] += fp
        counts["fn"] += fn
        counts["pred"] += int(pred.sum())
        counts["truth"] += int(truth.sum())

    if (i + 1) % 100 == 0 or i + 1 == len(records):
        print(f"Processed {i + 1}/{len(records)} validation images")

# ------------------------------------------------------------
# 5. Summarize threshold performance
# ------------------------------------------------------------

rows = []
for threshold in THRESHOLDS:
    values = per_image[threshold]
    counts = global_counts[threshold]

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    dice_den = 2 * tp + fp + fn
    union = tp + fp + fn

    rows.append({
        "threshold": threshold,
        "mean_image_dice": float(np.mean([x["dice"] for x in values])),
        "median_image_dice": float(np.median([x["dice"] for x in values])),
        "mean_image_iou": float(np.mean([x["iou"] for x in values])),
        "mean_image_precision": float(np.mean([x["precision"] for x in values])),
        "mean_image_recall": float(np.mean([x["recall"] for x in values])),
        "mean_predicted_coverage": float(
            np.mean([x["predicted_coverage"] for x in values])
        ),
        "mean_ground_truth_coverage": float(
            np.mean([x["ground_truth_coverage"] for x in values])
        ),
        "global_pixel_dice": (
            2 * tp / dice_den if dice_den else 1.0
        ),
        "global_pixel_iou": tp / union if union else 1.0,
        "global_pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "global_pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
        "global_predicted_pixels": counts["pred"],
        "global_ground_truth_pixels": counts["truth"],
    })

results = pd.DataFrame(rows)
results_path = OUT / "stage_4_6d_threshold_results.csv"
results.to_csv(results_path, index=False)

# Select threshold using mean per-image validation Dice only.
best_row = results.sort_values(
    ["mean_image_dice", "mean_image_recall"],
    ascending=False
).iloc[0]

summary = {
    "stage": "4.6D",
    "status": "THRESHOLD_DIAGNOSTIC_COMPLETE",
    "model_type": ckpt["model_type"],
    "checkpoint_best_epoch": int(ckpt["best_epoch"]),
    "validation_records_evaluated": len(records),
    "thresholds_evaluated": THRESHOLDS,
    "selection_metric": "mean_image_dice",
    "best_validation_threshold": float(best_row["threshold"]),
    "best_mean_image_dice": float(best_row["mean_image_dice"]),
    "best_mean_image_iou": float(best_row["mean_image_iou"]),
    "best_mean_image_precision": float(best_row["mean_image_precision"]),
    "best_mean_image_recall": float(best_row["mean_image_recall"]),
    "test_set_used": False,
    "training_performed": False,
    "results_csv": str(results_path),
    "elapsed_seconds": round(time.time() - start_time, 2),
}

summary_path = OUT / "stage_4_6d_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 68)
print("STAGE 4.6D COMPLETE")
print("=" * 68)
print(results[
    ["threshold", "mean_image_dice", "mean_image_precision",
     "mean_image_recall", "mean_predicted_coverage",
     "global_pixel_dice", "global_pixel_recall"]
].to_string(index=False))
print("\nBest validation threshold:", summary["best_validation_threshold"])
print("Best mean-image Dice:", round(summary["best_mean_image_dice"], 6))
print("Results CSV:", results_path)
print("Summary JSON:", summary_path)
print("Test set used: NO")
print("=" * 68)
