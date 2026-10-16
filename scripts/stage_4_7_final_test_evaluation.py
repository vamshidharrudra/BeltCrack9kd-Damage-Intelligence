
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
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

MANIFEST_PATH = PROJECT_ROOT / "canonical_dataset" / "canonical_manifest.csv"
MASK_ROOT = PROJECT_ROOT / "canonical_dataset" / "masks"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "segmentation"
    / "beltcrack9kd_cpu_pilot_best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "audit_outputs" / "stage_4_7_final_test"

IMAGE_SIZE = 256
BATCH_SIZE = 2
THRESHOLD = 0.10  # Selected previously using validation data
DEVICE = torch.device("cpu")


# ============================================================
# TRAINING-MATCHED PREPROCESSING
# ============================================================

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


def image_to_tensor(image):
    image = resize_and_pad(image.convert("RGB"), IMAGE_SIZE)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


def mask_to_tensor(mask):
    mask = resize_and_pad(mask.convert("L"), IMAGE_SIZE, is_mask=True)
    y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    return torch.from_numpy(y.copy()).unsqueeze(0)


# ============================================================
# SAME SMALL U-NET ARCHITECTURE
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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

        d4 = self.d4(torch.cat([self.u4(mid), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


# ============================================================
# PATH RESOLUTION
# ============================================================

def frame_number_from_row(row):
    for column in ("canonical_image_filename", "original_image_filename"):
        value = row.get(column)
        if pd.notna(value):
            match = re.search(r"(\d+)(?=\.[^.]+$)", str(value))
            if match:
                return int(match.group(1))
    raise ValueError("Cannot extract frame number: " + str(row.to_dict()))


def resolve_image_path(row):
    candidates = []
    for column in ("source_image", "canonical_image_path"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            path = Path(str(value))
            candidates.append(path)
            if not path.is_absolute():
                candidates.extend([DATASET_ROOT / path, PROJECT_ROOT / path])

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Test image not found: " + str(row.to_dict())
    )


def resolve_mask_path(row):
    split = str(row["split"]).strip().lower()
    sequence = str(row["sequence"]).replace("\\", "/").strip("/")
    frame = frame_number_from_row(row)

    candidates = [
        MASK_ROOT / split / Path(sequence).name / f"{frame}.png",
        MASK_ROOT / split / sequence / f"{frame}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        f"Test mask not found: sequence={sequence}, frame={frame}"
    )


# ============================================================
# METRICS
# ============================================================

def pixel_counts(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return tp, fp, fn, tn


def safe_divide(a, b):
    return float(a / b) if b else 0.0


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 4.7 - FINAL TEST EVALUATION")
    print("=" * 68)
    print("Training: NO")
    print("Threshold tuning: NO")
    print(f"Fixed validation-selected threshold: {THRESHOLD:.2f}")

    for path in (MANIFEST_PATH, CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    manifest = pd.read_csv(MANIFEST_PATH)
    required = {
        "split", "sequence", "canonical_image_filename",
        "original_image_filename", "source_image",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    # Filter first; only test rows are resolved and evaluated.
    test = manifest[
        manifest["split"].astype(str).str.lower().eq("test")
    ].copy().reset_index(drop=True)

    if test.empty:
        raise RuntimeError("No test records found.")

    print(f"Test records: {len(test)}")
    print("Resolving test image/mask pairs...")

    records = []
    for _, row in test.iterrows():
        records.append({
            "sequence": str(row["sequence"]),
            "filename": str(row["canonical_image_filename"]),
            "image_path": resolve_image_path(row),
            "mask_path": resolve_mask_path(row),
        })

    print(f"Resolved pairs: {len(records)}")
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint lacks model_state_dict.")

    model = SmallUNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Checkpoint best epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print("Running test inference...")

    image_rows = []
    total_tp = total_fp = total_fn = total_tn = 0
    image_dice_values = []
    image_precision_values = []
    image_recall_values = []
    image_coverage_values = []

    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start:start + BATCH_SIZE]
            xs, ys = [], []

            for record in batch_records:
                with Image.open(record["image_path"]) as img:
                    image = img.convert("RGB")
                with Image.open(record["mask_path"]) as img:
                    mask = img.convert("L")

                if image.size != mask.size:
                    raise RuntimeError(
                        f"Image/mask dimension mismatch: "
                        f"{record['image_path']} vs {record['mask_path']}"
                    )

                xs.append(image_to_tensor(image))
                ys.append(mask_to_tensor(mask))

            x = torch.stack(xs).to(DEVICE)
            y = torch.stack(ys).to(DEVICE)
            probabilities = torch.sigmoid(model(x)).cpu().numpy()
            targets = y.cpu().numpy()

            for j, record in enumerate(batch_records):
                prediction = probabilities[j, 0] >= THRESHOLD
                target = targets[j, 0] > 0

                tp, fp, fn, tn = pixel_counts(prediction, target)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                dice = safe_divide(2 * tp, 2 * tp + fp + fn)
                precision = safe_divide(tp, tp + fp)
                recall = safe_divide(tp, tp + fn)

                image_dice_values.append(dice)
                image_precision_values.append(precision)
                image_recall_values.append(recall)
                image_coverage_values.append(float(prediction.mean()))

                image_rows.append({
                    "sequence": record["sequence"],
                    "canonical_image_filename": record["filename"],
                    "threshold": THRESHOLD,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "ground_truth_coverage": float(target.mean()),
                    "predicted_coverage": float(prediction.mean()),
                    "true_positive_pixels": tp,
                    "false_positive_pixels": fp,
                    "false_negative_pixels": fn,
                    "true_negative_pixels": tn,
                })

            completed = min(start + len(batch_records), len(records))
            if completed % 200 == 0 or completed == len(records):
                print(f"Processed {completed}/{len(records)}")

    global_dice = safe_divide(
        2 * total_tp, 2 * total_tp + total_fp + total_fn
    )
    global_precision = safe_divide(total_tp, total_tp + total_fp)
    global_recall = safe_divide(total_tp, total_tp + total_fn)

    per_image_csv = OUTPUT_DIR / "stage_4_7_test_image_metrics.csv"
    pd.DataFrame(image_rows).to_csv(per_image_csv, index=False)

    summary = {
        "stage": "4.7",
        "status": "COMPLETED",
        "evaluation_split": "test",
        "test_records": len(records),
        "resolved_pairs": len(records),
        "training_performed": False,
        "threshold_tuning_performed": False,
        "threshold": THRESHOLD,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_best_epoch": checkpoint.get("best_epoch", "unknown"),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "preprocessing": "Stage 4.6B resize-and-pad and normalization",
        "mean_image_dice": float(np.mean(image_dice_values)),
        "mean_image_precision": float(np.mean(image_precision_values)),
        "mean_image_recall": float(np.mean(image_recall_values)),
        "mean_predicted_coverage": float(np.mean(image_coverage_values)),
        "global_pixel_dice": global_dice,
        "global_pixel_precision": global_precision,
        "global_pixel_recall": global_recall,
        "total_true_positive_pixels": total_tp,
        "total_false_positive_pixels": total_fp,
        "total_false_negative_pixels": total_fn,
        "per_image_metrics_csv": str(per_image_csv),
        "elapsed_seconds": round(time.time() - started, 2),
        "interpretation": (
            "Final evaluation of the existing pilot checkpoint on the test "
            "split. Results apply to this pilot and preprocessing setup."
        ),
    }

    summary_path = OUTPUT_DIR / "stage_4_7_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print()
    print("FINAL TEST METRICS")
    print(f"Mean image Dice:      {summary['mean_image_dice']:.6f}")
    print(f"Mean image precision: {summary['mean_image_precision']:.6f}")
    print(f"Mean image recall:    {summary['mean_image_recall']:.6f}")
    print(f"Global pixel Dice:    {global_dice:.6f}")
    print(f"Global pixel precision: {global_precision:.6f}")
    print(f"Global pixel recall:    {global_recall:.6f}")
    print(f"Mean predicted coverage: {summary['mean_predicted_coverage']:.6f}")
    print(f"Test images evaluated: {len(records)}")
    print(f"Per-image CSV: {per_image_csv}")
    print(f"Summary JSON: {summary_path}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    print("Training: NO")
    print("Threshold tuning: NO")
    print("STAGE 4.7 COMPLETED")


if __name__ == "__main__":
    main()
