
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

MANIFEST_PATH = (
    PROJECT_ROOT / "canonical_dataset" / "canonical_manifest.csv"
)
MASK_ROOT = PROJECT_ROOT / "canonical_dataset" / "masks"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "beltcrack9kd_cpu_pilot_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_6e_corrected_validation_diagnostic"
)

IMAGE_SIZE = 256
BATCH_SIZE = 2
THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90
]
EPS = 1e-8
DEVICE = torch.device("cpu")


# ============================================================
# EXACT PREPROCESSING FROM STAGE 4.6B
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
    mask = resize_and_pad(
        mask.convert("L"), IMAGE_SIZE, is_mask=True
    )
    y = np.asarray(mask, dtype=np.uint8)
    y = (y > 0).astype(np.float32)
    return torch.from_numpy(y.copy()).unsqueeze(0)


# ============================================================
# MODEL ARCHITECTURE FROM THE CPU PILOT
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
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

        self.out = nn.Conv2d(16, 1, kernel_size=1)

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
    for column in (
        "canonical_image_filename",
        "original_image_filename",
    ):
        value = row.get(column)
        if pd.notna(value):
            match = re.search(r"(\d+)(?=\.[^.]+$)", str(value))
            if match:
                return int(match.group(1))

    raise ValueError(
        "Could not extract a numeric frame number from row: "
        + str(row.to_dict())
    )


def resolve_image_path(row):
    candidates = []

    for column in ("source_image", "canonical_image_path"):
        value = row.get(column)

        if pd.notna(value) and str(value).strip():
            path = Path(str(value))
            candidates.append(path)

            if not path.is_absolute():
                candidates.append(DATASET_ROOT / path)
                candidates.append(PROJECT_ROOT / path)

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not resolve source image for manifest row: "
        + str(row.to_dict())
    )


def resolve_mask_path(row):
    split = str(row["split"]).strip().lower()
    sequence = str(row["sequence"]).replace("\\", "/").strip("/")
    sequence_folder = Path(sequence).name
    frame = frame_number_from_row(row)

    candidates = [
        MASK_ROOT / split / sequence_folder / f"{frame}.png",
        MASK_ROOT / split / sequence / f"{frame}.png",
    ]

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        f"Mask not found for split={split}, "
        f"sequence={sequence}, frame={frame}. "
        f"Checked: {candidates}"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return dice, precision, recall, tp, fp, fn


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 4.6E - CORRECTED VALIDATION DIAGNOSTIC")
    print("=" * 68)
    print("Device: CPU")
    print("Training: NO")
    print("Validation split only: YES")
    print("Test split access: NO")
    print("Preprocessing: Stage 4.6B resize-and-pad")
    print()

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "split",
        "sequence",
        "canonical_image_filename",
        "original_image_filename",
        "source_image",
    }
    missing_columns = required - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Filter before resolving any paths: no test records are loaded.
    validation = manifest[
        manifest["split"].astype(str).str.lower().eq("val")
    ].copy().reset_index(drop=True)

    if validation.empty:
        raise RuntimeError("No validation records found.")

    if validation["split"].astype(str).str.lower().ne("val").any():
        raise RuntimeError("Safety check failed: non-validation rows found.")

    print(f"Validation records: {len(validation)}")
    print("Resolving validation image/mask pairs...")

    records = []
    for index, row in validation.iterrows():
        image_path = resolve_image_path(row)
        mask_path = resolve_mask_path(row)

        records.append({
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            "sequence": str(row["sequence"]),
            "filename": str(row["canonical_image_filename"]),
        })

    print(f"Resolved pairs: {len(records)}")
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "No evaluation was performed."
        )

    model = SmallUNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    best_epoch = checkpoint.get("best_epoch", "unknown")
    print(f"Checkpoint best epoch: {best_epoch}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Running validation inference...")

    # Store predictions and targets once, then evaluate thresholds.
    # Images are processed in small batches to limit CPU memory use.
    probability_batches = []
    target_batches = []
    image_rows = []

    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start:start + BATCH_SIZE]
            xs = []
            ys = []

            for record in batch_records:
                with Image.open(record["image_path"]) as img:
                    image = img.convert("RGB")
                with Image.open(record["mask_path"]) as img:
                    mask = img.convert("L")

                if image.size != mask.size:
                    raise RuntimeError(
                        f"Image/mask size mismatch: "
                        f"{record['image_path']} vs {record['mask_path']}"
                    )

                xs.append(image_to_tensor(image))
                ys.append(mask_to_tensor(mask))

            x = torch.stack(xs, dim=0).to(DEVICE)
            y = torch.stack(ys, dim=0).to(DEVICE)

            logits = model(x)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()

            for local_index, record in enumerate(batch_records):
                probability_batches.append(probabilities[local_index, 0])
                target_batches.append(targets[local_index, 0])

                image_rows.append({
                    "sequence": record["sequence"],
                    "canonical_image_filename": record["filename"],
                    "image_path": str(record["image_path"]),
                    "mask_path": str(record["mask_path"]),
                    "ground_truth_coverage": float(
                        targets[local_index, 0].mean()
                    ),
                    "mean_probability": float(
                        probabilities[local_index, 0].mean()
                    ),
                    "max_probability": float(
                        probabilities[local_index, 0].max()
                    ),
                })

            completed = min(start + len(batch_records), len(records))
            if completed % 200 == 0 or completed == len(records):
                print(f"Processed {completed}/{len(records)}")

    if len(probability_batches) != len(validation):
        raise RuntimeError("Not all validation records were evaluated.")

    print("Computing threshold metrics...")

    threshold_rows = []
    for threshold in THRESHOLDS:
        image_dice = []
        image_precision = []
        image_recall = []
        predicted_coverage = []

        global_tp = 0
        global_fp = 0
        global_fn = 0

        for probability, target in zip(
            probability_batches, target_batches
        ):
            prediction = probability >= threshold
            dice, precision, recall, tp, fp, fn = calculate_metrics(
                prediction, target
            )

            image_dice.append(dice)
            image_precision.append(precision)
            image_recall.append(recall)
            predicted_coverage.append(float(prediction.mean()))

            global_tp += tp
            global_fp += fp
            global_fn += fn

        global_dice = (
            2 * global_tp
            / (2 * global_tp + global_fp + global_fn + EPS)
        )
        global_recall = global_tp / (
            global_tp + global_fn + EPS
        )
        global_precision = global_tp / (
            global_tp + global_fp + EPS
        )

        threshold_rows.append({
            "threshold": threshold,
            "mean_image_dice": float(np.mean(image_dice)),
            "mean_image_precision": float(np.mean(image_precision)),
            "mean_image_recall": float(np.mean(image_recall)),
            "mean_predicted_coverage": float(np.mean(predicted_coverage)),
            "global_pixel_dice": float(global_dice),
            "global_pixel_precision": float(global_precision),
            "global_pixel_recall": float(global_recall),
        })

    results = pd.DataFrame(threshold_rows)
    results.to_csv(
        OUTPUT_DIR / "stage_4_6e_corrected_threshold_results.csv",
        index=False,
    )

    pd.DataFrame(image_rows).to_csv(
        OUTPUT_DIR / "stage_4_6e_validation_prediction_summary.csv",
        index=False,
    )

    best_row = results.loc[results["mean_image_dice"].idxmax()]

    summary = {
        "stage": "4.6E",
        "status": "COMPLETED",
        "evaluation_split": "val",
        "training_performed": False,
        "test_split_used": False,
        "validation_records": int(len(validation)),
        "resolved_image_mask_pairs": int(len(records)),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_best_epoch": best_epoch,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "preprocessing": {
            "aspect_ratio_preserved": True,
            "center_padding": True,
            "image_resampling": "BILINEAR",
            "mask_resampling": "NEAREST",
            "image_normalization": "(x / 255 - 0.5) / 0.5",
            "mask_binarization": "mask > 0",
        },
        "best_validation_threshold_by_mean_image_dice": float(
            best_row["threshold"]
        ),
        "best_mean_image_dice": float(best_row["mean_image_dice"]),
        "best_mean_image_precision": float(
            best_row["mean_image_precision"]
        ),
        "best_mean_image_recall": float(best_row["mean_image_recall"]),
        "best_global_pixel_dice": float(best_row["global_pixel_dice"]),
        "elapsed_seconds": round(time.time() - started, 2),
        "interpretation_note": (
            "Validation-only diagnostic. Threshold selection is based on "
            "validation data and must not be reported as an unbiased test "
            "result. No test split was loaded."
        ),
    }

    with open(
        OUTPUT_DIR / "stage_4_6e_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("THRESHOLD RESULTS")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Best validation threshold: {best_row['threshold']:.2f}")
    print(f"Best mean-image Dice: {best_row['mean_image_dice']:.6f}")
    print(f"Best mean-image precision: {best_row['mean_image_precision']:.6f}")
    print(f"Best mean-image recall: {best_row['mean_image_recall']:.6f}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Training: NO")
    print("Test split used: NO")
    print("STAGE 4.6E COMPLETED")


if __name__ == "__main__":
    main()
