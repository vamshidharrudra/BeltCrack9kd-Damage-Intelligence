
from pathlib import Path
import csv
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


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
    / "stage_4_6f_qualitative_inspection"
)

IMAGE_SIZE = 256
THRESHOLD = 0.10
NUMBER_OF_IMAGES = 20
DEVICE = torch.device("cpu")


# ============================================================
# SAME PREPROCESSING AS STAGE 4.6B
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
    tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
    return image, tensor


def mask_to_array(mask):
    mask = resize_and_pad(
        mask.convert("L"), IMAGE_SIZE, is_mask=True
    )
    return (np.asarray(mask, dtype=np.uint8) > 0)


# ============================================================
# MODEL - SAME SMALL U-NET AS STAGE 4.6B
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3, padding=1, bias=False
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
# MANIFEST PATH RESOLUTION
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
        "Cannot extract frame number: " + str(row.to_dict())
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
        "Image not found for row: " + str(row.to_dict())
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
        f"Mask not found for sequence={sequence}, frame={frame}"
    )


# ============================================================
# METRICS AND VISUALIZATION
# ============================================================

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return dice, precision, recall, tp, fp, fn


def make_overlay(image, prediction):
    base = np.asarray(image.convert("RGB")).copy()

    # Mark predicted crack pixels in red.
    base[prediction] = (
        0.65 * base[prediction]
        + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return Image.fromarray(base)


def save_comparison(image, target, prediction, output_path):
    panel_width = IMAGE_SIZE
    header_height = 32
    panel_height = IMAGE_SIZE

    canvas = Image.new(
        "RGB",
        (panel_width * 4, panel_height + header_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    target_image = Image.fromarray(
        target.astype(np.uint8) * 255
    ).convert("RGB")

    prediction_image = Image.fromarray(
        prediction.astype(np.uint8) * 255
    ).convert("RGB")

    panels = [
        ("Input image", image.convert("RGB")),
        ("Ground truth", target_image),
        ("Predicted mask", prediction_image),
        ("Prediction overlay", make_overlay(image, prediction)),
    ]

    for index, (label, panel) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 6, 8), label, fill=(0, 0, 0))
        canvas.paste(panel, (x, header_height))

    canvas.save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT_DIR / "comparisons"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 4.6F - QUALITATIVE SEGMENTATION INSPECTION")
    print("=" * 68)
    print("Training: NO")
    print("Test split used: NO")
    print(f"Prediction threshold: {THRESHOLD:.2f}")

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
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    # Only validation rows are considered.
    validation = manifest[
        manifest["split"].astype(str).str.lower().eq("val")
    ].copy().reset_index(drop=True)

    if validation.empty:
        raise RuntimeError("No validation records found.")

    # Select up to 20 evenly spaced validation records deterministically.
    count = min(NUMBER_OF_IMAGES, len(validation))
    selected_indices = np.linspace(
        0, len(validation) - 1, num=count, dtype=int
    )
    selected = validation.iloc[selected_indices].copy()

    print(f"Validation records available: {len(validation)}")
    print(f"Selected images: {len(selected)}")

    records = []
    for _, row in selected.iterrows():
        records.append({
            "sequence": str(row["sequence"]),
            "filename": str(row["canonical_image_filename"]),
            "image_path": resolve_image_path(row),
            "mask_path": resolve_mask_path(row),
        })

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = SmallUNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Checkpoint best epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print("Generating comparison images...")

    rows = []

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            with Image.open(record["image_path"]) as img:
                original = img.convert("RGB")

            with Image.open(record["mask_path"]) as img:
                mask = img.convert("L")

            if original.size != mask.size:
                raise RuntimeError(
                    f"Image/mask dimensions differ: "
                    f"{record['image_path']} vs {record['mask_path']}"
                )

            processed_image, x = image_to_tensor(original)
            target = mask_to_array(mask)

            probability = torch.sigmoid(
                model(x.unsqueeze(0).to(DEVICE))
            )[0, 0].cpu().numpy()

            prediction = probability >= THRESHOLD

            dice, precision, recall, tp, fp, fn = calculate_metrics(
                prediction, target
            )

            output_name = f"validation_{index:02d}.png"
            output_path = images_dir / output_name

            save_comparison(
                processed_image,
                target,
                prediction,
                output_path,
            )

            rows.append({
                "sample_number": index,
                "sequence": record["sequence"],
                "canonical_image_filename": record["filename"],
                "source_image": str(record["image_path"]),
                "ground_truth_mask": str(record["mask_path"]),
                "comparison_image": str(output_path),
                "threshold": THRESHOLD,
                "ground_truth_coverage": float(target.mean()),
                "predicted_coverage": float(prediction.mean()),
                "mean_probability": float(probability.mean()),
                "max_probability": float(probability.max()),
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
            })

            print(f"Saved comparison {index}/{len(records)}")

    csv_path = OUTPUT_DIR / "stage_4_6f_sample_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = {
        "stage": "4.6F",
        "status": "COMPLETED",
        "evaluation_split": "val",
        "training_performed": False,
        "test_split_used": False,
        "selected_images": len(rows),
        "validation_records_available": len(validation),
        "threshold": THRESHOLD,
        "checkpoint": str(CHECKPOINT_PATH),
        "comparison_directory": str(images_dir),
        "sample_metrics_csv": str(csv_path),
        "mean_sample_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_sample_precision": float(
            np.mean([r["precision"] for r in rows])
        ),
        "mean_sample_recall": float(
            np.mean([r["recall"] for r in rows])
        ),
        "elapsed_seconds": round(time.time() - started, 2),
    }

    import json
    with open(
        OUTPUT_DIR / "stage_4_6f_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("STAGE 4.6F SUMMARY")
    print(f"Images inspected: {len(rows)}")
    print(f"Mean sample Dice: {summary['mean_sample_dice']:.6f}")
    print(f"Mean sample precision: {summary['mean_sample_precision']:.6f}")
    print(f"Mean sample recall: {summary['mean_sample_recall']:.6f}")
    print(f"Comparison images: {images_dir}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")
    print("Training: NO")
    print("Test split used: NO")
    print("STAGE 4.6F COMPLETED")


if __name__ == "__main__":
    main()
