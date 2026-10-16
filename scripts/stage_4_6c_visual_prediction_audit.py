
from pathlib import Path
import json
import re
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# STAGE 4.6C - VISUAL PREDICTION AUDIT
# Validation set only. No training. No test-set access.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")
MANIFEST_PATH = PROJECT_ROOT / "canonical_dataset" / "canonical_manifest.csv"
MASK_ROOT = PROJECT_ROOT / "canonical_dataset" / "masks"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "segmentation"
    / "beltcrack9kd_cpu_pilot_best.pt"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "audit_outputs"
    / "stage_4_6c_visual_prediction_audit"
)

IMAGE_SIZE = 256
BATCH_SIZE = 4
PREDICTION_THRESHOLD = 0.50
MAX_VISUALIZATIONS = 24
SEED = 42

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
VIS_DIR = OUTPUT_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)
torch.manual_seed(SEED)

print("=" * 72)
print("STAGE 4.6C - VISUAL PREDICTION AUDIT")
print("=" * 72)
print("Device: CPU")
print("Validation only: YES")
print("Training: NO")
print("Test set used: NO")
print("Checkpoint:", CHECKPOINT_PATH)


# ------------------------------------------------------------
# 1. Reconstruct the SmallUNet architecture
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Load checkpoint and validate architecture compatibility
# ------------------------------------------------------------

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

if checkpoint.get("model_type") != "SmallUNet":
    raise ValueError(
        f"Unexpected model type: {checkpoint.get('model_type')}"
    )

if int(checkpoint.get("image_size", -1)) != IMAGE_SIZE:
    raise ValueError(
        f"Expected image size {IMAGE_SIZE}, "
        f"checkpoint says {checkpoint.get('image_size')}"
    )

model = SmallUNet()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

print("Checkpoint loaded successfully.")
print("Best epoch:", checkpoint.get("best_epoch"))
print("Recorded best validation Dice:",
      checkpoint.get("best_validation_dice"))


# ------------------------------------------------------------
# 3. Read the authoritative manifest and select validation rows
# ------------------------------------------------------------

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype=str,
    keep_default_na=False
)

required_columns = [
    "split",
    "sequence",
    "canonical_image_filename",
    "original_image_filename",
    "canonical_image_path",
    "source_image",
]

missing_columns = [
    col for col in required_columns if col not in manifest.columns
]

if missing_columns:
    raise ValueError(f"Manifest columns missing: {missing_columns}")

val_df = manifest[
    manifest["split"].str.strip().str.lower().eq("val")
].copy().reset_index(drop=True)

if val_df.empty:
    raise ValueError("No validation rows found in the manifest.")

print("Total manifest rows:", len(manifest))
print("Validation records:", len(val_df))


# ------------------------------------------------------------
# 4. Resolve image and mask paths
# ------------------------------------------------------------

def extract_frame_number(filename):
    if not filename:
        return None

    stem = Path(str(filename).replace("\\", "/")).stem
    match = re.search(r"(\d+)$", stem)

    if not match:
        return None

    return int(match.group(1))


def sequence_folder(sequence):
    normalized = str(sequence).replace("\\", "/").strip("/")
    return Path(normalized).name


def existing_path(value):
    if not value:
        return None

    p = Path(str(value).strip().strip('"'))

    candidates = [p]

    if not p.is_absolute():
        candidates.extend([
            PROJECT_ROOT / p,
            DATASET_ROOT / p,
        ])

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except (OSError, ValueError):
            pass

    return None


def resolve_image(row):
    # Prefer paths recorded in the authoritative manifest.
    for column in ["canonical_image_path", "source_image"]:
        path = existing_path(row[column])
        if path is not None:
            return path

    # Try the original and canonical filenames under the dataset root.
    for column in [
        "canonical_image_filename",
        "original_image_filename",
    ]:
        filename = row[column]
        if not filename:
            continue

        p = Path(str(filename).replace("\\", "/"))

        for candidate in [
            DATASET_ROOT / p,
            PROJECT_ROOT / p,
        ]:
            if candidate.is_file():
                return candidate.resolve()

    return None


def resolve_mask(row):
    frame = extract_frame_number(row["canonical_image_filename"])

    if frame is None:
        frame = extract_frame_number(row["original_image_filename"])

    if frame is None:
        return None

    folder = sequence_folder(row["sequence"])
    split = str(row["split"]).strip().lower()

    candidate = MASK_ROOT / split / folder / f"{frame}.png"

    if candidate.is_file():
        return candidate.resolve()

    return None


# ------------------------------------------------------------
# 5. Resolve all paths before inference
# ------------------------------------------------------------

resolved_rows = []
missing_images = []
missing_masks = []
unresolved_frames = []

for idx, row in val_df.iterrows():
    image_path = resolve_image(row)
    mask_path = resolve_mask(row)

    frame = extract_frame_number(row["canonical_image_filename"])
    if frame is None:
        frame = extract_frame_number(row["original_image_filename"])

    if frame is None:
        unresolved_frames.append(idx)

    if image_path is None:
        missing_images.append(idx)

    if mask_path is None:
        missing_masks.append(idx)

    if image_path is not None and mask_path is not None and frame is not None:
        resolved_rows.append({
            "manifest_index": idx,
            "sequence": row["sequence"],
            "frame": frame,
            "image_path": image_path,
            "mask_path": mask_path,
            "canonical_image_filename": row["canonical_image_filename"],
        })

print("Unresolved frame numbers:", len(unresolved_frames))
print("Missing images:", len(missing_images))
print("Missing masks:", len(missing_masks))

if missing_images:
    print("First missing image rows:", missing_images[:10])

if missing_masks:
    print("First missing mask rows:", missing_masks[:10])

if unresolved_frames:
    print("First unresolved frame rows:", unresolved_frames[:10])

if missing_images or missing_masks or unresolved_frames:
    raise RuntimeError(
        "Validation path resolution failed. "
        "Review the row indices above before continuing."
    )

if len(resolved_rows) != len(val_df):
    raise RuntimeError(
        f"Resolved {len(resolved_rows)} of {len(val_df)} validation records."
    )

print("Validation path resolution: PASS")


# ------------------------------------------------------------
# 6. Preprocessing and segmentation metrics
# ------------------------------------------------------------

def load_pair(record):
    with Image.open(record["image_path"]) as im:
        image = im.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    with Image.open(record["mask_path"]) as im:
        mask = im.convert("L")
        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )
        mask_array = np.asarray(mask, dtype=np.uint8) > 0

    # Match the standard RGB / 255 preprocessing used by the pilot.
    image_tensor = torch.from_numpy(
        np.transpose(image_array, (2, 0, 1)).copy()
    ).float()

    return image_array, image_tensor, mask_array


def calculate_metrics(prediction, truth):
    pred = prediction.astype(bool)
    true = truth.astype(bool)

    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())

    dice_denominator = 2 * tp + fp + fn
    union = tp + fp + fn

    dice = (
        (2.0 * tp / dice_denominator)
        if dice_denominator else 1.0
    )
    iou = (tp / union) if union else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ground_truth_pixels": int(true.sum()),
        "predicted_pixels": int(pred.sum()),
        "ground_truth_coverage": float(true.mean()),
        "predicted_coverage": float(pred.mean()),
    }


# ------------------------------------------------------------
# 7. Visual comparison writer
# ------------------------------------------------------------

def save_visualization(image_array, truth, prediction, record, metrics):
    # Construct a 2x2 comparison image using PIL.
    image_u8 = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    truth_u8 = (truth.astype(np.uint8) * 255)
    pred_u8 = (prediction.astype(np.uint8) * 255)

    # Ground truth shown in green; prediction shown in red.
    truth_overlay = image_u8.copy()
    truth_overlay[truth] = (
        0.35 * truth_overlay[truth] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)

    pred_overlay = image_u8.copy()
    pred_overlay[prediction] = (
        0.35 * pred_overlay[prediction] + 0.65 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # A white pixel indicates overlap; red is false positive;
    # green is false negative.
    comparison = np.zeros_like(image_u8)
    comparison[np.logical_and(prediction, truth)] = [255, 255, 255]
    comparison[np.logical_and(prediction, ~truth)] = [255, 0, 0]
    comparison[np.logical_and(~prediction, truth)] = [0, 255, 0]

    panels = [
        ("Original image", image_u8),
        ("Ground truth mask", np.repeat(truth_u8[:, :, None], 3, axis=2)),
        ("Prediction mask", np.repeat(pred_u8[:, :, None], 3, axis=2)),
        ("Overlay: red=FP, green=FN, white=TP", comparison),
    ]

    panel_size = IMAGE_SIZE
    title_height = 30
    canvas = Image.new(
        "RGB",
        (panel_size * 2, (panel_size + title_height) * 2),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    for i, (title, pixels) in enumerate(panels):
        x = (i % 2) * panel_size
        y = (i // 2) * (panel_size + title_height)

        panel = Image.fromarray(pixels.astype(np.uint8))
        canvas.paste(panel, (x, y + title_height))
        draw.text((x + 4, y + 7), title, fill="black")

    filename = (
        f"{record['manifest_index']:05d}_"
        f"{record['sequence'].replace('/', '_').replace(chr(92), '_')}_"
        f"{record['frame']}_dice_{metrics['dice']:.3f}.png"
    )

    canvas.save(VIS_DIR / filename)


# ------------------------------------------------------------
# 8. Inference over the entire validation split
# ------------------------------------------------------------

records_out = []
visual_cache = []
start_time = time.time()

print("\nStarting validation inference...")

for start in range(0, len(resolved_rows), BATCH_SIZE):
    batch_records = resolved_rows[start:start + BATCH_SIZE]

    images = []
    pair_cache = []

    for record in batch_records:
        image_array, image_tensor, truth = load_pair(record)
        images.append(image_tensor)
        pair_cache.append((image_array, truth))

    batch_tensor = torch.stack(images, dim=0)

    with torch.inference_mode():
        logits = model(batch_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = probabilities[:, 0].numpy() >= PREDICTION_THRESHOLD

    for i, record in enumerate(batch_records):
        image_array, truth = pair_cache[i]
        prediction = predictions[i]

        metrics = calculate_metrics(prediction, truth)

        records_out.append({
            "manifest_index": record["manifest_index"],
            "sequence": record["sequence"],
            "frame": record["frame"],
            "image_path": str(record["image_path"]),
            "mask_path": str(record["mask_path"]),
            "canonical_image_filename": record["canonical_image_filename"],
            "threshold": PREDICTION_THRESHOLD,
            **metrics,
        })

        # Retain data only for selecting representative visualizations.
        visual_cache.append({
            "record": record,
            "image": image_array,
            "truth": truth,
            "prediction": prediction,
            "metrics": metrics,
        })

    completed = min(start + BATCH_SIZE, len(resolved_rows))
    if completed % 100 == 0 or completed == len(resolved_rows):
        print(f"Processed {completed}/{len(resolved_rows)} validation images")

metrics_df = pd.DataFrame(records_out)
metrics_csv = OUTPUT_ROOT / "stage_4_6c_validation_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False)


# ------------------------------------------------------------
# 9. Save representative visualizations
# ------------------------------------------------------------

# Include low, middle, and high Dice examples.
ordered = sorted(visual_cache, key=lambda item: item["metrics"]["dice"])
n = len(ordered)

selected = []
if n <= MAX_VISUALIZATIONS:
    selected = ordered
else:
    positions = np.linspace(0, n - 1, MAX_VISUALIZATIONS, dtype=int)
    selected = [ordered[int(pos)] for pos in positions]

for item in selected:
    save_visualization(
        item["image"],
        item["truth"],
        item["prediction"],
        item["record"],
        item["metrics"],
    )

print("Visualizations saved:", len(selected))


# ------------------------------------------------------------
# 10. Aggregate summary
# ------------------------------------------------------------

summary = {
    "stage": "4.6C",
    "status": "VISUAL_PREDICTION_AUDIT_COMPLETE",
    "model_type": checkpoint["model_type"],
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
    "checkpoint_recorded_best_validation_dice": float(
        checkpoint["best_validation_dice"]
    ),
    "image_size": IMAGE_SIZE,
    "prediction_threshold": PREDICTION_THRESHOLD,
    "validation_records_evaluated": int(len(metrics_df)),
    "test_set_used": False,
    "training_performed": False,
    "mean_dice": float(metrics_df["dice"].mean()),
    "median_dice": float(metrics_df["dice"].median()),
    "mean_iou": float(metrics_df["iou"].mean()),
    "mean_precision": float(metrics_df["precision"].mean()),
    "mean_recall": float(metrics_df["recall"].mean()),
    "mean_ground_truth_coverage": float(
        metrics_df["ground_truth_coverage"].mean()
    ),
    "mean_predicted_coverage": float(
        metrics_df["predicted_coverage"].mean()
    ),
    "images_with_zero_predicted_pixels": int(
        (metrics_df["predicted_pixels"] == 0).sum()
    ),
    "images_with_zero_true_positive_pixels": int(
        (metrics_df["tp"] == 0).sum()
    ),
    "visualizations_saved": int(len(selected)),
    "metrics_csv": str(metrics_csv),
    "visualization_directory": str(VIS_DIR),
    "elapsed_seconds": round(time.time() - start_time, 2),
}

summary_path = OUTPUT_ROOT / "stage_4_6c_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 72)
print("STAGE 4.6C COMPLETE")
print("=" * 72)
print("Validation images evaluated:", len(metrics_df))
print("Mean Dice:", round(summary["mean_dice"], 6))
print("Median Dice:", round(summary["median_dice"], 6))
print("Mean IoU:", round(summary["mean_iou"], 6))
print("Mean precision:", round(summary["mean_precision"], 6))
print("Mean recall:", round(summary["mean_recall"], 6))
print("Mean ground-truth crack coverage:",
      round(summary["mean_ground_truth_coverage"], 6))
print("Mean predicted crack coverage:",
      round(summary["mean_predicted_coverage"], 6))
print("Zero-prediction images:",
      summary["images_with_zero_predicted_pixels"])
print("Zero-true-positive images:",
      summary["images_with_zero_true_positive_pixels"])
print("Metrics CSV:", metrics_csv)
print("Visualizations:", VIS_DIR)
print("Summary JSON:", summary_path)
print("Test set used: NO")
print("=" * 72)
