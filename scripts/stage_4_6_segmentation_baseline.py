import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# STAGE 4.6
# LEAKAGE-SAFE CRACK SEGMENTATION BASELINE
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "canonical_dataset"
    / "canonical_manifest.csv"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "canonical_dataset"
    / "masks"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_6_segmentation_baseline"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 512

BATCH_SIZE = 4

NUM_EPOCHS = 10

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 3

THRESHOLD = 0.50

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# None = use all records.
# Keep these configurable so the baseline can be
# reduced if hardware memory is insufficient.

DEVICE_REQUESTED = "auto"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

# Torch is intentionally imported after basic checks.
try:

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

except ImportError as exc:

    print("=" * 80)
    print("PYTORCH IS NOT INSTALLED")
    print("=" * 80)
    print()
    print("Install PyTorch for your system, then rerun this script.")
    print()
    print(
        "Example CPU installation command:"
    )
    print(
        "python -m pip install torch torchvision"
    )
    print()
    print(f"Import error: {exc}")
    raise


# ============================================================
# DEVICE
# ============================================================

if DEVICE_REQUESTED == "auto":

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

else:

    DEVICE = torch.device(
        DEVICE_REQUESTED
    )


# ============================================================
# TORCH SEED
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def json_safe(value):

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, torch.Tensor):

        if value.numel() == 1:
            return float(value.detach().cpu().item())

        return value.detach().cpu().tolist()

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):

        if math.isfinite(value):
            return float(value)

        return None

    return value


def extract_frame_number(filename):

    stem = Path(
        str(filename)
    ).stem

    if stem.isdigit():
        return int(stem)

    digits = ""

    for char in reversed(stem):

        if char.isdigit():
            digits = char + digits

        else:
            break

    if digits:
        return int(digits)

    return None


def resolve_source_image(row):

    source_image = str(
        row["source_image"]
    ).strip()

    candidate = Path(source_image)

    if candidate.exists():
        return candidate

    candidate = (
        DATASET_ROOT
        / source_image
    )

    if candidate.exists():
        return candidate

    split = str(row["split"])

    sequence = str(row["sequence"])

    data_folder = sequence.split("/")[-1]

    filename = str(
        row["original_image_filename"]
    )

    fallback = (
        DATASET_ROOT
        / "BeltCrack9kd1"
        / "images"
        / split
        / data_folder
        / filename
    )

    if fallback.exists():
        return fallback

    return None


def resolve_mask_path(row):

    split = str(
        row["split"]
    )

    sequence = str(
        row["sequence"]
    )

    data_folder = sequence.split("/")[-1]

    frame_number = extract_frame_number(
        row["canonical_image_filename"]
    )

    if frame_number is None:
        return None

    path = (
        MASK_ROOT
        / split
        / data_folder
        / f"{frame_number}.png"
    )

    if path.exists():
        return path

    return None


# ============================================================
# LETTERBOX / PADDING
# ============================================================

def resize_and_pad(image, target_size):

    width, height = image.size

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (target_size, target_size),
        (0, 0, 0)
    )

    left = (
        target_size - new_width
    ) // 2

    top = (
        target_size - new_height
    ) // 2

    canvas.paste(
        resized,
        (left, top)
    )

    return canvas, scale, left, top


def resize_and_pad_mask(mask, target_size):

    width, height = mask.size

    scale = min(
        target_size / width,
        target_size / height
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = mask.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST
    )

    canvas = Image.new(
        "L",
        (target_size, target_size),
        0
    )

    left = (
        target_size - new_width
    ) // 2

    top = (
        target_size - new_height
    ) // 2

    canvas.paste(
        resized,
        (left, top)
    )

    return canvas


# ============================================================
# DATASET
# ============================================================

class BeltCrackSegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_size=512,
        augment=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_size = image_size
        self.augment = augment

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["_resolved_image"]
        )

        mask_path = Path(
            row["_resolved_mask"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image, _, _, _ = resize_and_pad(
            image,
            self.image_size
        )

        mask = resize_and_pad_mask(
            mask,
            self.image_size
        )

        # ----------------------------------------------------
        # SAFE AUGMENTATION
        # ----------------------------------------------------
        #
        # Horizontal flipping is used because it does not
        # change crack topology.
        #
        # Vertical flipping is intentionally NOT used because
        # conveyor orientation can be physically meaningful.
        #
        if self.augment:

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Normalize to approximately [-1, 1]
        image_array = (
            image_array - 0.5
        ) / 0.5

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1)
        ).float()

        mask_array = (
            np.asarray(mask, dtype=np.float32)
            / 255.0
        )

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0).float()

        return (
            image_tensor,
            mask_tensor
        )


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(
            3,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# METRICS
# ============================================================

def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    tp = (
        predictions * targets
    ).sum().item()

    fp = (
        predictions * (1.0 - targets)
    ).sum().item()

    fn = (
        (1.0 - predictions) * targets
    ).sum().item()

    intersection = tp

    union = (
        tp + fp + fn
    )

    dice = (
        (2.0 * tp + eps)
        /
        (2.0 * tp + fp + fn + eps)
    )

    iou = (
        (intersection + eps)
        /
        (union + eps)
    )

    precision = (
        (tp + eps)
        /
        (tp + fp + eps)
    )

    recall = (
        (tp + eps)
        /
        (tp + fn + eps)
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    eps=1e-7
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.reshape(
        probabilities.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        (2.0 * intersection + eps)
        /
        (denominator + eps)
    )

    return (
        1.0 - dice
    ).mean()


def combined_loss(
    logits,
    targets
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dloss = dice_loss(
        logits,
        targets
    )

    return (
        0.5 * bce
        +
        0.5 * dloss
    )


# ============================================================
# EPOCH FUNCTION
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    sample_count = 0

    for images, masks in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                masks
            )

            if training:

                loss.backward()

                optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        target_binary = (
            masks >= 0.5
        ).float()

        total_tp += (
            predictions
            * target_binary
        ).sum().item()

        total_fp += (
            predictions
            * (1.0 - target_binary)
        ).sum().item()

        total_fn += (
            (1.0 - predictions)
            * target_binary
        ).sum().item()

        sample_count += batch_size

    eps = 1e-7

    dice = (
        (2.0 * total_tp + eps)
        /
        (
            2.0 * total_tp
            + total_fp
            + total_fn
            + eps
        )
    )

    iou = (
        (total_tp + eps)
        /
        (
            total_tp
            + total_fp
            + total_fn
            + eps
        )
    )

    precision = (
        (total_tp + eps)
        /
        (
            total_tp
            + total_fp
            + eps
        )
    )

    recall = (
        (total_tp + eps)
        /
        (
            total_tp
            + total_fn
            + eps
        )
    )

    mean_loss = (
        total_loss / sample_count
        if sample_count > 0
        else float("nan")
    )

    return {
        "loss": mean_loss,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STAGE 4.6 - LEAKAGE-SAFE CRACK SEGMENTATION BASELINE")
print("=" * 80)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "CUDA device:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "CUDA available: NO"
    )

print()
print(
    f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Seed: {SEED}"
)


# ============================================================
# INPUT CHECK
# ============================================================

print()
print("=" * 80)
print("CHECKING VERIFIED INPUTS")
print("=" * 80)

if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

if not MASK_ROOT.exists():

    raise FileNotFoundError(
        f"Mask root not found: {MASK_ROOT}"
    )

print("Manifest: PASS")
print("Mask root: PASS")


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

required_columns = [
    "split",
    "component",
    "sequence",
    "original_image_filename",
    "canonical_image_filename",
    "source_image",
]

missing = [
    col
    for col in required_columns
    if col not in manifest.columns
]

if missing:

    raise ValueError(
        f"Missing manifest columns: {missing}"
    )

print()
print(
    f"Canonical records: {len(manifest)}"
)


# ============================================================
# RESOLVE IMAGE + MASK PATHS
# ============================================================

print()
print("=" * 80)
print("RESOLVING IMAGE / MASK PAIRS")
print("=" * 80)

resolved_images = []
resolved_masks = []

missing_images = 0
missing_masks = 0

for idx, row in manifest.iterrows():

    image_path = resolve_source_image(
        row
    )

    mask_path = resolve_mask_path(
        row
    )

    if image_path is None:
        missing_images += 1

    if mask_path is None:
        missing_masks += 1

    resolved_images.append(
        str(image_path)
        if image_path is not None
        else ""
    )

    resolved_masks.append(
        str(mask_path)
        if mask_path is not None
        else ""
    )

manifest["_resolved_image"] = (
    resolved_images
)

manifest["_resolved_mask"] = (
    resolved_masks
)

print(
    f"Missing source images: {missing_images}"
)

print(
    f"Missing masks: {missing_masks}"
)

if missing_images != 0:

    raise RuntimeError(
        "Source image resolution failed."
    )

if missing_masks != 0:

    raise RuntimeError(
        "Mask resolution failed."
    )


# ============================================================
# SPLIT COUNTS
# ============================================================

print()
print("=" * 80)
print("VERIFYING LEAKAGE-SAFE SPLITS")
print("=" * 80)

split_counts = (
    manifest
    .groupby("split")
    .size()
)

print(
    split_counts.to_string()
)

expected_counts = {
    "train": 9819,
    "val": 2102,
    "test": 2166,
}

for split_name, expected in expected_counts.items():

    actual = int(
        split_counts.get(
            split_name,
            0
        )
    )

    print(
        f"{split_name}: "
        f"{actual} / {expected}"
    )

    if actual != expected:

        raise RuntimeError(
            f"Unexpected {split_name} count."
        )


# ============================================================
# COMPONENT INTEGRITY
# ============================================================

component_split_conflicts = (
    manifest
    .groupby("component")["split"]
    .nunique()
)

conflicting_components = (
    component_split_conflicts[
        component_split_conflicts > 1
    ]
)

print()
print(
    "Components crossing splits:",
    len(conflicting_components)
)

if len(conflicting_components) != 0:

    raise RuntimeError(
        "Component leakage detected."
    )


# ============================================================
# SEQUENCE INTEGRITY
# ============================================================

sequence_split_conflicts = (
    manifest
    .groupby("sequence")["split"]
    .nunique()
)

conflicting_sequences = (
    sequence_split_conflicts[
        sequence_split_conflicts > 1
    ]
)

print(
    "Sequences crossing splits:",
    len(conflicting_sequences)
)

if len(conflicting_sequences) != 0:

    raise RuntimeError(
        "Sequence leakage detected."
    )


# ============================================================
# OPTIONAL SAMPLE LIMITS
# ============================================================

train_df = manifest[
    manifest["split"] == "train"
].copy()

val_df = manifest[
    manifest["split"] == "val"
].copy()

test_df = manifest[
    manifest["split"] == "test"
].copy()

if MAX_TRAIN_SAMPLES is not None:

    train_df = train_df.head(
        MAX_TRAIN_SAMPLES
    )

if MAX_VAL_SAMPLES is not None:

    val_df = val_df.head(
        MAX_VAL_SAMPLES
    )


print()
print(
    f"TRAIN records: {len(train_df)}"
)

print(
    f"VAL records: {len(val_df)}"
)

print(
    f"TEST records: {len(test_df)}"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = BeltCrackSegmentationDataset(
    train_df,
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = BeltCrackSegmentationDataset(
    val_df,
    image_size=IMAGE_SIZE,
    augment=False
)

test_dataset = BeltCrackSegmentationDataset(
    test_df,
    image_size=IMAGE_SIZE,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

pin_memory = (
    DEVICE.type == "cuda"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 80)
print("BUILDING BASELINE U-NET")
print("=" * 80)

model = UNet().to(
    DEVICE
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    f"Total parameters: {parameter_count:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameter_count:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 80)
print("TRAINING")
print("=" * 80)

history = []

best_val_dice = -1.0
best_epoch = -1
epochs_without_improvement = 0

best_model_path = (
    MODEL_ROOT
    / "beltcrack9kd_unet_baseline_best.pt"
)

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print()
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    train_metrics = run_epoch(
        model,
        train_loader,
        optimizer
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        optimizer=None
    )

    print(
        f"Train Loss: "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"Train Dice: "
        f"{train_metrics['dice']:.6f}"
    )

    print(
        f"Train IoU: "
        f"{train_metrics['iou']:.6f}"
    )

    print(
        f"Train Precision: "
        f"{train_metrics['precision']:.6f}"
    )

    print(
        f"Train Recall: "
        f"{train_metrics['recall']:.6f}"
    )

    print(
        f"Val Loss: "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Val Dice: "
        f"{val_metrics['dice']:.6f}"
    )

    print(
        f"Val IoU: "
        f"{val_metrics['iou']:.6f}"
    )

    print(
        f"Val Precision: "
        f"{val_metrics['precision']:.6f}"
    )

    print(
        f"Val Recall: "
        f"{val_metrics['recall']:.6f}"
    )

    history_record = {
        "epoch": epoch,

        "train_loss":
            train_metrics["loss"],

        "train_dice":
            train_metrics["dice"],

        "train_iou":
            train_metrics["iou"],

        "train_precision":
            train_metrics["precision"],

        "train_recall":
            train_metrics["recall"],

        "val_loss":
            val_metrics["loss"],

        "val_dice":
            val_metrics["dice"],

        "val_iou":
            val_metrics["iou"],

        "val_precision":
            val_metrics["precision"],

        "val_recall":
            val_metrics["recall"],
    }

    history.append(
        history_record
    )

    # --------------------------------------------------------
    # VALIDATION-LOCKED MODEL SELECTION
    # --------------------------------------------------------

    if (
        val_metrics["dice"]
        >
        best_val_dice
    ):

        best_val_dice = (
            val_metrics["dice"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_type":
                    "UNet",

                "stage":
                    "4.6",

                "project":
                    "BeltCrack9kd_Damage_Intelligence",

                "image_size":
                    IMAGE_SIZE,

                "feature_type":
                    "binary_crack_segmentation",

                "threshold":
                    THRESHOLD,

                "seed":
                    SEED,

                "parameters":
                    parameter_count,
            },
            best_model_path
        )

        print(
            "Best validation Dice so far."
        )

        print(
            f"Saved checkpoint: "
            f"{best_model_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No validation Dice improvement "
            f"for {epochs_without_improvement} "
            f"epoch(s)."
        )

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    OUTPUT_ROOT
    / "stage_4_6_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("=" * 80)
print("LOADING BEST VALIDATION CHECKPOINT")
print("=" * 80)

if not best_model_path.exists():

    raise RuntimeError(
        "Best model checkpoint was not created."
    )

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(
    DEVICE
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation Dice: "
    f"{best_val_dice:.6f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL VALIDATION EVALUATION")
print("=" * 80)

final_val_metrics = run_epoch(
    model,
    val_loader,
    optimizer=None
)

print(
    f"Validation Dice: "
    f"{final_val_metrics['dice']:.6f}"
)

print(
    f"Validation IoU: "
    f"{final_val_metrics['iou']:.6f}"
)

print(
    f"Validation Precision: "
    f"{final_val_metrics['precision']:.6f}"
)

print(
    f"Validation Recall: "
    f"{final_val_metrics['recall']:.6f}"
)


# ============================================================
# OFFICIAL TEST
# ============================================================

print()
print("=" * 80)
print("OFFICIAL HELD-OUT TEST EVALUATION")
print("=" * 80)

print(
    "IMPORTANT:"
)

print(
    "The TEST set was not used for model selection."
)

print(
    "It is evaluated only after the best validation "
    "checkpoint has been selected."
)

final_test_metrics = run_epoch(
    model,
    test_loader,
    optimizer=None
)

print(
    f"Test Dice: "
    f"{final_test_metrics['dice']:.6f}"
)

print(
    f"Test IoU: "
    f"{final_test_metrics['iou']:.6f}"
)

print(
    f"Test Precision: "
    f"{final_test_metrics['precision']:.6f}"
)

print(
    f"Test Recall: "
    f"{final_test_metrics['recall']:.6f}"
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "split": "validation",
            **final_val_metrics
        },
        {
            "split": "test",
            **final_test_metrics
        },
    ]
)

results_path = (
    OUTPUT_ROOT
    / "stage_4_6_segmentation_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "stage":
        "4.6",

    "status":
        "PASS",

    "project":
        "BeltCrack9kd_Damage_Intelligence",

    "model":
        "UNet",

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        NUM_EPOCHS,

    "epochs_completed":
        len(history),

    "best_epoch":
        best_epoch,

    "best_validation_dice":
        best_val_dice,

    "threshold":
        THRESHOLD,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "seed":
        SEED,

    "parameter_count":
        parameter_count,

    "train_records":
        len(train_df),

    "validation_records":
        len(val_df),

    "test_records":
        len(test_df),

    "component_split_conflicts":
        int(len(conflicting_components)),

    "sequence_split_conflicts":
        int(len(conflicting_sequences)),

    "missing_images":
        int(missing_images),

    "missing_masks":
        int(missing_masks),

    "validation_metrics":
        final_val_metrics,

    "test_metrics":
        final_test_metrics,

    "best_model_path":
        str(best_model_path),

    "history_path":
        str(history_path),

    "results_path":
        str(results_path),

    "test_policy":
        "Held-out test evaluated only after validation-locked model selection.",
}

summary_path = (
    OUTPUT_ROOT
    / "stage_4_6_segmentation_baseline_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_safe(summary),
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STAGE 4.6 STATUS: PASS")
print("=" * 80)

print()
print(
    f"Best validation Dice: "
    f"{best_val_dice:.6f}"
)

print(
    f"Test Dice: "
    f"{final_test_metrics['dice']:.6f}"
)

print(
    f"Test IoU: "
    f"{final_test_metrics['iou']:.6f}"
)

print(
    f"Test Precision: "
    f"{final_test_metrics['precision']:.6f}"
)

print(
    f"Test Recall: "
    f"{final_test_metrics['recall']:.6f}"
)

print()
print(
    f"Best model: {best_model_path}"
)

print(
    f"Training history: {history_path}"
)

print(
    f"Results: {results_path}"
)

print(
    f"Summary: {summary_path}"
)

print()
print("=" * 80)
print("STAGE 4.6 COMPLETE")
print("=" * 80)