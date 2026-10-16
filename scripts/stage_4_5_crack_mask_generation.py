import os
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
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

AUDIT_ROOT = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_5_crack_masks"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "canonical_dataset"
    / "masks"
)

ANNOTATION_AUDIT_PATH = (
    PROJECT_ROOT
    / "audit_outputs"
    / "stage_4_4_crack_detection_audit"
    / "stage_4_4_crack_annotation_records.csv"
)

MASK_ROOT.mkdir(parents=True, exist_ok=True)
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def extract_frame_number(filename):
    """
    Extract frame number from canonical/original numeric filename.

    Examples:
        5.jpg -> 5
        0121.jpg -> 121
    """
    stem = Path(str(filename)).stem

    if stem.isdigit():
        return int(stem)

    # Fallback: find trailing digits
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        else:
            break

    if digits:
        return int(digits)

    return None


def resolve_json_path(row):
    """
    Resolve the actual LabelMe JSON.

    Primary source:
        source_json from canonical manifest

    Fallback based on the verified dataset structure:
        BeltCrack9kd1/json/<split>/<data-folder>/<original_json_filename>
    """

    source_json = str(row.get("source_json", "")).strip()

    if source_json:
        candidate = Path(source_json)

        if candidate.exists():
            return candidate

        # Try relative to dataset root
        candidate = DATASET_ROOT / source_json

        if candidate.exists():
            return candidate

    split = str(row["split"]).strip()
    original_json = str(row["original_json_filename"]).strip()

    sequence = str(row["sequence"]).strip()

    if "/" in sequence:
        sequence_parts = sequence.split("/")
        data_folder = sequence_parts[-1]
    else:
        data_folder = sequence

    fallback = (
        DATASET_ROOT
        / "BeltCrack9kd1"
        / "json"
        / split
        / data_folder
        / original_json
    )

    if fallback.exists():
        return fallback

    return None


def resolve_image_path(row):
    """
    Resolve actual source image.
    """

    source_image = str(row.get("source_image", "")).strip()

    if source_image:
        candidate = Path(source_image)

        if candidate.exists():
            return candidate

        candidate = DATASET_ROOT / source_image

        if candidate.exists():
            return candidate

    split = str(row["split"]).strip()
    original_image = str(row["original_image_filename"]).strip()
    sequence = str(row["sequence"]).strip()

    if "/" in sequence:
        data_folder = sequence.split("/")[-1]
    else:
        data_folder = sequence

    fallback = (
        DATASET_ROOT
        / "BeltCrack9kd1"
        / "images"
        / split
        / data_folder
        / original_image
    )

    if fallback.exists():
        return fallback

    return None


def make_mask(width, height, polygons):
    """
    Rasterize crack polygons into a binary uint8 mask.

    Background = 0
    Crack = 255

    Out-of-bounds points are clipped to image boundaries.
    """

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    valid_polygon_count = 0
    clipped_polygon_count = 0
    clipped_point_count = 0
    total_point_count = 0

    for polygon in polygons:

        points = polygon["points"]

        cleaned = []

        polygon_was_clipped = False

        for point in points:

            if not isinstance(point, (list, tuple)):
                continue

            if len(point) < 2:
                continue

            x = safe_float(point[0])
            y = safe_float(point[1])

            if x is None or y is None:
                continue

            total_point_count += 1

            original_x = x
            original_y = y

            clipped_x = min(max(x, 0.0), width - 1.0)
            clipped_y = min(max(y, 0.0), height - 1.0)

            if (
                abs(clipped_x - original_x) > 1e-9
                or abs(clipped_y - original_y) > 1e-9
            ):
                clipped_point_count += 1
                polygon_was_clipped = True

            cleaned.append(
                (
                    int(round(clipped_x)),
                    int(round(clipped_y))
                )
            )

        # Need at least 3 unique points for a polygon
        unique_points = list(dict.fromkeys(cleaned))

        if len(unique_points) < 3:
            continue

        if polygon_was_clipped:
            clipped_polygon_count += 1

        draw.polygon(unique_points, fill=255)

        valid_polygon_count += 1

    return (
        mask,
        valid_polygon_count,
        clipped_polygon_count,
        clipped_point_count,
        total_point_count,
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STAGE 4.5 - CRACK MASK GENERATION & INTEGRITY AUDIT")
print("=" * 80)

print()
print("Purpose:")
print("Convert verified LabelMe crack polygons into binary crack masks.")
print("No ML model will be trained in this stage.")
print("Original images will NOT be duplicated.")
print()

# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)

required_inputs = [
    MANIFEST_PATH,
    DATASET_ROOT,
]

for path in required_inputs:

    print(f"{path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist: {path}"
        )

print("Input checks: PASS")


# ============================================================
# LOAD MANIFEST
# ============================================================

print()
print("=" * 80)
print("LOADING CANONICAL MANIFEST")
print("=" * 80)

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "split",
    "component",
    "sequence",
    "original_image_filename",
    "canonical_image_filename",
    "canonical_image_path",
    "original_json_filename",
    "canonical_json_filename",
    "source_image",
    "source_json",
]

missing_columns = [
    col for col in required_columns
    if col not in manifest.columns
]

if missing_columns:
    raise ValueError(
        f"Manifest schema mismatch. Missing columns: {missing_columns}"
    )

print(f"Manifest records: {len(manifest)}")
print(f"Manifest columns: {len(manifest.columns)}")
print("Manifest schema check: PASS")


# ============================================================
# OPTIONAL STAGE 4.4 CROSS-CHECK
# ============================================================

if ANNOTATION_AUDIT_PATH.exists():

    try:
        annotation_audit = pd.read_csv(
            ANNOTATION_AUDIT_PATH
        )

        print()
        print(
            f"Stage 4.4 annotation audit records: "
            f"{len(annotation_audit)}"
        )

        if len(annotation_audit) != len(manifest):
            print(
                "WARNING: Stage 4.4 record count differs "
                "from canonical manifest."
            )
        else:
            print(
                "Stage 4.4 annotation audit count: PASS"
            )

    except Exception as exc:

        print(
            f"WARNING: Could not load Stage 4.4 audit: {exc}"
        )


# ============================================================
# DUPLICATE MANIFEST KEY CHECK
# ============================================================

print()
print("=" * 80)
print("CHECKING MANIFEST UNIQUENESS")
print("=" * 80)

manifest["frame_number"] = manifest[
    "canonical_image_filename"
].apply(extract_frame_number)

duplicate_keys = manifest.duplicated(
    subset=["sequence", "frame_number"],
    keep=False
)

duplicate_count = int(duplicate_keys.sum())

print(
    f"Duplicate sequence/frame records: "
    f"{duplicate_count}"
)

if duplicate_count == 0:
    print("Manifest key uniqueness: PASS")
else:
    print(
        "WARNING: Duplicate sequence/frame records detected."
    )


# ============================================================
# MASK GENERATION
# ============================================================

print()
print("=" * 80)
print("GENERATING BINARY CRACK MASKS")
print("=" * 80)

records = []

total = len(manifest)

success_count = 0
json_missing = 0
json_invalid = 0
json_read_errors = 0
image_missing = 0
dimension_errors = 0
mask_errors = 0

total_shapes = 0
total_crack_shapes = 0
total_valid_polygons = 0

images_with_mask_pixels = 0
images_without_mask_pixels = 0

total_mask_pixels = 0

out_of_bounds_points = 0
out_of_bounds_polygons = 0

for idx, row in manifest.iterrows():

    if idx == 0 or (idx + 1) % 500 == 0 or idx == total - 1:
        print(
            f"Processed: {idx + 1} / {total}"
        )

    sequence = str(row["sequence"])
    split = str(row["split"])
    component = safe_int(row["component"])

    frame_number = extract_frame_number(
        row["canonical_image_filename"]
    )

    json_path = resolve_json_path(row)
    image_path = resolve_image_path(row)

    result = {
        "split": split,
        "component": component,
        "sequence": sequence,
        "frame_number": frame_number,
        "original_image_filename":
            str(row["original_image_filename"]),
        "original_json_filename":
            str(row["original_json_filename"]),
        "source_image":
            str(image_path) if image_path else "",
        "source_json":
            str(json_path) if json_path else "",
        "mask_path": "",
        "image_width_px": None,
        "image_height_px": None,
        "total_shapes": 0,
        "crack_shapes": 0,
        "valid_polygons": 0,
        "mask_foreground_pixels": 0,
        "mask_foreground_ratio": 0.0,
        "mask_nonempty": False,
        "clipped_polygons": 0,
        "clipped_points": 0,
        "status": "FAILED",
        "failure_reason": "",
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if json_path is None:

        json_missing += 1

        result["failure_reason"] = "JSON_MISSING"

        records.append(result)

        continue

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except json.JSONDecodeError:

        json_invalid += 1

        result["failure_reason"] = "JSON_INVALID"

        records.append(result)

        continue

    except Exception as exc:

        json_read_errors += 1

        result["failure_reason"] = (
            f"JSON_READ_ERROR:{type(exc).__name__}"
        )

        records.append(result)

        continue

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------

    width = safe_int(data.get("imageWidth"))
    height = safe_int(data.get("imageHeight"))

    if (
        width is None
        or height is None
        or width <= 0
        or height <= 0
    ):

        dimension_errors += 1

        result["failure_reason"] = "INVALID_DIMENSIONS"

        records.append(result)

        continue

    result["image_width_px"] = width
    result["image_height_px"] = height

    # --------------------------------------------------------
    # IMAGE EXISTENCE
    # --------------------------------------------------------

    if image_path is None:

        image_missing += 1

        result["failure_reason"] = "IMAGE_MISSING"

        records.append(result)

        continue

    # --------------------------------------------------------
    # SHAPES
    # --------------------------------------------------------

    shapes = data.get("shapes", [])

    if not isinstance(shapes, list):
        shapes = []

    result["total_shapes"] = len(shapes)

    total_shapes += len(shapes)

    crack_polygons = []

    for shape in shapes:

        if not isinstance(shape, dict):
            continue

        label = str(
            shape.get("label", "")
        ).strip().lower()

        if label != "crack":
            continue

        total_crack_shapes += 1

        points = shape.get("points", [])

        if not isinstance(points, list):
            continue

        crack_polygons.append(
            {
                "points": points
            }
        )

    result["crack_shapes"] = len(crack_polygons)

    # --------------------------------------------------------
    # CREATE MASK
    # --------------------------------------------------------

    try:

        (
            mask,
            valid_polygon_count,
            clipped_polygon_count,
            clipped_point_count,
            total_point_count,
        ) = make_mask(
            width,
            height,
            crack_polygons
        )

    except Exception as exc:

        mask_errors += 1

        result["failure_reason"] = (
            f"MASK_GENERATION_ERROR:{type(exc).__name__}"
        )

        records.append(result)

        continue

    result["valid_polygons"] = valid_polygon_count
    result["clipped_polygons"] = clipped_polygon_count
    result["clipped_points"] = clipped_point_count

    total_valid_polygons += valid_polygon_count

    out_of_bounds_points += clipped_point_count
    out_of_bounds_polygons += clipped_polygon_count

    # --------------------------------------------------------
    # MASK PIXELS
    # --------------------------------------------------------

    mask_array = np.asarray(mask)

    foreground_pixels = int(
        np.count_nonzero(mask_array)
    )

    mask_area = width * height

    foreground_ratio = (
        foreground_pixels / mask_area
        if mask_area > 0
        else 0.0
    )

    result["mask_foreground_pixels"] = foreground_pixels
    result["mask_foreground_ratio"] = foreground_ratio
    result["mask_nonempty"] = foreground_pixels > 0

    total_mask_pixels += foreground_pixels

    if foreground_pixels > 0:
        images_with_mask_pixels += 1
    else:
        images_without_mask_pixels += 1

    # --------------------------------------------------------
    # OUTPUT MASK PATH
    # --------------------------------------------------------

    data_folder = sequence.split("/")[-1]

    mask_directory = (
        MASK_ROOT
        / split
        / data_folder
    )

    mask_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    if frame_number is not None:
        mask_filename = f"{frame_number}.png"
    else:
        mask_filename = (
            Path(
                str(row["canonical_image_filename"])
            ).stem
            + ".png"
        )

    mask_path = (
        mask_directory
        / mask_filename
    )

    try:

        # Lossless binary mask
        mask.save(
            mask_path,
            format="PNG",
            optimize=True
        )

    except Exception as exc:

        mask_errors += 1

        result["failure_reason"] = (
            f"MASK_SAVE_ERROR:{type(exc).__name__}"
        )

        records.append(result)

        continue

    # --------------------------------------------------------
    # VERIFY SAVED MASK
    # --------------------------------------------------------

    try:

        with Image.open(mask_path) as verify_image:

            verify_image.load()

            verify_width, verify_height = (
                verify_image.size
            )

            verify_array = np.asarray(
                verify_image
            )

        if (
            verify_width != width
            or verify_height != height
        ):

            mask_errors += 1

            result["failure_reason"] = (
                "MASK_DIMENSION_MISMATCH"
            )

            records.append(result)

            continue

        verify_foreground = int(
            np.count_nonzero(verify_array)
        )

        if verify_foreground != foreground_pixels:

            mask_errors += 1

            result["failure_reason"] = (
                "MASK_FOREGROUND_MISMATCH"
            )

            records.append(result)

            continue

    except Exception as exc:

        mask_errors += 1

        result["failure_reason"] = (
            f"MASK_VERIFY_ERROR:{type(exc).__name__}"
        )

        records.append(result)

        continue

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    result["mask_path"] = str(mask_path)
    result["status"] = "PASS"
    result["failure_reason"] = ""

    success_count += 1

    records.append(result)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(records)

records_csv = (
    AUDIT_ROOT
    / "stage_4_5_mask_generation_records.csv"
)

results.to_csv(
    records_csv,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print()
print("=" * 80)
print("MASK GENERATION RESULTS")
print("=" * 80)

print(
    f"Manifest records: {len(manifest)}"
)

print(
    f"Successfully generated masks: "
    f"{success_count}"
)

print(
    f"JSON missing: {json_missing}"
)

print(
    f"JSON invalid: {json_invalid}"
)

print(
    f"JSON read errors: {json_read_errors}"
)

print(
    f"Image missing: {image_missing}"
)

print(
    f"Dimension errors: {dimension_errors}"
)

print(
    f"Mask errors: {mask_errors}"
)

print(
    f"Total LabelMe shapes: {total_shapes}"
)

print(
    f"Total crack polygons: {total_crack_shapes}"
)

print(
    f"Total valid polygons rasterized: "
    f"{total_valid_polygons}"
)

print(
    f"Images with non-empty masks: "
    f"{images_with_mask_pixels}"
)

print(
    f"Images with empty masks: "
    f"{images_without_mask_pixels}"
)

print(
    f"Clipped out-of-bounds points: "
    f"{out_of_bounds_points}"
)

print(
    f"Polygons containing clipped points: "
    f"{out_of_bounds_polygons}"
)


# ============================================================
# MASK DIMENSION AUDIT
# ============================================================

print()
print("=" * 80)
print("MASK DIMENSION AUDIT")
print("=" * 80)

successful = results[
    results["status"] == "PASS"
].copy()

dimension_match_count = 0
dimension_mismatch_count = 0
mask_read_error_count = 0

for _, row in successful.iterrows():

    mask_path = Path(row["mask_path"])

    try:

        with Image.open(mask_path) as img:

            width, height = img.size

        if (
            width == int(row["image_width_px"])
            and height == int(row["image_height_px"])
        ):
            dimension_match_count += 1
        else:
            dimension_mismatch_count += 1

    except Exception:

        mask_read_error_count += 1


print(
    f"Successful masks checked: {len(successful)}"
)

print(
    f"Dimension matches: {dimension_match_count}"
)

print(
    f"Dimension mismatches: {dimension_mismatch_count}"
)

print(
    f"Mask read errors: {mask_read_error_count}"
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

print()
print("=" * 80)
print("SPLIT MASK SUMMARY")
print("=" * 80)

split_summary = (
    results
    .groupby("split")
    .agg(
        records=("sequence", "size"),
        masks_generated=("status", lambda x: int(
            (x == "PASS").sum()
        )),
        mask_failures=("status", lambda x: int(
            (x != "PASS").sum()
        )),
        mean_foreground_ratio=(
            "mask_foreground_ratio",
            "mean"
        ),
        median_foreground_ratio=(
            "mask_foreground_ratio",
            "median"
        ),
        max_foreground_ratio=(
            "mask_foreground_ratio",
            "max"
        ),
    )
    .reset_index()
)

print(
    split_summary.to_string(index=False)
)

split_summary_path = (
    AUDIT_ROOT
    / "stage_4_5_split_mask_summary.csv"
)

split_summary.to_csv(
    split_summary_path,
    index=False
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

print()
print("=" * 80)
print("SEQUENCE MASK SUMMARY")
print("=" * 80)

sequence_summary = (
    results
    .groupby("sequence")
    .agg(
        records=("sequence", "size"),
        masks_generated=("status", lambda x: int(
            (x == "PASS").sum()
        )),
        mask_failures=("status", lambda x: int(
            (x != "PASS").sum()
        )),
        mean_foreground_ratio=(
            "mask_foreground_ratio",
            "mean"
        ),
        max_foreground_ratio=(
            "mask_foreground_ratio",
            "max"
        ),
        total_foreground_pixels=(
            "mask_foreground_pixels",
            "sum"
        ),
    )
    .reset_index()
)

print(
    sequence_summary.to_string(index=False)
)

sequence_summary_path = (
    AUDIT_ROOT
    / "stage_4_5_sequence_mask_summary.csv"
)

sequence_summary.to_csv(
    sequence_summary_path,
    index=False
)


# ============================================================
# DATASET-LEVEL INTEGRITY CHECKS
# ============================================================

print()
print("=" * 80)
print("MASK DATASET INTEGRITY CHECKS")
print("=" * 80)

manifest_records_present = (
    len(manifest) > 0
)

all_records_have_results = (
    len(results) == len(manifest)
)

all_masks_generated = (
    success_count == len(manifest)
)

no_json_missing = (
    json_missing == 0
)

no_json_invalid = (
    json_invalid == 0
)

no_json_read_errors = (
    json_read_errors == 0
)

no_dimension_errors = (
    dimension_errors == 0
)

no_mask_errors = (
    mask_errors == 0
)

all_images_have_cracks = (
    images_without_mask_pixels == 0
)

all_dimensions_match = (
    dimension_match_count == success_count
    and dimension_mismatch_count == 0
    and mask_read_error_count == 0
)

all_polygons_rasterized = (
    total_valid_polygons == total_crack_shapes
)

no_duplicate_sequence_frames = (
    duplicate_count == 0
)

checks = {
    "manifest_records_present":
        manifest_records_present,

    "all_records_have_results":
        all_records_have_results,

    "all_masks_generated":
        all_masks_generated,

    "json_missing_zero":
        no_json_missing,

    "json_invalid_zero":
        no_json_invalid,

    "json_read_errors_zero":
        no_json_read_errors,

    "dimension_errors_zero":
        no_dimension_errors,

    "mask_errors_zero":
        no_mask_errors,

    "all_images_have_nonempty_masks":
        all_images_have_cracks,

    "all_mask_dimensions_match":
        all_dimensions_match,

    "all_crack_polygons_rasterized":
        all_polygons_rasterized,

    "duplicate_sequence_frame_keys_zero":
        no_duplicate_sequence_frames,
}

for name, value in checks.items():

    print(
        f"{name}: "
        f"{'PASS' if value else 'FAIL'}"
    )


# ============================================================
# DISK USAGE
# ============================================================

print()
print("=" * 80)
print("MASK STORAGE")
print("=" * 80)

mask_file_count = 0
mask_total_bytes = 0

for path in MASK_ROOT.rglob("*.png"):

    try:
        mask_file_count += 1
        mask_total_bytes += path.stat().st_size
    except OSError:
        pass

mask_total_mb = (
    mask_total_bytes / (1024 ** 2)
)

mask_total_gb = (
    mask_total_bytes / (1024 ** 3)
)

print(
    f"Mask files: {mask_file_count}"
)

print(
    f"Mask storage: "
    f"{mask_total_mb:.2f} MB "
    f"({mask_total_gb:.3f} GB)"
)


# ============================================================
# FINAL DECISION
# ============================================================

core_checks = [
    manifest_records_present,
    all_records_have_results,
    all_masks_generated,
    no_json_missing,
    no_json_invalid,
    no_json_read_errors,
    no_dimension_errors,
    no_mask_errors,
    all_images_have_cracks,
    all_dimensions_match,
    all_polygons_rasterized,
    no_duplicate_sequence_frames,
]

if all(core_checks):

    decision = (
        "MASK_DATASET_READY_FOR_SEGMENTATION_TRAINING"
    )

else:

    decision = (
        "MASK_DATASET_REQUIRES_REVIEW"
    )


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "stage": "4.5",
    "status": "PASS" if all(core_checks) else "REVIEW",

    "decision": decision,

    "manifest_records": int(len(manifest)),
    "mask_files": int(mask_file_count),

    "successful_masks": int(success_count),
    "json_missing": int(json_missing),
    "json_invalid": int(json_invalid),
    "json_read_errors": int(json_read_errors),
    "image_missing": int(image_missing),
    "dimension_errors": int(dimension_errors),
    "mask_errors": int(mask_errors),

    "total_shapes": int(total_shapes),
    "total_crack_shapes": int(total_crack_shapes),
    "total_valid_polygons": int(total_valid_polygons),

    "images_with_nonempty_masks":
        int(images_with_mask_pixels),

    "images_without_nonempty_masks":
        int(images_without_mask_pixels),

    "out_of_bounds_points_clipped":
        int(out_of_bounds_points),

    "out_of_bounds_polygons_clipped":
        int(out_of_bounds_polygons),

    "dimension_matches":
        int(dimension_match_count),

    "dimension_mismatches":
        int(dimension_mismatch_count),

    "mask_read_errors":
        int(mask_read_error_count),

    "mask_storage_bytes":
        int(mask_total_bytes),

    "mask_storage_mb":
        float(mask_total_mb),

    "mask_storage_gb":
        float(mask_total_gb),

    "checks": checks,

    "output_mask_root": str(MASK_ROOT),
    "records_csv": str(records_csv),
    "split_summary_csv": str(split_summary_path),
    "sequence_summary_csv": str(sequence_summary_path),
}

summary_path = (
    AUDIT_ROOT
    / "stage_4_5_mask_generation_summary.json"
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
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print(f"STAGE 4.5 STATUS: {'PASS' if all(core_checks) else 'REVIEW'}")
print("=" * 80)

print()
print(f"Mask root: {MASK_ROOT}")
print(f"Records: {records_csv}")
print(f"Split summary: {split_summary_path}")
print(f"Sequence summary: {sequence_summary_path}")
print(f"Audit summary: {summary_path}")

print()
print(f"Decision: {decision}")
print()
print("=" * 80)
print("STAGE 4.5 COMPLETE")
print("=" * 80)