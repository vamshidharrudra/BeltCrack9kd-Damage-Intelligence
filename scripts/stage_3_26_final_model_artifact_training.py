# ============================================================
# STAGE 3.26 - FINAL MODEL ARTIFACT TRAINING
# BeltCrack9kd_Damage_Intelligence
#
# FINAL CORRECTED VERSION
#
# Purpose:
#   Train the final selected 33-feature physical damage
#   regression model using TRAIN + VAL only.
#
# IMPORTANT:
#   - TEST is NEVER used for training.
#   - No imputation.
#   - Rows missing any selected feature are excluded.
#   - Complete-feature policy exactly matches Stage 3.24.
#
# AUTHORITATIVE SOURCES:
#
#   Target:
#       stage_3_15_physical_regression_targets.csv
#
#   Temporal:
#       stage_3_17_temporal_physical_features.csv
#
#   Spatial:
#       stage_3_1_spatial_track_features.csv
#
#   Final 33 features:
#       stage_3_24_exact_final_feature_set.csv
#
#   Component assignment:
#       stage_0_9d1_component_assignments.csv
#
#   Per-frame component mapping:
#       canonical_dataset/canonical_manifest.csv
#
# Expected:
#
#   Complete records = 12149
#   TRAIN = 8573
#   VAL   = 1716
#   TEST  = 1860
#
#   Final artifact training:
#   TRAIN + VAL = 10289
#   TEST used   = 0
# ============================================================

from pathlib import Path
import json
import pickle
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"
MODELS_DIR = PROJECT_ROOT / "models"
CANONICAL_DIR = PROJECT_ROOT / "canonical_dataset"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_15_physical_regression_targets.csv"
)

TEMPORAL_FILE = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_features.csv"
)

SPATIAL_FILE = (
    AUDIT_DIR /
    "stage_3_1_spatial_track_features.csv"
)

FINAL_FEATURE_FILE = (
    AUDIT_DIR /
    "stage_3_24_exact_final_feature_set.csv"
)

COMPONENT_ASSIGNMENT_FILE = (
    AUDIT_DIR /
    "stage_0_9d1_component_assignments.csv"
)

MANIFEST_FILE = (
    CANONICAL_DIR /
    "canonical_manifest.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = (
    MODELS_DIR /
    "beltcrack9kd_final_gbr_33_features.pkl"
)

FEATURE_FILE = (
    MODELS_DIR /
    "beltcrack9kd_final_33_feature_list.json"
)

METADATA_FILE = (
    MODELS_DIR /
    "beltcrack9kd_final_model_metadata.json"
)

TRAINING_SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_26_final_model_training_summary.json"
)

TRAINING_RECORDS_FILE = (
    AUDIT_DIR /
    "stage_3_26_final_training_records.csv"
)

IMPORTANCE_FILE = (
    AUDIT_DIR /
    "stage_3_26_final_feature_importance.csv"
)

FRAME_COMPONENT_MAP_FILE = (
    AUDIT_DIR /
    "stage_3_26_frame_component_mapping.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("STAGE 3.26 - FINAL MODEL ARTIFACT TRAINING")
print("=" * 78)

print()
print("Project:")
print(PROJECT_ROOT)

print()
print("FINAL TRAINING POLICY:")
print("  - TRAIN + VAL only")
print("  - TEST excluded")
print("  - No imputation")
print("  - Complete 33-feature rows only")
print("  - Same complete-feature policy as Stage 3.24")

print()
print("CORRECT SPLIT MAPPING:")
print("  Component assignments -> component_id -> split")
print("  Canonical manifest    -> sequence/frame -> component_id")


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    TARGET_FILE,
    TEMPORAL_FILE,
    SPATIAL_FILE,
    FINAL_FEATURE_FILE,
    COMPONENT_ASSIGNMENT_FILE,
    MANIFEST_FILE,
]

print()
print("=" * 78)
print("CHECKING REQUIRED FILES")
print("=" * 78)

for file_path in required_files:

    exists = file_path.exists()

    print(
        f"{file_path.name:<65}"
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:

        raise FileNotFoundError(
            "Required file not found:\n"
            f"{file_path}"
        )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 78)
print("LOADING DATA")
print("=" * 78)

target = pd.read_csv(
    TARGET_FILE
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

feature_set = pd.read_csv(
    FINAL_FEATURE_FILE
)

component_assignments = pd.read_csv(
    COMPONENT_ASSIGNMENT_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

print()
print(
    "Target records:",
    len(target)
)

print(
    "Temporal records:",
    len(temporal)
)

print(
    "Spatial records:",
    len(spatial)
)

print(
    "Feature-selection rows:",
    len(feature_set)
)

print(
    "Component assignment rows:",
    len(component_assignments)
)

print(
    "Canonical manifest records:",
    len(manifest)
)


# ============================================================
# DUPLICATE COLUMN CHECK
# ============================================================

def check_duplicate_columns(
    df,
    name
):

    duplicated = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated:

        raise ValueError(
            f"{name} has duplicate columns:\n"
            f"{duplicated}"
        )


check_duplicate_columns(
    target,
    "TARGET"
)

check_duplicate_columns(
    temporal,
    "TEMPORAL"
)

check_duplicate_columns(
    spatial,
    "SPATIAL"
)

check_duplicate_columns(
    component_assignments,
    "COMPONENT_ASSIGNMENTS"
)

check_duplicate_columns(
    manifest,
    "MANIFEST"
)


# ============================================================
# FEATURE COLUMN
# ============================================================

feature_column = None

for candidate in [
    "feature",
    "feature_name",
    "selected_feature",
    "final_feature",
    "column",
    "name",
]:

    if candidate in feature_set.columns:

        feature_column = candidate
        break


if feature_column is None:

    raise ValueError(
        "Could not identify Stage 3.24 "
        "feature column."
    )


# ============================================================
# EXACT 33 FEATURES
# ============================================================

final_features = (
    feature_set[
        feature_column
    ]
    .dropna()
    .astype(str)
    .tolist()
)

final_features = list(
    dict.fromkeys(
        final_features
    )
)

print()
print("=" * 78)
print("FINAL FEATURE SET")
print("=" * 78)

print(
    "Feature count:",
    len(final_features)
)

if len(final_features) != 33:

    raise ValueError(
        "Expected exactly 33 features."
    )

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = (
    "future_crack_area_ratio_t1"
)

if TARGET_COLUMN not in target.columns:

    raise ValueError(
        "Target column not found."
    )


# ============================================================
# KEY COLUMNS
# ============================================================

for name, df in [
    ("TARGET", target),
    ("TEMPORAL", temporal),
    ("SPATIAL", spatial),
]:

    for key in [
        "sequence",
        "current_frame",
    ]:

        if key not in df.columns:

            raise ValueError(
                f"{name} missing key: {key}"
            )


# ============================================================
# FEATURE SOURCE IDENTIFICATION
# ============================================================

temporal_features = []
spatial_features = []
missing_features = []

for feature in final_features:

    in_temporal = (
        feature in temporal.columns
    )

    in_spatial = (
        feature in spatial.columns
    )

    if in_temporal and not in_spatial:

        temporal_features.append(
            feature
        )

    elif in_spatial and not in_temporal:

        spatial_features.append(
            feature
        )

    elif in_temporal and in_spatial:

        # If a feature exists in both, use temporal
        # only after value consistency is checked.
        temporal_features.append(
            feature
        )

    else:

        missing_features.append(
            feature
        )


if missing_features:

    raise ValueError(
        "Missing final features:\n"
        +
        "\n".join(
            missing_features
        )
    )


print()
print(
    "Temporal features:",
    len(temporal_features)
)

print(
    "Spatial features:",
    len(spatial_features)
)

print(
    "Total:",
    len(
        set(
            temporal_features
            +
            spatial_features
        )
    )
)


# ============================================================
# PREPARE TARGET
# ============================================================

target_use = target[
    [
        "sequence",
        "current_frame",
        TARGET_COLUMN,
    ]
].copy()

target_use[
    "sequence"
] = target_use[
    "sequence"
].astype(str)

target_use[
    "current_frame"
] = pd.to_numeric(
    target_use[
        "current_frame"
    ],
    errors="coerce"
)

target_use[
    TARGET_COLUMN
] = pd.to_numeric(
    target_use[
        TARGET_COLUMN
    ],
    errors="coerce"
)


# ============================================================
# PREPARE TEMPORAL
# ============================================================

temporal_columns = [
    "sequence",
    "current_frame",
]

for feature in final_features:

    if (
        feature in temporal.columns
        and feature not in temporal_columns
    ):

        temporal_columns.append(
            feature
        )


temporal_use = temporal[
    temporal_columns
].copy()

temporal_use[
    "sequence"
] = temporal_use[
    "sequence"
].astype(str)

temporal_use[
    "current_frame"
] = pd.to_numeric(
    temporal_use[
        "current_frame"
    ],
    errors="coerce"
)


# ============================================================
# PREPARE SPATIAL
# ============================================================

spatial_columns = [
    "sequence",
    "current_frame",
]

for feature in final_features:

    if (
        feature in spatial.columns
        and feature not in spatial_columns
    ):

        spatial_columns.append(
            feature
        )


spatial_use = spatial[
    spatial_columns
].copy()

spatial_use[
    "sequence"
] = spatial_use[
    "sequence"
].astype(str)

spatial_use[
    "current_frame"
] = pd.to_numeric(
    spatial_use[
        "current_frame"
    ],
    errors="coerce"
)


# ============================================================
# DUPLICATE KEY CHECK
# ============================================================

for name, df in [
    ("TARGET", target_use),
    ("TEMPORAL", temporal_use),
    ("SPATIAL", spatial_use),
]:

    duplicates = int(
        df.duplicated(
            subset=[
                "sequence",
                "current_frame",
            ]
        ).sum()
    )

    print(
        f"{name} duplicate keys:",
        duplicates
    )

    if duplicates != 0:

        raise ValueError(
            f"{name} contains duplicate keys."
        )


# ============================================================
# MERGE TARGET + TEMPORAL + SPATIAL
# ============================================================

print()
print("=" * 78)
print("BUILDING COMPLETE FEATURE MATRIX")
print("=" * 78)

merged = target_use.merge(
    temporal_use,
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

merged = merged.merge(
    spatial_use,
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "Merged records:",
    len(merged)
)


# ============================================================
# TARGET CLEANING
# ============================================================

merged = merged.dropna(
    subset=[
        TARGET_COLUMN
    ]
).copy()


# ============================================================
# NUMERIC FEATURES
# ============================================================

for feature in final_features:

    merged[
        feature
    ] = pd.to_numeric(
        merged[
            feature
        ],
        errors="coerce"
    )


# ============================================================
# COMPLETE FEATURE POLICY
# ============================================================

print()
print("=" * 78)
print("APPLYING STAGE 3.24 COMPLETE-FEATURE POLICY")
print("=" * 78)

missing_before = (
    merged[
        final_features
    ]
    .isna()
    .sum()
)

missing_before = (
    missing_before[
        missing_before > 0
    ]
    .sort_values(
        ascending=False
    )
)

print()
print(
    "Features with missing values:",
    len(missing_before)
)

if len(missing_before) > 0:

    print(
        missing_before.to_string()
    )


complete_mask = (
    merged[
        final_features
    ]
    .notna()
    .all(
        axis=1
    )
)

complete_count = int(
    complete_mask.sum()
)

incomplete_count = (
    len(merged)
    -
    complete_count
)

print()
print(
    "Records before completeness filter:",
    len(merged)
)

print(
    "Complete records:",
    complete_count
)

print(
    "Incomplete records excluded:",
    incomplete_count
)


# ============================================================
# FILTER
# ============================================================

merged = merged[
    complete_mask
].copy()

print()
print(
    "Records after completeness filter:",
    len(merged)
)


# ============================================================
# FINITE CHECK
# ============================================================

X_check = merged[
    final_features
].to_numpy(
    dtype=float
)

y_check = merged[
    TARGET_COLUMN
].to_numpy(
    dtype=float
)

finite_mask = (
    np.isfinite(
        X_check
    ).all(
        axis=1
    )
    &
    np.isfinite(
        y_check
    )
)

nonfinite_count = int(
    (~finite_mask).sum()
)

print(
    "Non-finite records:",
    nonfinite_count
)

if nonfinite_count != 0:

    merged = merged[
        finite_mask
    ].copy()


# ============================================================
# ============================================================
# CORRECT COMPONENT -> FRAME MAPPING
# ============================================================
# ============================================================

print()
print("=" * 78)
print("BUILDING PER-FRAME COMPONENT/SPLIT MAPPING")
print("=" * 78)


# ------------------------------------------------------------
# Check component assignment schema
# ------------------------------------------------------------

print()
print(
    "Component assignment columns:"
)

print(
    component_assignments.columns.tolist()
)


if "component_id" not in component_assignments.columns:

    raise ValueError(
        "component_id not found in "
        "component assignment file."
    )

if "assigned_split" not in component_assignments.columns:

    raise ValueError(
        "assigned_split not found in "
        "component assignment file."
    )


# ------------------------------------------------------------
# Build component -> split map
# ------------------------------------------------------------

component_assignments[
    "component_id"
] = pd.to_numeric(
    component_assignments[
        "component_id"
    ],
    errors="coerce"
)

component_assignments[
    "assigned_split"
] = (
    component_assignments[
        "assigned_split"
    ]
    .astype(str)
    .str.upper()
)


component_map = (
    component_assignments[
        [
            "component_id",
            "assigned_split",
        ]
    ]
    .dropna(
        subset=[
            "component_id"
        ]
    )
    .drop_duplicates(
        subset=[
            "component_id"
        ]
    )
)


print()
print(
    "Component -> split mappings:",
    len(component_map)
)

print(
    component_map.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Validate component IDs
# ------------------------------------------------------------

component_split_duplicates = (
    component_assignments
    .groupby(
        "component_id"
    )[
        "assigned_split"
    ]
    .nunique()
)

conflicting_components = (
    component_split_duplicates[
        component_split_duplicates > 1
    ]
)

if len(conflicting_components) > 0:

    raise ValueError(
        "Components assigned to multiple splits:\n"
        +
        "\n".join(
            str(x)
            for x in conflicting_components.index
        )
    )


# ------------------------------------------------------------
# Manifest schema
# ------------------------------------------------------------

print()
print(
    "Canonical manifest columns:"
)

print(
    manifest.columns.tolist()
)


required_manifest_columns = [
    "sequence",
    "component",
    "canonical_image_filename",
]

missing_manifest_columns = [
    c
    for c in required_manifest_columns
    if c not in manifest.columns
]

if missing_manifest_columns:

    raise ValueError(
        "Canonical manifest missing columns:\n"
        +
        "\n".join(
            missing_manifest_columns
        )
    )


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame_number(
    value
):

    if pd.isna(value):

        return np.nan

    text = str(value)

    stem = Path(
        text
    ).stem

    # First try exact numeric stem.
    if stem.isdigit():

        return int(stem)

    # Otherwise search trailing digits.
    match = re.search(
        r"(\d+)$",
        stem
    )

    if match:

        return int(
            match.group(1)
        )

    # Otherwise search any numeric sequence.
    match = re.search(
        r"(\d+)",
        stem
    )

    if match:

        return int(
            match.group(1)
        )

    return np.nan


# ============================================================
# BUILD MANIFEST FRAME TABLE
# ============================================================

manifest_use = manifest[
    [
        "sequence",
        "component",
        "canonical_image_filename",
    ]
].copy()

manifest_use[
    "sequence"
] = manifest_use[
    "sequence"
].astype(str)

manifest_use[
    "component"
] = pd.to_numeric(
    manifest_use[
        "component"
    ],
    errors="coerce"
)

manifest_use[
    "current_frame"
] = (
    manifest_use[
        "canonical_image_filename"
    ]
    .apply(
        extract_frame_number
    )
)

manifest_use[
    "current_frame"
] = pd.to_numeric(
    manifest_use[
        "current_frame"
    ],
    errors="coerce"
)


# ============================================================
# MANIFEST FRAME VALIDATION
# ============================================================

manifest_missing_frames = int(
    manifest_use[
        "current_frame"
    ].isna().sum()
)

print()
print(
    "Manifest records:",
    len(manifest_use)
)

print(
    "Manifest missing frame numbers:",
    manifest_missing_frames
)

if manifest_missing_frames != 0:

    raise ValueError(
        "Could not extract frame numbers "
        "from canonical manifest."
    )


# ============================================================
# MANIFEST DUPLICATE CHECK
# ============================================================

manifest_duplicate_keys = int(
    manifest_use.duplicated(
        subset=[
            "sequence",
            "current_frame",
        ]
    ).sum()
)

print(
    "Manifest duplicate sequence/frame keys:",
    manifest_duplicate_keys
)

if manifest_duplicate_keys != 0:

    raise ValueError(
        "Canonical manifest contains duplicate "
        "sequence/frame keys."
    )


# ============================================================
# JOIN COMPONENT ASSIGNMENT
# ============================================================

manifest_use = manifest_use.merge(
    component_map,
    left_on="component",
    right_on="component_id",
    how="left",
    validate="many_to_one",
)

manifest_use.drop(
    columns=[
        "component_id"
    ],
    inplace=True
)


# ============================================================
# CHECK MISSING COMPONENT SPLITS
# ============================================================

missing_manifest_splits = int(
    manifest_use[
        "assigned_split"
    ].isna().sum()
)

print(
    "Manifest records without split:",
    missing_manifest_splits
)

if missing_manifest_splits != 0:

    raise ValueError(
        "Some manifest components do not have "
        "a Stage 0.9D1 split assignment."
    )


# ============================================================
# SAVE FRAME COMPONENT MAP
# ============================================================

frame_map_output = manifest_use[
    [
        "sequence",
        "current_frame",
        "component",
        "assigned_split",
    ]
].copy()

frame_map_output.to_csv(
    FRAME_COMPONENT_MAP_FILE,
    index=False
)


# ============================================================
# MERGE FRAME SPLIT INTO COMPLETE MATRIX
# ============================================================

merged = merged.merge(
    frame_map_output,
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print()
print(
    "Complete records after frame-component mapping:",
    len(merged)
)


# ============================================================
# VERIFY EXPECTED COMPLETE MATRIX
# ============================================================

print()
print("=" * 78)
print("VERIFYING STAGE 3.24 COMPLETE MATRIX")
print("=" * 78)

print(
    "Expected complete records:",
    12149
)

print(
    "Actual complete records:",
    len(merged)
)

if len(merged) != 12149:

    raise ValueError(
        "Complete record count does not match "
        "Stage 3.24.\n"
        f"Expected: 12149\n"
        f"Actual: {len(merged)}"
    )


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print()
print("=" * 78)
print("VERIFYING SPLIT DISTRIBUTION")
print("=" * 78)

split_counts = (
    merged[
        "assigned_split"
    ]
    .value_counts()
)

print(
    split_counts.to_string()
)


expected_split_counts = {
    "TRAIN": 8573,
    "VAL": 1716,
    "TEST": 1860,
}

for split_name, expected_count in (
    expected_split_counts.items()
):

    actual_count = int(
        split_counts.get(
            split_name,
            0
        )
    )

    print(
        f"{split_name}: "
        f"expected={expected_count}, "
        f"actual={actual_count}"
    )

    if actual_count != expected_count:

        raise ValueError(
            "Stage 3.24 split mismatch for "
            f"{split_name}."
        )


# ============================================================
# SPLIT CONFLICT CHECK
# ============================================================

component_split_check = (
    merged[
        [
            "component",
            "assigned_split",
        ]
    ]
    .drop_duplicates()
)

component_conflict_counts = (
    component_split_check
    .groupby(
        "component"
    )[
        "assigned_split"
    ]
    .nunique()
)

component_conflicts = (
    component_conflict_counts[
        component_conflict_counts > 1
    ]
)

print()
print(
    "Component split conflicts:",
    len(component_conflicts)
)

if len(component_conflicts) != 0:

    raise ValueError(
        "Component split conflicts detected."
    )


# ============================================================
# SEQUENCE SPLIT CHECK
# ============================================================

sequence_split_check = (
    merged[
        [
            "sequence",
            "assigned_split",
        ]
    ]
    .drop_duplicates()
)

sequence_conflict_counts = (
    sequence_split_check
    .groupby(
        "sequence"
    )[
        "assigned_split"
    ]
    .nunique()
)

sequence_conflicts = (
    sequence_conflict_counts[
        sequence_conflict_counts > 1
    ]
)

print(
    "Sequence split conflicts:",
    len(sequence_conflicts)
)

if len(sequence_conflicts) != 0:

    raise ValueError(
        "Sequence split conflicts detected."
    )


# ============================================================
# FINAL TRAIN + VAL
# ============================================================

trainval = merged[
    merged[
        "assigned_split"
    ].isin(
        [
            "TRAIN",
            "VAL",
        ]
    )
].copy()

train_count = int(
    (
        trainval[
            "assigned_split"
        ]
        == "TRAIN"
    ).sum()
)

val_count = int(
    (
        trainval[
            "assigned_split"
        ]
        == "VAL"
    ).sum()
)

test_count = int(
    (
        merged[
            "assigned_split"
        ]
        == "TEST"
    ).sum()
)


print()
print("=" * 78)
print("FINAL TRAINING DATA")
print("=" * 78)

print(
    "TRAIN:",
    train_count
)

print(
    "VAL:",
    val_count
)

print(
    "TRAIN + VAL:",
    len(trainval)
)

print(
    "TEST:",
    test_count
)

print(
    "TEST records used for training:",
    0
)


if train_count != 8573:

    raise ValueError(
        "Unexpected TRAIN count."
    )

if val_count != 1716:

    raise ValueError(
        "Unexpected VAL count."
    )

if test_count != 1860:

    raise ValueError(
        "Unexpected TEST count."
    )


# ============================================================
# SEQUENCES / COMPONENTS
# ============================================================

trainval_sequences = (
    trainval[
        "sequence"
    ]
    .nunique()
)

trainval_components = (
    trainval[
        "component"
    ]
    .nunique()
)

print()
print(
    "TRAIN+VAL sequences:",
    trainval_sequences
)

print(
    "TRAIN+VAL components:",
    trainval_components
)


# ============================================================
# FINAL MATRIX
# ============================================================

X_trainval = trainval[
    final_features
].to_numpy(
    dtype=float
)

y_trainval = trainval[
    TARGET_COLUMN
].to_numpy(
    dtype=float
)


print()
print(
    "Final X shape:",
    X_trainval.shape
)

print(
    "Final y shape:",
    y_trainval.shape
)


if X_trainval.shape != (
    10289,
    33,
):

    raise ValueError(
        "Unexpected final training matrix shape.\n"
        f"Expected: (10289, 33)\n"
        f"Actual: {X_trainval.shape}"
    )


# ============================================================
# FINAL MODEL PARAMETERS
# ============================================================

MODEL_PARAMETERS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "random_state": 42,
}


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 78)
print("TRAINING FINAL 33-FEATURE MODEL")
print("=" * 78)

print()
print(
    "Algorithm:",
    "GradientBoostingRegressor"
)

print(
    "Feature count:",
    33
)

print(
    "Training records:",
    len(trainval)
)

print(
    "TEST records used:",
    0
)

print()

for key, value in MODEL_PARAMETERS.items():

    print(
        f"{key}: {value}"
    )


model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

print()
print(
    "Fitting model..."
)

model.fit(
    X_trainval,
    y_trainval
)

print(
    "Training complete."
)


# ============================================================
# TRAINING METRICS
# ============================================================

trainval_prediction = model.predict(
    X_trainval
)

trainval_mae = mean_absolute_error(
    y_trainval,
    trainval_prediction
)

trainval_rmse = np.sqrt(
    mean_squared_error(
        y_trainval,
        trainval_prediction
    )
)

trainval_r2 = r2_score(
    y_trainval,
    trainval_prediction
)


print()
print("=" * 78)
print("TRAIN + VAL FIT METRICS")
print("=" * 78)

print(
    "MAE:",
    f"{trainval_mae:.8f}"
)

print(
    "RMSE:",
    f"{trainval_rmse:.8f}"
)

print(
    "R2:",
    f"{trainval_r2:.8f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature":
        final_features,

    "importance":
        model.feature_importances_,
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


print()
print("=" * 78)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 78)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# TRAINING RECORDS
# ============================================================

training_records = trainval[
    [
        "sequence",
        "current_frame",
        "component",
        "assigned_split",
        TARGET_COLUMN,
    ]
].copy()

training_records.to_csv(
    TRAINING_RECORDS_FILE,
    index=False
)


# ============================================================
# MODEL PACKAGE
# ============================================================

model_package = {

    "model":
        model,

    "model_type":
        "GradientBoostingRegressor",

    "feature_names":
        final_features,

    "target":
        TARGET_COLUMN,

    "feature_count":
        33,

    "complete_feature_policy":
        "Rows missing any selected feature are excluded",

    "training_policy":
        "TRAIN + VAL only; TEST excluded",

    "parameters":
        MODEL_PARAMETERS,

    "project":
        "BeltCrack9kd_Damage_Intelligence",

    "stage":
        "3.26",
}


with open(
    MODEL_FILE,
    "wb"
) as f:

    pickle.dump(
        model_package,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# FEATURE METADATA
# ============================================================

feature_metadata = {

    "stage":
        "3.26",

    "feature_count":
        33,

    "features":
        final_features,

    "temporal_features":
        temporal_features,

    "spatial_features":
        spatial_features,

    "target":
        TARGET_COLUMN,

    "complete_feature_policy":
        True,

    "test_used_for_training":
        False,

    "test_records_used":
        0,
}


with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_metadata,
        f,
        indent=2
    )


# ============================================================
# MODEL METADATA
# ============================================================

metadata = {

    "project":
        "BeltCrack9kd_Damage_Intelligence",

    "stage":
        "3.26",

    "model_type":
        "GradientBoostingRegressor",

    "feature_count":
        33,

    "features":
        final_features,

    "target":
        TARGET_COLUMN,

    "complete_feature_records":
        12149,

    "incomplete_records_excluded":
        incomplete_count,

    "train_records":
        train_count,

    "validation_records":
        val_count,

    "trainval_records":
        len(trainval),

    "test_records_excluded":
        test_count,

    "test_records_used_for_training":
        0,

    "trainval_sequences":
        trainval_sequences,

    "trainval_components":
        trainval_components,

    "parameters":
        MODEL_PARAMETERS,

    "trainval_fit_metrics":
        {
            "mae":
                float(trainval_mae),

            "rmse":
                float(trainval_rmse),

            "r2":
                float(trainval_r2),
        },

    "official_stage_3_25_test_metrics":
        {
            "mae":
                0.00127064,

            "rmse":
                0.00180600,

            "r2":
                0.74991233,
        },

    "artifact_files":
        {
            "model":
                str(MODEL_FILE),

            "feature_list":
                str(FEATURE_FILE),

            "metadata":
                str(METADATA_FILE),

            "importance":
                str(IMPORTANCE_FILE),

            "frame_component_mapping":
                str(FRAME_COMPONENT_MAP_FILE),
        },
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.26",

    "stage_name":
        "Final Model Artifact Training",

    "status":
        "PASS",

    "split_mapping_method":
        "canonical_manifest sequence/frame -> "
        "component -> Stage 0.9D1 assigned_split",

    "complete_feature_policy":
        {
            "enabled":
                True,

            "imputation":
                False,

            "incomplete_rows_excluded":
                incomplete_count,

            "complete_records":
                len(merged),
        },

    "training_policy":
        {
            "train_records":
                train_count,

            "validation_records":
                val_count,

            "trainval_records":
                len(trainval),

            "test_records":
                test_count,

            "test_records_used":
                0,

            "test_excluded":
                True,
        },

    "model":
        {
            "type":
                "GradientBoostingRegressor",

            "feature_count":
                33,

            "target":
                TARGET_COLUMN,

            "parameters":
                MODEL_PARAMETERS,
        },

    "fit_metrics":
        {
            "mae":
                float(trainval_mae),

            "rmse":
                float(trainval_rmse),

            "r2":
                float(trainval_r2),
        },

    "official_held_out_test":
        {
            "stage":
                "3.25",

            "mae":
                0.00127064,

            "rmse":
                0.00180600,

            "r2":
                0.74991233,
        },

    "artifacts":
        {
            "model":
                str(MODEL_FILE),

            "feature_list":
                str(FEATURE_FILE),

            "metadata":
                str(METADATA_FILE),

            "importance":
                str(IMPORTANCE_FILE),

            "training_records":
                str(TRAINING_RECORDS_FILE),

            "frame_component_mapping":
                str(FRAME_COMPONENT_MAP_FILE),
        },
}


with open(
    TRAINING_SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# VERIFY ARTIFACTS
# ============================================================

print()
print("=" * 78)
print("VERIFYING MODEL ARTIFACTS")
print("=" * 78)

artifact_files = [
    MODEL_FILE,
    FEATURE_FILE,
    METADATA_FILE,
    TRAINING_SUMMARY_FILE,
    TRAINING_RECORDS_FILE,
    IMPORTANCE_FILE,
    FRAME_COMPONENT_MAP_FILE,
]

for file_path in artifact_files:

    exists = file_path.exists()

    size = (
        file_path.stat().st_size
        if exists
        else 0
    )

    print(
        f"{file_path.name:<60}"
        f"{'OK' if exists else 'MISSING':<10}"
        f"{size} bytes"
    )

    if not exists:

        raise RuntimeError(
            "Artifact missing:\n"
            f"{file_path}"
        )


# ============================================================
# RELOAD MODEL
# ============================================================

print()
print(
    "Reloading saved model..."
)

with open(
    MODEL_FILE,
    "rb"
) as f:

    loaded_package = pickle.load(
        f
    )

loaded_model = (
    loaded_package[
        "model"
    ]
)

loaded_features = (
    loaded_package[
        "feature_names"
    ]
)


if loaded_features != final_features:

    raise RuntimeError(
        "Saved feature list does not match "
        "Stage 3.24 final feature list."
    )


# ============================================================
# RELOAD PREDICTION CHECK
# ============================================================

reload_prediction = (
    loaded_model.predict(
        X_trainval[:10]
    )
)

if len(
    reload_prediction
) != 10:

    raise RuntimeError(
        "Reloaded model prediction check failed."
    )


print(
    "Reloaded model check: PASS"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print("STAGE 3.26 FINAL SUMMARY")
print("=" * 78)

print()
print(
    "Final model:",
    "GradientBoostingRegressor"
)

print(
    "Final feature count:",
    33
)

print(
    "Complete records:",
    len(merged)
)

print(
    "TRAIN records:",
    train_count
)

print(
    "VAL records:",
    val_count
)

print(
    "TRAIN + VAL:",
    len(trainval)
)

print(
    "TEST records:",
    test_count
)

print(
    "TEST records used:",
    0
)

print()
print(
    "TRAIN+VAL fit MAE:",
    f"{trainval_mae:.8f}"
)

print(
    "TRAIN+VAL fit RMSE:",
    f"{trainval_rmse:.8f}"
)

print(
    "TRAIN+VAL fit R2:",
    f"{trainval_r2:.8f}"
)

print()
print(
    "Official held-out TEST MAE:",
    "0.00127064"
)

print(
    "Official held-out TEST R2:",
    "0.74991233"
)

print()
print(
    "Model artifact:",
    MODEL_FILE
)

print(
    "Feature list:",
    FEATURE_FILE
)

print(
    "Metadata:",
    METADATA_FILE
)

print(
    "Feature importance:",
    IMPORTANCE_FILE
)

print(
    "Frame/component mapping:",
    FRAME_COMPONENT_MAP_FILE
)

print(
    "Training summary:",
    TRAINING_SUMMARY_FILE
)

print()
print("=" * 78)
print("STAGE 3.26 STATUS: PASS")
print("=" * 78)