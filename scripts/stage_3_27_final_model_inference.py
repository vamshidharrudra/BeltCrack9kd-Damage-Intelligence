import os
import json
import pickle
import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STAGE 3.27 - FINAL MODEL INFERENCE / DAMAGE INTELLIGENCE
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "beltcrack9kd_final_gbr_33_features.pkl"
)

FEATURE_LIST_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "beltcrack9kd_final_33_feature_list.json"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "beltcrack9kd_final_model_metadata.json"
)

TARGET_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_15_physical_regression_targets.csv"
)

TEMPORAL_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_17_temporal_physical_features.csv"
)

SPATIAL_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_1_spatial_track_features.csv"
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

ASSIGNMENT_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_0_9d1_component_assignments.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

PREDICTION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_27_final_inference_predictions.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_27_final_inference_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def extract_frame(value):

    value = str(value)

    stem = os.path.splitext(
        os.path.basename(value)
    )[0]

    if stem.isdigit():
        return int(stem)

    numbers = re.findall(
        r"\d+",
        stem
    )

    if not numbers:
        return np.nan

    return int(numbers[-1])


def find_sequence_column(df, name):

    candidates = [
        "sequence",
        "sequence_id",
        "sequence_name"
    ]

    for col in candidates:

        if col in df.columns:
            return col

    raise ValueError(
        f"{name} does not contain a sequence identifier."
    )


# ============================================================
# IMPORTANT MODEL LOADER
# ============================================================

def find_predictor(obj, path="root"):

    """
    Recursively search a loaded pickle object for an object
    that provides a predict() method.

    This handles the Stage 3.26 artifact being saved as a
    dictionary containing the actual estimator.
    """

    if hasattr(obj, "predict"):

        print(
            "\nActual predictor found at:",
            path
        )

        print(
            "Predictor type:",
            type(obj).__name__
        )

        return obj


    if isinstance(obj, dict):

        for key, value in obj.items():

            print(
                f"Inspecting artifact key: {key}"
            )

            result = find_predictor(
                value,
                path=f"{path}['{key}']"
            )

            if result is not None:
                return result


    if isinstance(obj, (list, tuple)):

        for i, value in enumerate(obj):

            result = find_predictor(
                value,
                path=f"{path}[{i}]"
            )

            if result is not None:
                return result


    return None


# ============================================================
# START
# ============================================================

print_header(
    "STAGE 3.27 - FINAL MODEL INFERENCE / DAMAGE INTELLIGENCE"
)

print("\nProject:")
print(PROJECT_ROOT)

print("\nInference policy:")
print(" - Load saved Stage 3.26 model")
print(" - Extract actual estimator from artifact")
print(" - Load exact 33-feature list")
print(" - Do NOT retrain model")
print(" - Do NOT use test labels to construct features")
print(" - Demonstrate inference on held-out TEST records")
print(" - Compare predictions with official test targets only AFTER prediction")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print_header(
    "CHECKING REQUIRED FILES"
)

required_files = [
    MODEL_PATH,
    FEATURE_LIST_PATH,
    METADATA_PATH,
    TARGET_PATH,
    TEMPORAL_PATH,
    SPATIAL_PATH,
    MANIFEST_PATH,
    ASSIGNMENT_PATH
]

for path in required_files:

    if os.path.exists(path):

        print(
            os.path.basename(path),
            "FOUND"
        )

    else:

        print(
            os.path.basename(path),
            "MISSING"
        )

        raise FileNotFoundError(path)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

print_header(
    "LOADING SAVED MODEL ARTIFACT"
)

with open(
    MODEL_PATH,
    "rb"
) as f:

    artifact = pickle.load(f)


print(
    "\nSaved artifact type:",
    type(artifact).__name__
)


if isinstance(artifact, dict):

    print(
        "\nArtifact dictionary keys:"
    )

    for key in artifact.keys():

        print(
            " -",
            key
        )


# Find actual estimator
model = find_predictor(artifact)


if model is None:

    raise TypeError(
        "\nCould not find an object with "
        "a predict() method inside the saved artifact."
    )


print(
    "\nModel loaded successfully:"
)

print(
    type(model).__name__
)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

with open(
    FEATURE_LIST_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_data = json.load(f)


if isinstance(feature_data, list):

    feature_list = feature_data

elif isinstance(feature_data, dict):

    if "features" in feature_data:

        feature_list = feature_data["features"]

    elif "feature_names" in feature_data:

        feature_list = feature_data["feature_names"]

    else:

        raise ValueError(
            "Could not locate feature list inside JSON."
        )

else:

    raise ValueError(
        "Unsupported feature-list JSON format."
    )


feature_list = list(feature_list)


print(
    "\nFinal feature count:",
    len(feature_list)
)


for i, feature in enumerate(
    feature_list,
    1
):

    print(
        f"{i:02d}. {feature}"
    )


if len(feature_list) != 33:

    raise ValueError(
        f"Expected exactly 33 features, "
        f"found {len(feature_list)}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(
    "\nMetadata loaded successfully."
)


# ============================================================
# LOAD DATA
# ============================================================

print_header(
    "LOADING INFERENCE DATA"
)

target = pd.read_csv(
    TARGET_PATH
)

temporal = pd.read_csv(
    TEMPORAL_PATH
)

spatial = pd.read_csv(
    SPATIAL_PATH
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

assignments = pd.read_csv(
    ASSIGNMENT_PATH
)


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
    "Manifest records:",
    len(manifest)
)

print(
    "Assignment rows:",
    len(assignments)
)


# ============================================================
# BUILD FRAME IDENTIFIERS
# ============================================================

print_header(
    "BUILDING FRAME IDENTIFIERS"
)


# Target
if "current_frame" in target.columns:

    target["frame_key"] = pd.to_numeric(
        target["current_frame"],
        errors="coerce"
    )

elif "frame_number" in target.columns:

    target["frame_key"] = pd.to_numeric(
        target["frame_number"],
        errors="coerce"
    )

else:

    raise ValueError(
        "Target file does not contain "
        "current_frame or frame_number."
    )


# Temporal
if "current_frame" in temporal.columns:

    temporal["frame_key"] = pd.to_numeric(
        temporal["current_frame"],
        errors="coerce"
    )

elif "frame_number" in temporal.columns:

    temporal["frame_key"] = pd.to_numeric(
        temporal["frame_number"],
        errors="coerce"
    )

else:

    raise ValueError(
        "Temporal file does not contain "
        "current_frame or frame_number."
    )


# Spatial
if "current_frame" in spatial.columns:

    spatial["frame_key"] = pd.to_numeric(
        spatial["current_frame"],
        errors="coerce"
    )

elif "frame_number" in spatial.columns:

    spatial["frame_key"] = pd.to_numeric(
        spatial["frame_number"],
        errors="coerce"
    )

else:

    raise ValueError(
        "Spatial file does not contain "
        "current_frame or frame_number."
    )


# Manifest
manifest["frame_key"] = manifest[
    "canonical_image_filename"
].apply(
    extract_frame
)

manifest["frame_key"] = pd.to_numeric(
    manifest["frame_key"],
    errors="coerce"
)


print(
    "Manifest frame numbers extracted:",
    manifest["frame_key"].notna().sum(),
    "/",
    len(manifest)
)


# ============================================================
# SEQUENCE COLUMNS
# ============================================================

target_seq_col = find_sequence_column(
    target,
    "Target"
)

temporal_seq_col = find_sequence_column(
    temporal,
    "Temporal"
)

spatial_seq_col = find_sequence_column(
    spatial,
    "Spatial"
)

manifest_seq_col = find_sequence_column(
    manifest,
    "Manifest"
)


print(
    "\nSequence columns:"
)

print(
    "Target   :",
    target_seq_col
)

print(
    "Temporal :",
    temporal_seq_col
)

print(
    "Spatial  :",
    spatial_seq_col
)

print(
    "Manifest :",
    manifest_seq_col
)


target["sequence_key"] = (
    target[target_seq_col]
    .astype(str)
)

temporal["sequence_key"] = (
    temporal[temporal_seq_col]
    .astype(str)
)

spatial["sequence_key"] = (
    spatial[spatial_seq_col]
    .astype(str)
)

manifest["sequence_key"] = (
    manifest[manifest_seq_col]
    .astype(str)
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

print_header(
    "CHECKING DUPLICATE FRAME KEYS"
)

target_duplicates = target.duplicated(
    ["sequence_key", "frame_key"]
).sum()

temporal_duplicates = temporal.duplicated(
    ["sequence_key", "frame_key"]
).sum()

spatial_duplicates = spatial.duplicated(
    ["sequence_key", "frame_key"]
).sum()

manifest_duplicates = manifest.duplicated(
    ["sequence_key", "frame_key"]
).sum()


print(
    "Target duplicates   :",
    target_duplicates
)

print(
    "Temporal duplicates :",
    temporal_duplicates
)

print(
    "Spatial duplicates  :",
    spatial_duplicates
)

print(
    "Manifest duplicates :",
    manifest_duplicates
)


if target_duplicates != 0:
    raise ValueError(
        "Duplicate target sequence/frame keys."
    )

if temporal_duplicates != 0:
    raise ValueError(
        "Duplicate temporal sequence/frame keys."
    )

if spatial_duplicates != 0:
    raise ValueError(
        "Duplicate spatial sequence/frame keys."
    )

if manifest_duplicates != 0:
    raise ValueError(
        "Duplicate manifest sequence/frame keys."
    )


# ============================================================
# TARGET
# ============================================================

print_header(
    "VERIFYING REGRESSION TARGET"
)

target_candidates = [
    "future_crack_area_ratio_t1",
    "future_damage_t1",
    "target"
]

target_column = None

for col in target_candidates:

    if col in target.columns:

        target_column = col
        break


if target_column is None:

    raise ValueError(
        "Could not find t+1 physical damage target."
    )


print(
    "Target column:",
    target_column
)


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

print_header(
    "BUILDING FINAL 33-FEATURE MATRIX"
)


base = target[
    [
        "sequence_key",
        "frame_key",
        target_column
    ]
].copy()


# ----------------------------
# TEMPORAL FEATURES
# ----------------------------

temporal_feature_names = [
    f
    for f in feature_list
    if f in temporal.columns
]


temporal_small = temporal[
    [
        "sequence_key",
        "frame_key"
    ]
    +
    temporal_feature_names
].copy()


base = base.merge(
    temporal_small,
    on=[
        "sequence_key",
        "frame_key"
    ],
    how="left"
)


# ----------------------------
# SPATIAL FEATURES
# ----------------------------

spatial_feature_names = [
    f
    for f in feature_list
    if f in spatial.columns
]


spatial_small = spatial[
    [
        "sequence_key",
        "frame_key"
    ]
    +
    spatial_feature_names
].copy()


base = base.merge(
    spatial_small,
    on=[
        "sequence_key",
        "frame_key"
    ],
    how="left",
    suffixes=(
        "",
        "_spatial"
    )
)


# ============================================================
# RESOLVE EXACT FEATURES
# ============================================================

missing_features = []

X = pd.DataFrame(
    index=base.index
)


for feature in feature_list:

    if feature in base.columns:

        source_column = feature

    elif f"{feature}_spatial" in base.columns:

        source_column = f"{feature}_spatial"

    else:

        missing_features.append(
            feature
        )

        continue


    X[feature] = pd.to_numeric(
        base[source_column],
        errors="coerce"
    )


if missing_features:

    print(
        "\nMISSING REQUIRED FEATURES:"
    )

    for feature in missing_features:

        print(
            " -",
            feature
        )

    raise ValueError(
        "One or more final 33 features are missing."
    )


y = pd.to_numeric(
    base[target_column],
    errors="coerce"
)


print(
    "\nMerged records:",
    len(base)
)

print(
    "Feature matrix:",
    X.shape
)


# ============================================================
# COMPLETE FEATURE POLICY
# ============================================================

print_header(
    "APPLYING FINAL COMPLETE-FEATURE POLICY"
)


complete_mask = (
    X.notna().all(axis=1)
    &
    y.notna()
)


complete_count = int(
    complete_mask.sum()
)


print(
    "Complete inference records:",
    complete_count
)

print(
    "Incomplete records:",
    len(X) - complete_count
)


if complete_count == 0:

    raise ValueError(
        "No complete feature records available."
    )


X_complete = X.loc[
    complete_mask
].copy()

y_complete = y.loc[
    complete_mask
].copy()

base_complete = base.loc[
    complete_mask
].copy()


# ============================================================
# FINITE CHECK
# ============================================================

nonfinite_mask = ~np.isfinite(
    X_complete.to_numpy()
).all(axis=1)


print(
    "Non-finite feature records:",
    int(nonfinite_mask.sum())
)


if nonfinite_mask.any():

    raise ValueError(
        "Non-finite values detected."
    )


# ============================================================
# BUILD COMPONENT/SPLIT MAPPING
# ============================================================

print_header(
    "IDENTIFYING HELD-OUT TEST RECORDS"
)


manifest_map = manifest[
    [
        "sequence_key",
        "frame_key",
        "component"
    ]
].copy()


manifest_map["component_id"] = pd.to_numeric(
    manifest_map["component"],
    errors="coerce"
)


manifest_map = manifest_map[
    [
        "sequence_key",
        "frame_key",
        "component_id"
    ]
]


base_complete = base_complete.merge(
    manifest_map,
    on=[
        "sequence_key",
        "frame_key"
    ],
    how="left"
)


if base_complete[
    "component_id"
].isna().any():

    raise ValueError(
        "Some inference records have no component mapping."
    )


component_split = assignments[
    [
        "component_id",
        "assigned_split"
    ]
].copy()


component_split["component_id"] = pd.to_numeric(
    component_split["component_id"],
    errors="coerce"
)


component_split["assigned_split"] = (
    component_split["assigned_split"]
    .astype(str)
    .str.upper()
)


base_complete = base_complete.merge(
    component_split,
    on="component_id",
    how="left"
)


if base_complete[
    "assigned_split"
].isna().any():

    raise ValueError(
        "Some inference records have no assigned split."
    )


print(
    base_complete[
        "assigned_split"
    ].value_counts().sort_index()
)


# ============================================================
# FINAL MODEL INFERENCE
# ============================================================

print_header(
    "RUNNING FINAL MODEL INFERENCE"
)


# IMPORTANT:
# The saved artifact was a dictionary.
# 'model' above is now the actual estimator.

print(
    "Predictor:",
    type(model).__name__
)

print(
    "Input shape:",
    X_complete.shape
)


predictions = model.predict(
    X_complete
)


predictions = np.asarray(
    predictions,
    dtype=float
)


print(
    "Predictions generated:",
    len(predictions)
)


# ============================================================
# BUILD RESULTS
# ============================================================

results = base_complete[
    [
        "sequence_key",
        "frame_key",
        "component_id",
        "assigned_split",
        target_column
    ]
].copy()


results["predicted_future_crack_area_ratio"] = (
    predictions
)


results["prediction_error"] = (
    results[
        "predicted_future_crack_area_ratio"
    ]
    -
    results[target_column]
)


results["absolute_error"] = (
    results["prediction_error"]
    .abs()
)


results["current_damage"] = (
    X_complete[
        "crack_area_ratio"
    ].values
)


results["predicted_damage_change"] = (
    results[
        "predicted_future_crack_area_ratio"
    ]
    -
    results["current_damage"]
)


results["predicted_change_percent"] = np.where(
    results["current_damage"].values != 0,

    (
        results[
            "predicted_damage_change"
        ].values
        /
        results[
            "current_damage"
        ].values
    ) * 100.0,

    np.nan
)


# ============================================================
# DAMAGE STATUS
# ============================================================

def classify_prediction(change):

    if change > 0.001:

        return "INCREASING"

    elif change < -0.001:

        return "DECREASING"

    else:

        return "STABLE"


results["predicted_damage_status"] = (
    results[
        "predicted_damage_change"
    ].apply(
        classify_prediction
    )
)


# ============================================================
# DAMAGE LEVEL
# ============================================================

def damage_level(value):

    if value < 0.002:

        return "VERY_LOW"

    elif value < 0.005:

        return "LOW"

    elif value < 0.008:

        return "MEDIUM"

    elif value < 0.012:

        return "HIGH"

    elif value < 0.016:

        return "VERY_HIGH"

    else:

        return "EXTREME"


results["predicted_damage_level"] = (
    results[
        "predicted_future_crack_area_ratio"
    ].apply(
        damage_level
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    PREDICTION_OUTPUT,
    index=False
)


print(
    "\nInference predictions saved:"
)

print(
    PREDICTION_OUTPUT
)


# ============================================================
# HELD-OUT TEST EVALUATION
# ============================================================

print_header(
    "OFFICIAL HELD-OUT TEST INFERENCE"
)


test_results = results[
    results["assigned_split"] == "TEST"
].copy()


print(
    "TEST inference records:",
    len(test_results)
)


if len(test_results) == 0:

    raise ValueError(
        "No TEST records found."
    )


test_y = test_results[
    target_column
].values


test_pred = test_results[
    "predicted_future_crack_area_ratio"
].values


test_mae = mean_absolute_error(
    test_y,
    test_pred
)


test_rmse = np.sqrt(
    mean_squared_error(
        test_y,
        test_pred
    )
)


test_r2 = r2_score(
    test_y,
    test_pred
)


print(
    "\nTEST MAE :",
    f"{test_mae:.8f}"
)

print(
    "TEST RMSE:",
    f"{test_rmse:.8f}"
)

print(
    "TEST R2  :",
    f"{test_r2:.8f}"
)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

print_header(
    "PREDICTED DAMAGE STATUS DISTRIBUTION"
)

print(
    test_results[
        "predicted_damage_status"
    ].value_counts()
)


print_header(
    "PREDICTED DAMAGE LEVEL DISTRIBUTION"
)

print(
    test_results[
        "predicted_damage_level"
    ].value_counts()
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

abs_errors = test_results[
    "absolute_error"
].values


print_header(
    "TEST ERROR DISTRIBUTION"
)


print(
    "Mean absolute error:",
    f"{np.mean(abs_errors):.8f}"
)

print(
    "Median absolute error:",
    f"{np.median(abs_errors):.8f}"
)

print(
    "P90 absolute error:",
    f"{np.percentile(abs_errors, 90):.8f}"
)

print(
    "P95 absolute error:",
    f"{np.percentile(abs_errors, 95):.8f}"
)

print(
    "P99 absolute error:",
    f"{np.percentile(abs_errors, 99):.8f}"
)

print(
    "Maximum absolute error:",
    f"{np.max(abs_errors):.8f}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print_header(
    "SAMPLE DAMAGE INTELLIGENCE PREDICTIONS"
)


for _, row in test_results.head(
    10
).iterrows():

    print(
        "\nSequence:",
        row["sequence_key"]
    )

    print(
        "Frame:",
        int(row["frame_key"])
    )

    print(
        "Current damage:",
        f"{row['current_damage']:.6f}"
    )

    print(
        "Predicted next damage:",
        f"{row['predicted_future_crack_area_ratio']:.6f}"
    )

    print(
        "Predicted change:",
        f"{row['predicted_damage_change']:.6f}"
    )

    print(
        "Predicted change %:",
        f"{row['predicted_change_percent']:.2f}%"
    )

    print(
        "Status:",
        row["predicted_damage_status"]
    )

    print(
        "Damage level:",
        row["predicted_damage_level"]
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage": "3.27",

    "status": "PASS",

    "model": type(model).__name__,

    "model_artifact": MODEL_PATH,

    "feature_count": len(feature_list),

    "features": feature_list,

    "complete_records": int(
        len(results)
    ),

    "test_records": int(
        len(test_results)
    ),

    "test_mae": float(
        test_mae
    ),

    "test_rmse": float(
        test_rmse
    ),

    "test_r2": float(
        test_r2
    ),

    "mean_absolute_error": float(
        np.mean(abs_errors)
    ),

    "median_absolute_error": float(
        np.median(abs_errors)
    ),

    "p90_absolute_error": float(
        np.percentile(abs_errors, 90)
    ),

    "p95_absolute_error": float(
        np.percentile(abs_errors, 95)
    ),

    "p99_absolute_error": float(
        np.percentile(abs_errors, 99)
    ),

    "maximum_absolute_error": float(
        np.max(abs_errors)
    ),

    "test_sequences": int(
        test_results[
            "sequence_key"
        ].nunique()
    ),

    "test_components": int(
        test_results[
            "component_id"
        ].nunique()
    ),

    "test_used_for_model_training": False,

    "inference_only": True
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "STAGE 3.27 FINAL SUMMARY"
)


print(
    "Model:",
    type(model).__name__
)

print(
    "Feature count:",
    len(feature_list)
)

print(
    "Inference records:",
    len(results)
)

print(
    "TEST records:",
    len(test_results)
)

print(
    "TEST MAE:",
    f"{test_mae:.8f}"
)

print(
    "TEST RMSE:",
    f"{test_rmse:.8f}"
)

print(
    "TEST R2:",
    f"{test_r2:.8f}"
)

print(
    "\nPrediction output:"
)

print(
    PREDICTION_OUTPUT
)

print(
    "\nSummary output:"
)

print(
    SUMMARY_OUTPUT
)


print(
    "\n" + "=" * 78
)

print(
    "STAGE 3.27 STATUS: PASS"
)

print(
    "=" * 78
)