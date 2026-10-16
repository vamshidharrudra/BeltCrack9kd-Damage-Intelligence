import os
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.22
# VALIDATION-LOCKED FINAL FEATURE SELECTION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

TARGET_FILE = AUDIT_DIR / "stage_3_15_physical_regression_targets.csv"
TEMPORAL_FILE = AUDIT_DIR / "stage_3_17_temporal_physical_features.csv"
SPATIAL_FILE = AUDIT_DIR / "stage_3_1_spatial_track_features.csv"
MANIFEST_FILE = PROJECT_ROOT / "canonical_dataset" / "canonical_manifest.csv"
ASSIGNMENT_FILE = AUDIT_DIR / "stage_0_9d1_component_assignments.csv"

OUT_RANKING = AUDIT_DIR / "stage_3_22_validation_feature_ranking.csv"
OUT_COMPARISON = AUDIT_DIR / "stage_3_22_validation_feature_set_comparison.csv"
OUT_LOCKED_SPATIAL = AUDIT_DIR / "stage_3_22_locked_spatial_features.csv"
OUT_FINAL_FEATURES = AUDIT_DIR / "stage_3_22_final_feature_set.csv"
OUT_RESULTS = AUDIT_DIR / "stage_3_22_final_model_results.csv"
OUT_SUMMARY = AUDIT_DIR / "stage_3_22_validation_locked_selection_summary.json"

RANDOM_STATE = 42

GB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

TOP_K_VALUES = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 132]

SELECTION_TOLERANCE = 0.01

TEMPORAL_FEATURES = [
    "crack_area_ratio",
    "lag_1_damage",
    "lag_2_damage",
    "lag_3_damage",
    "lag_5_damage",
    "delta_1_damage",
    "delta_2_damage",
    "delta_3_damage",
    "acceleration_1_damage",
    "acceleration_2_damage",
    "rolling3_mean_damage",
    "rolling3_std_damage",
    "rolling5_mean_damage",
    "rolling5_std_damage",
    "rolling5_min_damage",
    "rolling5_max_damage",
    "recent3_damage_slope",
    "recent5_damage_slope",
]

TARGET_COLUMN = "future_crack_area_ratio_t1"


# ============================================================
# HELPERS
# ============================================================

def ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def check_duplicates(df, keys, name):
    dup = int(df.duplicated(keys).sum())

    print(f"{name} duplicate keys: {dup}")

    if dup > 0:
        examples = (
            df.loc[
                df.duplicated(keys, keep=False),
                keys
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"{name} contains duplicate keys.\n\n{examples}"
        )


def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "rmse": rmse(
            y_true,
            y_pred
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
    }


def make_model():
    return GradientBoostingRegressor(
        **GB_PARAMS
    )


def extract_frame_number(value):
    """
    Robust frame extraction.

    The canonical filename can contain prefixes/suffixes,
    so do not assume Path(value).stem is itself numeric.

    Strategy:
    1. Try complete stem as integer.
    2. Try basename without extension.
    3. Search numeric tokens.
    4. Prefer the final numeric token.

    Examples handled:
        0.jpg
        639.jpg
        frame_639.jpg
        639_frame.jpg
        something_639_image.jpg
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    # Normalize slash styles.
    text = text.replace("\\", "/")

    # Get final path component.
    basename = text.split("/")[-1]

    # Remove extension.
    stem = Path(basename).stem

    # --------------------------------------------------------
    # Exact integer filename
    # --------------------------------------------------------

    if re.fullmatch(
        r"\d+",
        stem
    ):
        try:
            return int(stem)
        except Exception:
            pass

    # --------------------------------------------------------
    # Numeric tokens
    # --------------------------------------------------------

    tokens = re.findall(
        r"\d+",
        stem
    )

    if tokens:
        try:
            return int(tokens[-1])
        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback: search complete original value
    # --------------------------------------------------------

    tokens = re.findall(
        r"\d+",
        text
    )

    if tokens:
        try:
            return int(tokens[-1])
        except Exception:
            pass

    return np.nan


def find_column(df, candidates, label):

    for c in candidates:

        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find {label}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def detect_spatial_features(df):

    excluded = {
        "sequence",
        "current_frame",
        "future_frame",
        "component",
        "component_id",
        "split",
        "assigned_split",
        "record_id",
        "source_image",
        "source_json",
    }

    spatial_features = []

    for c in df.columns:

        if c in excluded:
            continue

        lc = c.lower()

        spatial_like = (
            "spatial" in lc
            or c.startswith("current_")
            or c in {
                "track_continuity_quality",
                "persistent_fraction",
                "future_coverage",
                "mean_match_iou",
                "mean_center_distance",
                "mean_area_similarity",
                "matched_crack_count",
                "matched_fraction_current",
                "mean_match_score",
                "spatial_match_quality",
                "continuity_score",
                "track_continuity_quality",
            }
        )

        if spatial_like:

            if pd.api.types.is_numeric_dtype(
                df[c]
            ):
                spatial_features.append(c)

    return sorted(
        set(spatial_features)
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("STAGE 3.22 - VALIDATION-LOCKED FINAL FEATURE SELECTION")
print("=" * 80)

print("\nProject:")
print(PROJECT_ROOT)

print("\nMethodological rule:")
print(
    "TEST SET WILL NOT BE USED FOR FEATURE RANKING "
    "OR FEATURE SELECTION."
)
print("Feature ranking -> VALIDATION")
print("Feature-count selection -> VALIDATION")
print("Final evaluation -> TEST")


# ============================================================
# INPUT CHECK
# ============================================================

for path, label in [
    (TARGET_FILE, "Physical target file"),
    (TEMPORAL_FILE, "Temporal feature file"),
    (SPATIAL_FILE, "Spatial feature file"),
    (MANIFEST_FILE, "Canonical manifest"),
    (ASSIGNMENT_FILE, "Component assignment file"),
]:

    ensure_exists(
        path,
        label
    )


# ============================================================
# LOAD
# ============================================================

print("\n[1/12] Loading inputs...")

target = pd.read_csv(
    TARGET_FILE
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

assignments = pd.read_csv(
    ASSIGNMENT_FILE
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
    "Assignments:",
    len(assignments)
)


# ============================================================
# TARGET
# ============================================================

print("\n[2/12] Normalizing target keys...")

target_sequence = find_column(
    target,
    ["sequence"],
    "target sequence"
)

target_frame = find_column(
    target,
    ["current_frame"],
    "target current frame"
)

target_col = find_column(
    target,
    [TARGET_COLUMN],
    "future physical target"
)

target = target.copy()

target["sequence"] = (
    target[target_sequence]
    .astype(str)
    .str.strip()
)

target["current_frame"] = pd.to_numeric(
    target[target_frame],
    errors="coerce"
)

target = target[
    target["current_frame"].notna()
    & target[target_col].notna()
].copy()

target["current_frame"] = (
    target["current_frame"]
    .astype(int)
)

check_duplicates(
    target,
    ["sequence", "current_frame"],
    "Target"
)

print(
    "Valid target records:",
    len(target)
)


# ============================================================
# TEMPORAL
# ============================================================

print("\n[3/12] Normalizing temporal features...")

temporal_sequence = find_column(
    temporal,
    ["sequence"],
    "temporal sequence"
)

temporal_frame = find_column(
    temporal,
    ["current_frame"],
    "temporal current frame"
)

temporal["sequence"] = (
    temporal[temporal_sequence]
    .astype(str)
    .str.strip()
)

temporal["current_frame"] = pd.to_numeric(
    temporal[temporal_frame],
    errors="coerce"
)

missing_temporal = [
    c
    for c in TEMPORAL_FEATURES
    if c not in temporal.columns
]

if missing_temporal:

    raise KeyError(
        "Missing expected temporal features:\n"
        + "\n".join(
            missing_temporal
        )
    )

temporal = temporal[
    temporal["current_frame"].notna()
].copy()

temporal["current_frame"] = (
    temporal["current_frame"]
    .astype(int)
)

check_duplicates(
    temporal,
    ["sequence", "current_frame"],
    "Temporal"
)

print(
    "Temporal features:",
    len(TEMPORAL_FEATURES)
)


# ============================================================
# SPATIAL
# ============================================================

print("\n[4/12] Normalizing spatial features...")

spatial_sequence = find_column(
    spatial,
    ["sequence"],
    "spatial sequence"
)

spatial_frame = find_column(
    spatial,
    ["current_frame"],
    "spatial current frame"
)

spatial["sequence"] = (
    spatial[spatial_sequence]
    .astype(str)
    .str.strip()
)

spatial["current_frame"] = pd.to_numeric(
    spatial[spatial_frame],
    errors="coerce"
)

spatial = spatial[
    spatial["current_frame"].notna()
].copy()

spatial["current_frame"] = (
    spatial["current_frame"]
    .astype(int)
)

check_duplicates(
    spatial,
    ["sequence", "current_frame"],
    "Spatial"
)

spatial_features = detect_spatial_features(
    spatial
)

print(
    "Detected spatial features:",
    len(spatial_features)
)

if len(spatial_features) < 100:

    raise RuntimeError(
        "Unexpectedly few spatial features detected. "
        f"Detected {len(spatial_features)}."
    )

print(
    "Expected Stage 3.20/3.21 spatial count: 132"
)


# ============================================================
# MANIFEST
# ============================================================

print(
    "\n[5/12] Building sequence/frame -> component mapping..."
)

manifest = manifest.copy()

manifest["sequence"] = (
    manifest["sequence"]
    .astype(str)
    .str.strip()
)

manifest_filename_column = find_column(
    manifest,
    ["canonical_image_filename"],
    "canonical image filename"
)

print(
    "Manifest filename column:",
    manifest_filename_column
)

# ------------------------------------------------------------
# IMPORTANT FIX
# ------------------------------------------------------------
# The previous version assumed the filename stem itself
# was numeric. The canonical filenames are allowed to contain
# additional text. Use robust extraction.
# ------------------------------------------------------------

manifest["current_frame"] = (
    manifest[
        manifest_filename_column
    ]
    .apply(extract_frame_number)
)

valid_manifest_frames = int(
    manifest["current_frame"]
    .notna()
    .sum()
)

print(
    "Manifest frame numbers extracted:",
    valid_manifest_frames,
    "/",
    len(manifest)
)

if valid_manifest_frames == 0:

    print("\nDEBUG: first manifest filenames:")

    print(
        manifest[
            manifest_filename_column
        ]
        .head(20)
        .to_string(index=False)
    )

    raise RuntimeError(
        "Could not extract ANY frame numbers from "
        "canonical_image_filename."
    )

manifest = manifest[
    manifest["current_frame"].notna()
].copy()

manifest["current_frame"] = (
    manifest["current_frame"]
    .astype(int)
)

component_column = find_column(
    manifest,
    ["component"],
    "manifest component"
)

manifest["component_id"] = (
    manifest[component_column]
)

manifest_map = (
    manifest[
        [
            "sequence",
            "current_frame",
            "component_id",
        ]
    ]
    .drop_duplicates()
)

check_duplicates(
    manifest_map,
    [
        "sequence",
        "current_frame",
    ],
    "Manifest sequence/frame"
)

print(
    "Manifest mappings:",
    len(manifest_map)
)


# ============================================================
# ASSIGNMENTS
# ============================================================

print(
    "\n[6/12] Loading leakage-safe split assignments..."
)

assignment_component = find_column(
    assignments,
    ["component_id"],
    "assignment component"
)

assignment_split = find_column(
    assignments,
    ["assigned_split"],
    "assignment split"
)

assignments["component_id"] = (
    assignments[
        assignment_component
    ]
)

assignments["assigned_split"] = (
    assignments[
        assignment_split
    ]
    .astype(str)
    .str.strip()
)

component_split = (
    assignments[
        [
            "component_id",
            "assigned_split",
        ]
    ]
    .drop_duplicates()
)

component_conflicts = (
    component_split
    .groupby("component_id")[
        "assigned_split"
    ]
    .nunique()
)

if (
    component_conflicts > 1
).any():

    raise RuntimeError(
        "Component split conflicts detected."
    )

print(
    "Components with assignments:",
    component_split[
        "component_id"
    ].nunique()
)


# ============================================================
# MERGE
# ============================================================

print(
    "\n[7/12] Building exact common modeling dataset..."
)

base = target[
    [
        "sequence",
        "current_frame",
        target_col,
    ]
].copy()

base = base.merge(
    temporal[
        [
            "sequence",
            "current_frame",
        ]
        + TEMPORAL_FEATURES
    ],
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After target + temporal merge:",
    len(base)
)

base = base.merge(
    spatial[
        [
            "sequence",
            "current_frame",
        ]
        + spatial_features
    ],
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After spatial merge:",
    len(base)
)

base = base.merge(
    manifest_map,
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After manifest merge:",
    len(base)
)

base = base.merge(
    component_split,
    on="component_id",
    how="left",
    validate="many_to_one",
)

print(
    "After split assignment merge:",
    len(base)
)

if base.empty:

    raise RuntimeError(
        "Merged dataset is empty. "
        "The frame/sequence keys do not align."
    )

if base["assigned_split"].isna().any():

    missing_split = int(
        base["assigned_split"]
        .isna()
        .sum()
    )

    raise RuntimeError(
        f"{missing_split} records have no assigned split."
    )

if base.duplicated(
    [
        "sequence",
        "current_frame",
    ]
).any():

    raise RuntimeError(
        "Duplicate sequence/frame rows "
        "after final merge."
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

all_model_features = (
    TEMPORAL_FEATURES
    + spatial_features
)

if (
    len(
        set(all_model_features)
    )
    != len(all_model_features)
):

    duplicates = (
        pd.Series(all_model_features)
        [
            pd.Series(
                all_model_features
            ).duplicated()
        ]
        .tolist()
    )

    raise RuntimeError(
        f"Duplicate feature names: {duplicates}"
    )

for c in (
    all_model_features
    + [target_col]
):

    base[c] = pd.to_numeric(
        base[c],
        errors="coerce"
    )

before_complete = len(base)

base = base.dropna(
    subset=(
        all_model_features
        + [target_col]
    )
).copy()

print(
    "Complete records:",
    len(base),
    "/",
    before_complete
)

print(
    "Sequences:",
    base["sequence"].nunique()
)

print(
    "Components:",
    base["component_id"].nunique()
)

print("\nSplit distribution:")

print(
    base[
        "assigned_split"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

expected_counts = {
    "train": 8573,
    "val": 1716,
    "test": 1860,
}

for split_name, expected in (
    expected_counts.items()
):

    actual = int(
        (
            base["assigned_split"]
            == split_name
        ).sum()
    )

    print(
        f"{split_name.upper()}: "
        f"{actual} "
        f"(expected {expected})"
    )

    if actual != expected:

        raise RuntimeError(
            f"Unexpected {split_name} count: "
            f"{actual} != {expected}"
        )


# ============================================================
# DATA MATRICES
# ============================================================

print(
    "\n[8/12] Creating leakage-safe "
    "train/validation/test matrices..."
)

train_df = base[
    base["assigned_split"]
    == "train"
].copy()

val_df = base[
    base["assigned_split"]
    == "val"
].copy()

test_df = base[
    base["assigned_split"]
    == "test"
].copy()

feature_names = (
    TEMPORAL_FEATURES
    + spatial_features
)

feature_index = {
    name: i
    for i, name
    in enumerate(
        feature_names
    )
}

X_train_full = (
    train_df[
        feature_names
    ]
    .to_numpy(
        dtype=float
    )
)

X_val_full = (
    val_df[
        feature_names
    ]
    .to_numpy(
        dtype=float
    )
)

X_test_full = (
    test_df[
        feature_names
    ]
    .to_numpy(
        dtype=float
    )
)

y_train = (
    train_df[
        target_col
    ]
    .to_numpy(
        dtype=float
    )
)

y_val = (
    val_df[
        target_col
    ]
    .to_numpy(
        dtype=float
    )
)

y_test = (
    test_df[
        target_col
    ]
    .to_numpy(
        dtype=float
    )
)

print(
    "TRAIN:",
    X_train_full.shape
)

print(
    "VAL:",
    X_val_full.shape
)

print(
    "TEST:",
    X_test_full.shape
)

print(
    "Temporal features:",
    len(TEMPORAL_FEATURES)
)

print(
    "Spatial features:",
    len(spatial_features)
)

print(
    "Total features:",
    len(feature_names)
)


# ============================================================
# FULL MODEL
# ============================================================

print(
    "\n[9/12] Training full model on TRAIN only..."
)

full_model = make_model()

full_model.fit(
    X_train_full,
    y_train
)

val_full_pred = (
    full_model.predict(
        X_val_full
    )
)

val_full_metrics = regression_metrics(
    y_val,
    val_full_pred
)

print("\nFULL MODEL VALIDATION")

print(
    "MAE :",
    f"{val_full_metrics['mae']:.8f}"
)

print(
    "RMSE:",
    f"{val_full_metrics['rmse']:.8f}"
)

print(
    "R2  :",
    f"{val_full_metrics['r2']:.8f}"
)


# ============================================================
# VALIDATION PERMUTATION IMPORTANCE
# ============================================================

print(
    "\n[10/12] Computing permutation importance "
    "on VALIDATION only..."
)

perm = permutation_importance(
    full_model,
    X_val_full,
    y_val,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

ranking = pd.DataFrame({
    "feature": feature_names,
    "permutation_importance_mae":
        perm.importances_mean,
    "permutation_importance_std":
        perm.importances_std,
})

ranking["feature_group"] = np.where(
    ranking["feature"].isin(
        spatial_features
    ),
    "SPATIAL",
    "TEMPORAL",
)

ranking["absolute_importance"] = (
    ranking[
        "permutation_importance_mae"
    ].abs()
)

ranking = ranking.sort_values(
    "permutation_importance_mae",
    ascending=False
).reset_index(
    drop=True
)

ranking["validation_rank"] = (
    np.arange(
        len(ranking)
    )
    + 1
)

ranking.to_csv(
    OUT_RANKING,
    index=False
)

spatial_ranking = ranking[
    ranking["feature_group"]
    == "SPATIAL"
].copy()

spatial_ranking = (
    spatial_ranking
    .sort_values(
        "permutation_importance_mae",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nTOP 30 SPATIAL FEATURES "
    "- VALIDATION RANKING:"
)

print(
    spatial_ranking[
        [
            "validation_rank",
            "feature",
            "permutation_importance_mae",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# VALIDATION FEATURE-COUNT SELECTION
# ============================================================

print(
    "\n[11/12] Validation-based "
    "feature-count selection..."
)

spatial_ranked_names = (
    spatial_ranking[
        "feature"
    ]
    .tolist()
)

candidate_sets = [
    {
        "name": "TEMPORAL_ONLY",
        "spatial_k": 0,
        "features":
            TEMPORAL_FEATURES.copy(),
    }
]

for k in TOP_K_VALUES:

    if k > len(
        spatial_ranked_names
    ):
        continue

    selected_spatial = (
        spatial_ranked_names[:k]
    )

    candidate_sets.append(
        {
            "name":
                f"TEMPORAL_PLUS_TOP_{k}_SPATIAL",
            "spatial_k":
                k,
            "features":
                (
                    TEMPORAL_FEATURES
                    + selected_spatial
                ),
        }
    )

candidate_results = []

for candidate in candidate_sets:

    features = candidate[
        "features"
    ]

    indices = [
        feature_index[f]
        for f in features
    ]

    Xtr = (
        X_train_full[
            :,
            indices
        ]
    )

    Xv = (
        X_val_full[
            :,
            indices
        ]
    )

    model = make_model()

    model.fit(
        Xtr,
        y_train
    )

    pred = model.predict(
        Xv
    )

    metrics = regression_metrics(
        y_val,
        pred
    )

    candidate_results.append(
        {
            "candidate":
                candidate["name"],
            "spatial_features":
                candidate["spatial_k"],
            "total_features":
                len(features),
            "validation_mae":
                metrics["mae"],
            "validation_rmse":
                metrics["rmse"],
            "validation_r2":
                metrics["r2"],
        }
    )

    print(
        f"{candidate['name']:40s} | "
        f"features={len(features):3d} | "
        f"VAL MAE={metrics['mae']:.8f} | "
        f"VAL R2={metrics['r2']:.6f}"
    )

comparison = pd.DataFrame(
    candidate_results
)

best_val_mae = float(
    comparison[
        "validation_mae"
    ].min()
)

comparison[
    "within_1pct_of_best"
] = (
    comparison[
        "validation_mae"
    ]
    <=
    best_val_mae
    * (
        1.0
        + SELECTION_TOLERANCE
    )
)

eligible = comparison[
    comparison[
        "within_1pct_of_best"
    ]
].copy()

if eligible.empty:

    raise RuntimeError(
        "No candidate is within the "
        "validation selection tolerance."
    )

eligible = eligible.sort_values(
    [
        "total_features",
        "validation_mae",
    ],
    ascending=[
        True,
        True,
    ],
)

selected_row = (
    eligible.iloc[0]
)

selected_candidate_name = (
    selected_row[
        "candidate"
    ]
)

selected_spatial_k = int(
    selected_row[
        "spatial_features"
    ]
)

selected_total_features = int(
    selected_row[
        "total_features"
    ]
)

selected_candidate = next(
    c
    for c in candidate_sets
    if c["name"]
    == selected_candidate_name
)

selected_features = (
    selected_candidate[
        "features"
    ]
)

comparison[
    "selection_best_validation"
] = (
    comparison[
        "validation_mae"
    ]
    == best_val_mae
)

comparison[
    "selected_final_candidate"
] = (
    comparison[
        "candidate"
    ]
    == selected_candidate_name
)

comparison.to_csv(
    OUT_COMPARISON,
    index=False
)


# ============================================================
# SAVE LOCKED FEATURES
# ============================================================

locked_spatial = (
    spatial_ranking[
        spatial_ranking[
            "feature"
        ].isin(
            selected_features
        )
    ]
    .copy()
)

locked_spatial = (
    locked_spatial[
        locked_spatial[
            "feature"
        ].isin(
            spatial_features
        )
    ]
)

locked_spatial.to_csv(
    OUT_LOCKED_SPATIAL,
    index=False
)

final_feature_table = pd.DataFrame({
    "feature_order":
        np.arange(
            1,
            len(
                selected_features
            ) + 1
        ),

    "feature":
        selected_features,

    "feature_group":
        [
            (
                "TEMPORAL"
                if f in TEMPORAL_FEATURES
                else "SPATIAL"
            )
            for f in selected_features
        ],
})

final_feature_table.to_csv(
    OUT_FINAL_FEATURES,
    index=False
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print(
    "\n[12/12] Locking feature set and "
    "performing final TEST evaluation..."
)

print(
    "\nSELECTED FEATURE SET"
)

print(
    "Candidate:",
    selected_candidate_name
)

print(
    "Spatial features:",
    selected_spatial_k
)

print(
    "Total features:",
    selected_total_features
)

print(
    "Best validation MAE:",
    f"{best_val_mae:.8f}"
)

print(
    "Selected validation MAE:",
    f"{selected_row['validation_mae']:.8f}"
)


selected_indices = [
    feature_index[f]
    for f in selected_features
]


# ------------------------------------------------------------
# TRAIN + VALIDATION
# ------------------------------------------------------------

train_val_df = pd.concat(
    [
        train_df,
        val_df,
    ],
    ignore_index=True
)

X_train_val = (
    train_val_df[
        feature_names
    ]
    .to_numpy(
        dtype=float
    )
)

y_train_val = (
    train_val_df[
        target_col
    ]
    .to_numpy(
        dtype=float
    )
)

X_train_val_selected = (
    X_train_val[
        :,
        selected_indices
    ]
)

X_test_selected = (
    X_test_full[
        :,
        selected_indices
    ]
)

final_model = make_model()

final_model.fit(
    X_train_val_selected,
    y_train_val
)

final_test_pred = (
    final_model.predict(
        X_test_selected
    )
)

final_test_metrics = (
    regression_metrics(
        y_test,
        final_test_pred
    )
)


# ============================================================
# PERSISTENCE
# ============================================================

persistence_index = feature_index[
    "crack_area_ratio"
]

persistence_pred = (
    X_test_full[
        :,
        persistence_index
    ]
)

persistence_metrics = (
    regression_metrics(
        y_test,
        persistence_pred
    )
)


# ============================================================
# TEMPORAL ONLY
# ============================================================

temporal_indices = [
    feature_index[f]
    for f in TEMPORAL_FEATURES
]

X_train_val_temporal = (
    X_train_val[
        :,
        temporal_indices
    ]
)

X_test_temporal = (
    X_test_full[
        :,
        temporal_indices
    ]
)

temporal_model = make_model()

temporal_model.fit(
    X_train_val_temporal,
    y_train_val
)

temporal_test_pred = (
    temporal_model.predict(
        X_test_temporal
    )
)

temporal_test_metrics = (
    regression_metrics(
        y_test,
        temporal_test_pred
    )
)


# ============================================================
# FULL 150-FEATURE FIXED BENCHMARK
# ============================================================

full_final_model = make_model()

full_final_model.fit(
    X_train_val,
    y_train_val
)

full_test_pred = (
    full_final_model.predict(
        X_test_full
    )
)

full_test_metrics = (
    regression_metrics(
        y_test,
        full_test_pred
    )
)


# ============================================================
# IMPROVEMENTS
# ============================================================

selected_vs_persistence = (
    1.0
    -
    (
        final_test_metrics["mae"]
        /
        persistence_metrics["mae"]
    )
) * 100.0

selected_vs_temporal = (
    1.0
    -
    (
        final_test_metrics["mae"]
        /
        temporal_test_metrics["mae"]
    )
) * 100.0

selected_vs_full = (
    1.0
    -
    (
        final_test_metrics["mae"]
        /
        full_test_metrics["mae"]
    )
) * 100.0


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "model":
            "PERSISTENCE",

        "feature_count":
            1,

        "spatial_feature_count":
            0,

        "test_mae":
            persistence_metrics["mae"],

        "test_rmse":
            persistence_metrics["rmse"],

        "test_r2":
            persistence_metrics["r2"],

        "used_for_selection":
            False,
    },

    {
        "model":
            "TEMPORAL_ONLY_18",

        "feature_count":
            len(TEMPORAL_FEATURES),

        "spatial_feature_count":
            0,

        "test_mae":
            temporal_test_metrics["mae"],

        "test_rmse":
            temporal_test_metrics["rmse"],

        "test_r2":
            temporal_test_metrics["r2"],

        "used_for_selection":
            False,
    },

    {
        "model":
            "LOCKED_REDUCED_FINAL",

        "feature_count":
            selected_total_features,

        "spatial_feature_count":
            selected_spatial_k,

        "test_mae":
            final_test_metrics["mae"],

        "test_rmse":
            final_test_metrics["rmse"],

        "test_r2":
            final_test_metrics["r2"],

        "used_for_selection":
            True,
    },

    {
        "model":
            "FULL_150_FIXED_BENCHMARK",

        "feature_count":
            len(feature_names),

        "spatial_feature_count":
            len(spatial_features),

        "test_mae":
            full_test_metrics["mae"],

        "test_rmse":
            full_test_metrics["rmse"],

        "test_r2":
            full_test_metrics["r2"],

        "used_for_selection":
            False,
    },
])

results.to_csv(
    OUT_RESULTS,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.22",

    "name":
        "Validation-Locked Final Feature Selection",

    "methodology": {

        "feature_ranking_source":
            "VALIDATION_ONLY",

        "feature_count_selection_source":
            "VALIDATION_ONLY",

        "final_test_evaluation":
            "HELD_OUT_TEST",

        "test_used_for_feature_ranking":
            False,

        "test_used_for_feature_selection":
            False,

        "selection_tolerance_fraction":
            SELECTION_TOLERANCE,

        "retraining_after_selection":
            "TRAIN_PLUS_VALIDATION",
    },

    "dataset": {

        "common_records":
            int(len(base)),

        "train_records":
            int(len(train_df)),

        "validation_records":
            int(len(val_df)),

        "test_records":
            int(len(test_df)),

        "temporal_features":
            int(len(TEMPORAL_FEATURES)),

        "spatial_features":
            int(len(spatial_features)),

        "total_features":
            int(len(feature_names)),
    },

    "validation_selection": {

        "best_validation_mae":
            float(best_val_mae),

        "selected_candidate":
            selected_candidate_name,

        "selected_spatial_features":
            int(selected_spatial_k),

        "selected_total_features":
            int(
                selected_total_features
            ),

        "selected_validation_mae":
            float(
                selected_row[
                    "validation_mae"
                ]
            ),

        "selected_validation_rmse":
            float(
                selected_row[
                    "validation_rmse"
                ]
            ),

        "selected_validation_r2":
            float(
                selected_row[
                    "validation_r2"
                ]
            ),
    },

    "final_test": {

        "locked_reduced_mae":
            final_test_metrics["mae"],

        "locked_reduced_rmse":
            final_test_metrics["rmse"],

        "locked_reduced_r2":
            final_test_metrics["r2"],

        "persistence_mae":
            persistence_metrics["mae"],

        "persistence_rmse":
            persistence_metrics["rmse"],

        "persistence_r2":
            persistence_metrics["r2"],

        "temporal_only_mae":
            temporal_test_metrics["mae"],

        "temporal_only_rmse":
            temporal_test_metrics["rmse"],

        "temporal_only_r2":
            temporal_test_metrics["r2"],

        "full_150_mae":
            full_test_metrics["mae"],

        "full_150_rmse":
            full_test_metrics["rmse"],

        "full_150_r2":
            full_test_metrics["r2"],
    },

    "improvements_percent": {

        "locked_vs_persistence_mae":
            float(
                selected_vs_persistence
            ),

        "locked_vs_temporal_mae":
            float(
                selected_vs_temporal
            ),

        "locked_vs_full_mae":
            float(
                selected_vs_full
            ),
    },

    "output_files": {

        "validation_ranking":
            str(OUT_RANKING),

        "validation_comparison":
            str(OUT_COMPARISON),

        "locked_spatial_features":
            str(OUT_LOCKED_SPATIAL),

        "final_feature_set":
            str(OUT_FINAL_FEATURES),

        "final_results":
            str(OUT_RESULTS),

        "summary":
            str(OUT_SUMMARY),
    },

    "status":
        "PASS",
}


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STAGE 3.22 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nVALIDATION-LOCKED SELECTION"
)

print(
    "Best validation MAE:",
    f"{best_val_mae:.8f}"
)

print(
    "Selected candidate:",
    selected_candidate_name
)

print(
    "Selected spatial features:",
    selected_spatial_k
)

print(
    "Selected total features:",
    selected_total_features
)

print(
    "Selected validation MAE:",
    f"{selected_row['validation_mae']:.8f}"
)

print(
    "\nFINAL HELD-OUT TEST"
)

print(
    "Persistence MAE:",
    f"{persistence_metrics['mae']:.8f}"
)

print(
    "Temporal-only MAE:",
    f"{temporal_test_metrics['mae']:.8f}"
)

print(
    "Locked reduced MAE:",
    f"{final_test_metrics['mae']:.8f}"
)

print(
    "Locked reduced RMSE:",
    f"{final_test_metrics['rmse']:.8f}"
)

print(
    "Locked reduced R2:",
    f"{final_test_metrics['r2']:.8f}"
)

print(
    "Full 150 benchmark MAE:",
    f"{full_test_metrics['mae']:.8f}"
)

print(
    "Full 150 benchmark R2:",
    f"{full_test_metrics['r2']:.8f}"
)

print(
    "\nIMPROVEMENT"
)

print(
    "Locked vs persistence:",
    f"{selected_vs_persistence:.4f}%"
)

print(
    "Locked vs temporal-only:",
    f"{selected_vs_temporal:.4f}%"
)

print(
    "Locked vs full 150:",
    f"{selected_vs_full:.4f}%"
)

print(
    "\nOutputs:"
)

for path in [
    OUT_RANKING,
    OUT_COMPARISON,
    OUT_LOCKED_SPATIAL,
    OUT_FINAL_FEATURES,
    OUT_RESULTS,
    OUT_SUMMARY,
]:

    print(
        path
    )

print(
    "\nFinal status: PASS"
)

print(
    "=" * 80
)