import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

SPATIAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_features.csv"
)

TEMPORAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

TARGET_FILE = os.path.join(
    AUDIT_DIR,
    "stage_1_3_prediction_targets.csv"
)

RESULT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_3b_common_subset_results.csv"
)

SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_3b_common_subset_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "loss": "huber",
    "random_state": RANDOM_STATE
}

KEY_COLUMNS = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"
]


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def unique_preserve_order(values):
    return list(dict.fromkeys(values))


def find_target_column(df):

    preferred = [
        "target_crack_area_ratio",
        "target_value",
        "target_damage",
        "target_ratio",
        "target"
    ]

    for col in preferred:
        if col in df.columns:
            return col

    candidates = []

    for col in df.columns:

        lower = col.lower()

        if (
            "target" in lower
            or "future" in lower
        ):

            if col in [
                "target_current_frame",
                "target_future_frame",
                "target_frame",
                "target_horizon"
            ]:
                continue

            if pd.api.types.is_numeric_dtype(
                df[col]
            ):
                candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    preferred_candidates = [
        col
        for col in candidates
        if any(
            token in col.lower()
            for token in [
                "crack",
                "damage",
                "ratio",
                "area"
            ]
        )
    ]

    if len(preferred_candidates) == 1:
        return preferred_candidates[0]

    return None


def prepare_targets(targets):

    print(
        "\nPreparing Stage 1.3 targets..."
    )

    if "horizon" in targets.columns:

        horizon = pd.to_numeric(
            targets["horizon"],
            errors="coerce"
        )

        targets = targets[
            horizon == 1
        ].copy()

    else:

        targets = targets.copy()

    target_column = find_target_column(
        targets
    )

    print(
        "Detected target-value column:",
        target_column
    )

    if target_column is None:

        raise RuntimeError(
            "Could not identify target value column."
        )

    if "current_frame" in targets.columns:

        targets["current_frame"] = pd.to_numeric(
            targets["current_frame"],
            errors="coerce"
        )

    elif "target_current_frame" in targets.columns:

        targets["current_frame"] = pd.to_numeric(
            targets["target_current_frame"],
            errors="coerce"
        )

    elif "frame_number" in targets.columns:

        targets["current_frame"] = pd.to_numeric(
            targets["frame_number"],
            errors="coerce"
        )

    else:

        raise RuntimeError(
            "Could not identify current frame."
        )

    if "future_frame" in targets.columns:

        targets["future_frame"] = pd.to_numeric(
            targets["future_frame"],
            errors="coerce"
        )

    elif "target_future_frame" in targets.columns:

        targets["future_frame"] = pd.to_numeric(
            targets["target_future_frame"],
            errors="coerce"
        )

    else:

        targets["future_frame"] = (
            targets["current_frame"] + 1
        )

    targets["target_t_plus_1"] = pd.to_numeric(
        targets[target_column],
        errors="coerce"
    )

    required = KEY_COLUMNS + [
        "target_t_plus_1"
    ]

    missing = [
        col
        for col in required
        if col not in targets.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing target columns:\n"
            + "\n".join(missing)
        )

    targets = targets[
        required
    ].dropna()

    targets["current_frame"] = (
        targets["current_frame"]
        .astype(int)
    )

    targets["future_frame"] = (
        targets["future_frame"]
        .astype(int)
    )

    targets = targets.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="first"
    )

    print(
        "Primary t+1 target records:",
        len(targets)
    )

    return targets


def spatial_groups(columns):

    groups = {
        "lag": [],
        "change": [],
        "acceleration": [],
        "rolling": [],
        "trend": []
    }

    for col in columns:

        lower = col.lower()

        if "lag" in lower:
            groups["lag"].append(col)

        elif "delta" in lower:
            groups["change"].append(col)

        elif "accel" in lower:
            groups["acceleration"].append(col)

        elif "rolling" in lower:
            groups["rolling"].append(col)

        elif (
            "recent" in lower
            or "slope" in lower
            or "velocity" in lower
        ):
            groups["trend"].append(col)

    return groups


def evaluate_model(
    train_df,
    test_df,
    features,
    model_name
):

    usable = [
        col
        for col in features
        if col in train_df.columns
        and col in test_df.columns
    ]

    train_complete = train_df.dropna(
        subset=usable + [
            "target_t_plus_1"
        ]
    )

    test_complete = test_df.dropna(
        subset=usable + [
            "target_t_plus_1"
        ]
    )

    if len(usable) == 0:
        raise RuntimeError(
            f"No features available for {model_name}"
        )

    model = GradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        train_complete[usable],
        train_complete[
            "target_t_plus_1"
        ]
    )

    predictions = model.predict(
        test_complete[usable]
    )

    y_true = test_complete[
        "target_t_plus_1"
    ].values

    return {
        "model": model_name,
        "feature_count": len(usable),
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions
            )
        ),
        "rmse": rmse(
            y_true,
            predictions
        ),
        "r2": float(
            r2_score(
                y_true,
                predictions
            )
        ),
        "n_test_records": int(
            len(test_complete)
        )
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 3.3B - COMMON-SUBSET VALIDATION")
print("=" * 70)


# ============================================================
# FILE CHECKS
# ============================================================

for path, label in [

    (
        SPATIAL_FILE,
        "Stage 3.1 spatial features"
    ),

    (
        TEMPORAL_FILE,
        "Stage 2.2 temporal features"
    ),

    (
        TARGET_FILE,
        "Stage 1.3 targets"
    )
]:

    print(
        f"\nChecking {label}..."
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    print(
        f"{label}: FOUND"
    )


# ============================================================
# LOAD
# ============================================================

print(
    "\nLoading spatial features..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial records:",
    len(spatial)
)


print(
    "\nLoading temporal features..."
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
)


print(
    "\nLoading targets..."
)

targets_raw = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets_raw)
)


# ============================================================
# TARGETS
# ============================================================

targets = prepare_targets(
    targets_raw
)


# ============================================================
# SPATIAL FRAME PREPARATION
# ============================================================

spatial["current_frame"] = pd.to_numeric(
    spatial["current_frame"],
    errors="coerce"
)

spatial["future_frame"] = pd.to_numeric(
    spatial["future_frame"],
    errors="coerce"
)

spatial = spatial.dropna(
    subset=[
        "current_frame",
        "future_frame"
    ]
)

spatial["current_frame"] = (
    spatial["current_frame"]
    .astype(int)
)

spatial["future_frame"] = (
    spatial["future_frame"]
    .astype(int)
)


# ============================================================
# REMOVE TARGET FROM SPATIAL
# ============================================================

target_like = []

for col in spatial.columns:

    lower = col.lower()

    if (
        "target" in lower
        or lower.startswith("future_")
    ):

        if col not in KEY_COLUMNS:

            target_like.append(col)

if target_like:

    print(
        "\nRemoving target/future columns:"
    )

    for col in target_like:
        print(
            " ",
            col
        )

    spatial = spatial.drop(
        columns=target_like
    )


# ============================================================
# TEMPORAL FRAME PREPARATION
# ============================================================

if "frame_number" not in temporal.columns:

    raise RuntimeError(
        "Stage 2.2 is missing frame_number."
    )

temporal["current_frame"] = pd.to_numeric(
    temporal["frame_number"],
    errors="coerce"
)

temporal["future_frame"] = (
    temporal["current_frame"] + 1
)

temporal = temporal.dropna(
    subset=[
        "current_frame",
        "future_frame"
    ]
)

temporal["current_frame"] = (
    temporal["current_frame"]
    .astype(int)
)

temporal["future_frame"] = (
    temporal["future_frame"]
    .astype(int)
)


# ============================================================
# CURRENT FEATURES
# ============================================================

current_features = [
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "annotation_shape_count",
    "image_area_px",
    "current_crack_area_ratio"
]

current_features = [
    col
    for col in current_features
    if col in temporal.columns
]

current_features = unique_preserve_order(
    current_features
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

temporal_features = []

for col in temporal.columns:

    if col in [
        "sequence",
        "component",
        "split",
        "frame_number",
        "current_frame",
        "future_frame"
    ]:
        continue

    lower = col.lower()

    if (
        lower.startswith("lag1_")
        or lower.startswith("lag2_")
        or lower.startswith("lag3_")
        or lower.startswith("delta1_")
        or lower.startswith("delta2_")
        or lower.startswith("accel_")
        or lower.startswith("rolling3_")
        or lower.startswith("recent_")
    ):

        temporal_features.append(
            col
        )

temporal_features = unique_preserve_order(
    temporal_features
)


# ============================================================
# SPATIAL HISTORICAL FEATURES
# ============================================================

future_dependent_spatial = [
    "matched_crack_count",
    "persistent_fraction",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score",
    "matched_fraction_current",
    "unmatched_current_crack_count",
    "unmatched_current_fraction",
    "spatial_match_quality",
    "spatial_instability",
    "track_continuity_quality",
    "continuity_score"
]

spatial_features = []

for col in spatial.columns:

    if not col.startswith(
        "spatial_"
    ):
        continue

    if col in future_dependent_spatial:
        continue

    lower = col.lower()

    if any(
        token in lower
        for token in [
            "lag",
            "delta",
            "accel",
            "rolling",
            "recent",
            "slope",
            "velocity"
        ]
    ):

        spatial_features.append(
            col
        )

spatial_features = unique_preserve_order(
    spatial_features
)

groups = spatial_groups(
    spatial_features
)


# ============================================================
# MERGE
# ============================================================

temporal_merge_columns = (
    current_features
    + temporal_features
)

temporal_merge_columns = (
    unique_preserve_order(
        temporal_merge_columns
    )
)

temporal_subset = temporal[
    KEY_COLUMNS
    + temporal_merge_columns
].copy()

print(
    "\nMerging spatial and temporal features..."
)

merged = spatial.merge(
    temporal_subset,
    on=KEY_COLUMNS,
    how="inner",
    suffixes=(
        "",
        "_temporal"
    )
)

print(
    "Merged records:",
    len(merged)
)


# ============================================================
# TARGET MERGE
# ============================================================

merged = merged.merge(
    targets[
        KEY_COLUMNS
        + [
            "target_t_plus_1"
        ]
    ],
    on=KEY_COLUMNS,
    how="inner"
)

print(
    "Records after target merge:",
    len(merged)
)


# ============================================================
# CHECK TARGET
# ============================================================

if "target_t_plus_1" not in merged.columns:

    raise RuntimeError(
        "Target was not successfully merged."
    )


# ============================================================
# SPLIT
# ============================================================

train = merged[
    merged["split"] == "train"
].copy()

test = merged[
    merged["split"] == "test"
].copy()

print(
    "\nOriginal merged split:"
)

print(
    "TRAIN:",
    len(train)
)

print(
    "TEST:",
    len(test)
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

model_features = {

    "CURRENT_ONLY":
        current_features,

    "CURRENT_PLUS_TEMPORAL":
        current_features
        + temporal_features,

    "TEMPORAL_PLUS_SPATIAL_CHANGE":
        current_features
        + temporal_features
        + groups["change"],

    "TEMPORAL_PLUS_SPATIAL_TREND":
        current_features
        + temporal_features
        + groups["trend"],

    "FULL_SPATIAL_TEMPORAL":
        current_features
        + temporal_features
        + spatial_features
}


for name in model_features:

    model_features[name] = (
        unique_preserve_order(
            model_features[name]
        )
    )


# ============================================================
# DETERMINE COMMON COMPLETE TEST SUBSET
# ============================================================

print("\n")
print("=" * 70)
print("DETERMINING COMMON COMPLETE TEST SUBSET")
print("=" * 70)

required_for_common = []

for features in model_features.values():

    required_for_common.extend(
        features
    )

required_for_common = (
    unique_preserve_order(
        required_for_common
    )
)

required_for_common.append(
    "target_t_plus_1"
)

required_for_common = (
    unique_preserve_order(
        required_for_common
    )
)

missing_required = [
    col
    for col in required_for_common
    if col not in test.columns
]

if missing_required:

    print(
        "\nMissing columns:"
    )

    for col in missing_required:
        print(
            " ",
            col
        )

    raise RuntimeError(
        "Required common-subset columns are missing."
    )

common_test = test.dropna(
    subset=required_for_common
).copy()

print(
    "\nCommon test records:",
    len(common_test)
)

if len(common_test) == 0:

    raise RuntimeError(
        "No common complete test records."
    )


# ============================================================
# COMMON TRAIN SUBSET
#
# Training is also standardized so that all models are trained
# using the same training records. This makes the comparison
# cleaner.
# ============================================================

common_train = train.dropna(
    subset=required_for_common
).copy()

print(
    "Common training records:",
    len(common_train)
)


# ============================================================
# CHECK SEQUENCE / COMPONENT INTEGRITY
# ============================================================

sequence_conflicts = int(
    (
        merged.groupby(
            "sequence"
        )["split"].nunique()
        > 1
    ).sum()
)

component_conflicts = int(
    (
        merged.groupby(
            "component"
        )["split"].nunique()
        > 1
    ).sum()
)

print(
    "\nSequence split conflicts:",
    sequence_conflicts
)

print(
    "Component split conflicts:",
    component_conflicts
)

if sequence_conflicts > 0:

    raise RuntimeError(
        "Sequence leakage detected."
    )

if component_conflicts > 0:

    raise RuntimeError(
        "Component leakage detected."
    )


# ============================================================
# TRAIN ON COMMON SUBSET
# ============================================================

print("\n")
print("=" * 70)
print("COMMON-SUBSET MODEL EVALUATION")
print("=" * 70)

results = []

for model_name, features in (
    model_features.items()
):

    print(
        f"\nModel: {model_name}"
    )

    print(
        "Feature count:",
        len(features)
    )

    model = GradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        common_train[
            features
        ],
        common_train[
            "target_t_plus_1"
        ]
    )

    predictions = model.predict(
        common_test[
            features
        ]
    )

    y_true = common_test[
        "target_t_plus_1"
    ].values

    model_mae = mean_absolute_error(
        y_true,
        predictions
    )

    model_rmse = rmse(
        y_true,
        predictions
    )

    model_r2 = r2_score(
        y_true,
        predictions
    )

    results.append({

        "model":
            model_name,

        "feature_count":
            len(features),

        "mae":
            float(model_mae),

        "rmse":
            float(model_rmse),

        "r2":
            float(model_r2),

        "n_train_records":
            int(len(common_train)),

        "n_test_records":
            int(len(common_test))
    })


# ============================================================
# PERSISTENCE
# ============================================================

print(
    "\nCalculating common-subset persistence..."
)

if "current_crack_area_ratio" in common_test.columns:

    persistence_column = (
        "current_crack_area_ratio"
    )

else:

    persistence_column = (
        "crack_area_ratio"
    )

persistence_pred = common_test[
    persistence_column
].values

persistence_true = common_test[
    "target_t_plus_1"
].values

persistence_mae = mean_absolute_error(
    persistence_true,
    persistence_pred
)

persistence_rmse = rmse(
    persistence_true,
    persistence_pred
)

persistence_r2 = r2_score(
    persistence_true,
    persistence_pred
)

results.append({

    "model":
        "PERSISTENCE",

    "feature_count":
        0,

    "mae":
        float(persistence_mae),

    "rmse":
        float(persistence_rmse),

    "r2":
        float(persistence_r2),

    "n_train_records":
        int(len(common_train)),

    "n_test_records":
        int(len(common_test))
})


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df[
    "mae_improvement_vs_persistence_pct"
] = (
    1
    -
    results_df["mae"]
    /
    persistence_mae
) * 100.0

results_df = results_df.sort_values(
    "mae"
).reset_index(
    drop=True
)


# ============================================================
# SPATIAL IMPROVEMENT
# ============================================================

temporal_result = results_df[
    results_df["model"]
    == "CURRENT_PLUS_TEMPORAL"
]

full_result = results_df[
    results_df["model"]
    == "FULL_SPATIAL_TEMPORAL"
]

change_result = results_df[
    results_df["model"]
    == "TEMPORAL_PLUS_SPATIAL_CHANGE"
]

if (
    len(temporal_result) == 1
    and len(full_result) == 1
):

    temporal_mae = float(
        temporal_result.iloc[0][
            "mae"
        ]
    )

    full_mae = float(
        full_result.iloc[0][
            "mae"
        ]
    )

    spatial_improvement = (
        1
        -
        full_mae
        /
        temporal_mae
    ) * 100.0

else:

    temporal_mae = None
    full_mae = None
    spatial_improvement = None


if len(change_result) == 1:

    change_mae = float(
        change_result.iloc[0][
            "mae"
        ]
    )

else:

    change_mae = None


# ============================================================
# DECISION
# ============================================================

if (
    spatial_improvement is not None
    and spatial_improvement > 0
):

    decision = (
        "COMMON_SUBSET_SPATIAL_SIGNAL_SUPPORTED"
    )

else:

    decision = (
        "COMMON_SUBSET_SPATIAL_SIGNAL_NOT_SUPPORTED"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


summary = {

    "stage":
        "3.3B",

    "status":
        "PASS",

    "common_train_records":
        int(len(common_train)),

    "common_test_records":
        int(len(common_test)),

    "current_feature_count":
        int(len(current_features)),

    "temporal_feature_count":
        int(len(temporal_features)),

    "spatial_feature_count":
        int(len(spatial_features)),

    "spatial_group_counts": {

        name:
            int(len(cols))

        for name, cols
        in groups.items()
    },

    "persistence": {

        "mae":
            float(persistence_mae),

        "rmse":
            float(persistence_rmse),

        "r2":
            float(persistence_r2)
    },

    "temporal_model": {

        "mae":
            temporal_mae
    },

    "full_spatial_temporal": {

        "mae":
            full_mae
    },

    "spatial_change_model": {

        "mae":
            change_mae
    },

    "spatial_improvement_vs_temporal_pct":
        spatial_improvement,

    "decision":
        decision,

    "sequence_split_conflicts":
        sequence_conflicts,

    "component_split_conflicts":
        component_conflicts,

    "output":
        RESULT_FILE
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.3B COMMON-SUBSET RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("STAGE 3.3B SUMMARY")
print("=" * 70)

print(
    "Common training records:",
    len(common_train)
)

print(
    "Common test records:",
    len(common_test)
)

print(
    "Persistence TEST MAE:",
    f"{persistence_mae:.8f}"
)

if temporal_mae is not None:

    print(
        "Temporal TEST MAE:",
        f"{temporal_mae:.8f}"
    )

if full_mae is not None:

    print(
        "Full spatial + temporal TEST MAE:",
        f"{full_mae:.8f}"
    )

if change_mae is not None:

    print(
        "Spatial-change TEST MAE:",
        f"{change_mae:.8f}"
    )

if spatial_improvement is not None:

    print(
        "Spatial improvement vs temporal:",
        f"{spatial_improvement:.4f}%"
    )

print(
    "\nDecision:",
    decision
)

print(
    "\nResults:"
)

print(
    RESULT_FILE
)

print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nFINAL STATUS: PASS"
)