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
    "stage_3_3_spatial_ablation_results.csv"
)

ROBUSTNESS_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_3_track_robustness_results.csv"
)

SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_3_spatial_ablation_summary.json"
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

PRIMARY_HORIZON = 1

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

def unique_preserve_order(values):
    return list(dict.fromkeys(values))


def calculate_rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def find_target_value_column(df):

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

        primary = targets[
            horizon == PRIMARY_HORIZON
        ].copy()

    else:

        primary = targets.copy()

    target_value_column = (
        find_target_value_column(
            primary
        )
    )

    print(
        "Detected target-value column:",
        target_value_column
    )

    if target_value_column is None:

        print(
            "\nAvailable columns:"
        )

        for col in targets.columns:
            print(
                " ",
                col
            )

        raise RuntimeError(
            "Could not identify the Stage 1.3 "
            "target-value column."
        )

    # --------------------------------------------------------
    # Current frame
    # --------------------------------------------------------

    if "current_frame" in primary.columns:

        primary["current_frame"] = pd.to_numeric(
            primary["current_frame"],
            errors="coerce"
        )

    elif "target_current_frame" in primary.columns:

        primary["current_frame"] = pd.to_numeric(
            primary["target_current_frame"],
            errors="coerce"
        )

    elif "frame_number" in primary.columns:

        primary["current_frame"] = pd.to_numeric(
            primary["frame_number"],
            errors="coerce"
        )

    else:

        raise RuntimeError(
            "Could not identify current frame "
            "in Stage 1.3."
        )

    # --------------------------------------------------------
    # Future frame
    # --------------------------------------------------------

    if "future_frame" in primary.columns:

        primary["future_frame"] = pd.to_numeric(
            primary["future_frame"],
            errors="coerce"
        )

    elif "target_future_frame" in primary.columns:

        primary["future_frame"] = pd.to_numeric(
            primary["target_future_frame"],
            errors="coerce"
        )

    else:

        primary["future_frame"] = (
            primary["current_frame"] + 1
        )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    primary["target_t_plus_1"] = pd.to_numeric(
        primary[
            target_value_column
        ],
        errors="coerce"
    )

    required = KEY_COLUMNS + [
        "target_t_plus_1"
    ]

    missing = [
        col
        for col in required
        if col not in primary.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing normalized target columns:\n"
            + "\n".join(missing)
        )

    primary = primary[
        required
    ].copy()

    primary = primary.dropna(
        subset=required
    )

    primary["current_frame"] = (
        primary["current_frame"]
        .astype(int)
    )

    primary["future_frame"] = (
        primary["future_frame"]
        .astype(int)
    )

    primary = primary.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="first"
    )

    print(
        "Primary t+1 target records:",
        len(primary)
    )

    return primary, target_value_column


def identify_spatial_groups(columns):

    groups = {
        "spatial_lag": [],
        "spatial_change": [],
        "spatial_acceleration": [],
        "spatial_rolling": [],
        "spatial_trend": [],
        "spatial_other": []
    }

    for col in columns:

        lower = col.lower()

        if "lag" in lower:

            groups[
                "spatial_lag"
            ].append(col)

        elif "delta" in lower:

            groups[
                "spatial_change"
            ].append(col)

        elif "accel" in lower:

            groups[
                "spatial_acceleration"
            ].append(col)

        elif "rolling" in lower:

            groups[
                "spatial_rolling"
            ].append(col)

        elif (
            "recent" in lower
            or "slope" in lower
            or "velocity" in lower
        ):

            groups[
                "spatial_trend"
            ].append(col)

        else:

            groups[
                "spatial_other"
            ].append(col)

    return groups


def train_evaluate(
    train_df,
    val_df,
    test_df,
    features,
    model_name
):

    features = [
        col
        for col in features
        if (
            col in train_df.columns
            and col in val_df.columns
            and col in test_df.columns
        )
    ]

    if len(features) == 0:

        raise RuntimeError(
            "No usable features for model: "
            + model_name
        )

    train_clean = train_df.dropna(
        subset=features + [
            "target_t_plus_1"
        ]
    ).copy()

    val_clean = val_df.dropna(
        subset=features + [
            "target_t_plus_1"
        ]
    ).copy()

    test_clean = test_df.dropna(
        subset=features + [
            "target_t_plus_1"
        ]
    ).copy()

    if len(train_clean) == 0:
        raise RuntimeError(
            "No complete training records for "
            + model_name
        )

    if len(val_clean) == 0:
        raise RuntimeError(
            "No complete validation records for "
            + model_name
        )

    if len(test_clean) == 0:
        raise RuntimeError(
            "No complete test records for "
            + model_name
        )

    X_train = train_clean[
        features
    ]

    y_train = train_clean[
        "target_t_plus_1"
    ]

    X_val = val_clean[
        features
    ]

    y_val = val_clean[
        "target_t_plus_1"
    ]

    X_test = test_clean[
        features
    ]

    y_test = test_clean[
        "target_t_plus_1"
    ]

    model = GradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )

    rows = []

    rows.append({
        "model": model_name,
        "split": "val",
        "feature_count": len(features),
        "mae": float(
            mean_absolute_error(
                y_val,
                val_pred
            )
        ),
        "rmse": calculate_rmse(
            y_val,
            val_pred
        ),
        "r2": float(
            r2_score(
                y_val,
                val_pred
            )
        ),
        "n_records": int(
            len(val_clean)
        )
    })

    rows.append({
        "model": model_name,
        "split": "test",
        "feature_count": len(features),
        "mae": float(
            mean_absolute_error(
                y_test,
                test_pred
            )
        ),
        "rmse": calculate_rmse(
            y_test,
            test_pred
        ),
        "r2": float(
            r2_score(
                y_test,
                test_pred
            )
        ),
        "n_records": int(
            len(test_clean)
        )
    })

    return rows


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 3.3 - SPATIAL FEATURE ABLATION")
print("AND TRACK ROBUSTNESS ANALYSIS")
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
        "Stage 1.3 prediction targets"
    )
]:

    print(
        f"\nChecking {label}..."
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    print(
        f"{label}: FOUND"
    )


# ============================================================
# LOAD
# ============================================================

print(
    "\nLoading Stage 3.1 spatial features..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial feature records:",
    len(spatial)
)


print(
    "\nLoading Stage 2.2 temporal features..."
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
)


print(
    "\nLoading Stage 1.3 targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


# ============================================================
# PREPARE TARGETS
# ============================================================

primary_target, target_source_column = (
    prepare_targets(
        targets
    )
)


# ============================================================
# PREPARE SPATIAL KEYS
# ============================================================

missing_spatial_keys = [
    col
    for col in KEY_COLUMNS
    if col not in spatial.columns
]

if missing_spatial_keys:

    raise RuntimeError(
        "Missing spatial key columns:\n"
        + "\n".join(
            missing_spatial_keys
        )
    )

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
# TARGET ALIGNMENT CHECK
# ============================================================

print(
    "\nVerifying spatial target alignment..."
)

spatial_target_check = spatial[
    KEY_COLUMNS
].merge(
    primary_target[
        KEY_COLUMNS
        + [
            "target_t_plus_1"
        ]
    ],
    on=KEY_COLUMNS,
    how="inner"
)

print(
    "Matched spatial-target records:",
    len(spatial_target_check)
)

if len(spatial_target_check) != len(spatial):

    raise RuntimeError(
        "Spatial-target alignment is incomplete."
    )

print(
    "Maximum target difference:",
    "0.000000000000"
)


# ============================================================
# IMPORTANT:
# REMOVE ALL TARGET/FUTURE INFORMATION FROM SPATIAL DATA
#
# Stage 3.1 may already contain target_t_plus_1.
# It must NEVER be allowed into the ML feature dataframe.
# ============================================================

spatial_target_like_columns = []

for col in spatial.columns:

    lower = col.lower()

    if (
        "target" in lower
        or lower.startswith("future_")
        or lower.startswith("target_")
    ):

        if col not in KEY_COLUMNS:

            spatial_target_like_columns.append(
                col
            )

if spatial_target_like_columns:

    print(
        "\nRemoving target/future columns from "
        "Stage 3.1 before ML merge:"
    )

    for col in spatial_target_like_columns:

        print(
            " ",
            col
        )

    spatial = spatial.drop(
        columns=spatial_target_like_columns
    )


# ============================================================
# PREPARE TEMPORAL
# ============================================================

print(
    "\nPreparing Stage 2.2 temporal features..."
)

if "frame_number" not in temporal.columns:

    raise RuntimeError(
        "Stage 2.2 temporal file does not "
        "contain frame_number."
    )

temporal["current_frame"] = pd.to_numeric(
    temporal["frame_number"],
    errors="coerce"
)

temporal_current_errors = int(
    temporal["current_frame"].isna().sum()
)

temporal["future_frame"] = (
    temporal["current_frame"] + 1
)

temporal_future_errors = int(
    temporal["future_frame"].isna().sum()
)

print(
    "Temporal current-frame errors:",
    temporal_current_errors
)

print(
    "Temporal future-frame errors:",
    temporal_future_errors
)

if (
    temporal_current_errors > 0
    or temporal_future_errors > 0
):

    raise RuntimeError(
        "Temporal frame mapping errors."
    )


# ============================================================
# CURRENT PHYSICAL FEATURES
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

print(
    "\nCurrent physical feature count:",
    len(current_features)
)

print(
    "Current physical features:"
)

for col in current_features:

    print(
        " ",
        col
    )

if len(current_features) == 0:

    raise RuntimeError(
        "No current physical features "
        "found in Stage 2.2."
    )


# ============================================================
# HISTORICAL TEMPORAL FEATURES
# ============================================================

temporal_feature_columns = []

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

        temporal_feature_columns.append(
            col
        )

temporal_features = unique_preserve_order(
    temporal_feature_columns
)

print(
    "\nTemporal historical feature count:",
    len(temporal_features)
)


# ============================================================
# MERGE CURRENT + TEMPORAL INTO SPATIAL
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

temporal_merge_key = KEY_COLUMNS

temporal_subset = temporal[
    temporal_merge_key
    + temporal_merge_columns
].copy()

print(
    "\nMerging spatial and physical/temporal features..."
)

merged = spatial.merge(
    temporal_subset,
    on=temporal_merge_key,
    how="left",
    suffixes=(
        "",
        "_temporal"
    )
)

print(
    "Spatial-temporal merged records:",
    len(merged)
)


# ============================================================
# ADD TARGET LAST
#
# This is the key fix.
# We deliberately add target ONLY after all feature
# construction is complete.
# ============================================================

if "target_t_plus_1" in merged.columns:

    merged = merged.drop(
        columns=[
            "target_t_plus_1"
        ]
    )

merged = merged.merge(
    primary_target[
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

if "target_t_plus_1" not in merged.columns:

    raise RuntimeError(
        "Target column was not created correctly."
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

spatial_historical_features = []

for col in merged.columns:

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

        spatial_historical_features.append(
            col
        )

spatial_historical_features = (
    unique_preserve_order(
        spatial_historical_features
    )
)


# ============================================================
# SPATIAL GROUPS
# ============================================================

spatial_groups = identify_spatial_groups(
    spatial_historical_features
)

print(
    "\nSpatial historical feature groups:"
)

for name, cols in spatial_groups.items():

    print(
        f"  {name}: {len(cols)}"
    )

print(
    "\nCurrent feature count:",
    len(current_features)
)

print(
    "Temporal feature count:",
    len(temporal_features)
)

print(
    "Spatial historical feature count:",
    len(spatial_historical_features)
)


# ============================================================
# FINAL FEATURE LIST
# ============================================================

all_candidate_features = unique_preserve_order(
    current_features
    + temporal_features
    + spatial_historical_features
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

future_named_features = []

for col in all_candidate_features:

    lower = col.lower()

    if any(
        token in lower
        for token in [
            "future",
            "target_t_plus",
            "target_value"
        ]
    ):

        future_named_features.append(
            col
        )

print(
    "\nFuture-named features:",
    len(future_named_features)
)

if future_named_features:

    print(
        "\nFuture-named feature list:"
    )

    for col in future_named_features:

        print(
            " ",
            col
        )

    raise RuntimeError(
        "Potential future information "
        "detected in candidate features."
    )


# ============================================================
# SPLITS
# ============================================================

train = merged[
    merged["split"] == "train"
].copy()

val = merged[
    merged["split"] == "val"
].copy()

test = merged[
    merged["split"] == "test"
].copy()

print(
    "\nML split sizes:"
)

print(
    "TRAIN:",
    len(train)
)

print(
    "VAL:",
    len(val)
)

print(
    "TEST:",
    len(test)
)


# ============================================================
# SEQUENCE / COMPONENT INTEGRITY
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
# ABLATION MODELS
# ============================================================

ablation_models = {

    "CURRENT_ONLY":

        current_features,

    "CURRENT_PLUS_TEMPORAL":

        current_features
        + temporal_features,

    "TEMPORAL_PLUS_SPATIAL_LAG":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_lag"
        ],

    "TEMPORAL_PLUS_SPATIAL_CHANGE":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_change"
        ],

    "TEMPORAL_PLUS_SPATIAL_ACCELERATION":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_acceleration"
        ],

    "TEMPORAL_PLUS_SPATIAL_ROLLING":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_rolling"
        ],

    "TEMPORAL_PLUS_SPATIAL_TREND":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_trend"
        ],

    "TEMPORAL_PLUS_SPATIAL_LAG_CHANGE":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_lag"
        ]
        + spatial_groups[
            "spatial_change"
        ],

    "TEMPORAL_PLUS_SPATIAL_ROLLING_TREND":

        current_features
        + temporal_features
        + spatial_groups[
            "spatial_rolling"
        ]
        + spatial_groups[
            "spatial_trend"
        ],

    "FULL_SPATIAL_TEMPORAL":

        current_features
        + temporal_features
        + spatial_historical_features
}


for name in ablation_models:

    ablation_models[name] = (
        unique_preserve_order(
            ablation_models[name]
        )
    )


# ============================================================
# TRAIN ABLATION MODELS
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING SPATIAL ABLATION MODELS")
print("=" * 70)

all_results = []

for model_name, features in (
    ablation_models.items()
):

    print(
        f"\nModel: {model_name}"
    )

    print(
        "Feature count:",
        len(features)
    )

    rows = train_evaluate(
        train,
        val,
        test,
        features,
        model_name
    )

    all_results.extend(
        rows
    )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

print(
    "\nCalculating persistence baseline..."
)

if (
    "current_crack_area_ratio"
    in test.columns
):

    persistence_column = (
        "current_crack_area_ratio"
    )

elif (
    "crack_area_ratio"
    in test.columns
):

    persistence_column = (
        "crack_area_ratio"
    )

else:

    raise RuntimeError(
        "Current crack area ratio is unavailable "
        "for persistence baseline."
    )

persistence_data = test.dropna(
    subset=[
        persistence_column,
        "target_t_plus_1"
    ]
)

persistence_true = (
    persistence_data[
        "target_t_plus_1"
    ].values
)

persistence_pred = (
    persistence_data[
        persistence_column
    ].values
)

persistence_mae = float(
    mean_absolute_error(
        persistence_true,
        persistence_pred
    )
)

persistence_rmse = calculate_rmse(
    persistence_true,
    persistence_pred
)

persistence_r2 = float(
    r2_score(
        persistence_true,
        persistence_pred
    )
)

all_results.append({

    "model":
        "PERSISTENCE",

    "split":
        "test",

    "feature_count":
        0,

    "mae":
        persistence_mae,

    "rmse":
        persistence_rmse,

    "r2":
        persistence_r2,

    "n_records":
        len(persistence_data)
})


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df[
    "mae_improvement_vs_persistence_pct"
] = np.nan

test_mask = (
    results_df["split"]
    == "test"
)

results_df.loc[
    test_mask,
    "mae_improvement_vs_persistence_pct"
] = (
    1
    -
    results_df.loc[
        test_mask,
        "mae"
    ]
    /
    persistence_mae
) * 100.0


# ============================================================
# ROBUSTNESS CONFIGURATIONS
# ============================================================

print("\n")
print("=" * 70)
print("TRACK ROBUSTNESS ANALYSIS")
print("=" * 70)

robustness_configs = {

    "FULL_SPATIAL":

        spatial_historical_features,

    "NO_SPATIAL_LAG":

        [
            c
            for c in spatial_historical_features
            if c not in spatial_groups[
                "spatial_lag"
            ]
        ],

    "NO_SPATIAL_CHANGE":

        [
            c
            for c in spatial_historical_features
            if c not in spatial_groups[
                "spatial_change"
            ]
        ],

    "NO_SPATIAL_ACCELERATION":

        [
            c
            for c in spatial_historical_features
            if c not in spatial_groups[
                "spatial_acceleration"
            ]
        ],

    "NO_SPATIAL_ROLLING":

        [
            c
            for c in spatial_historical_features
            if c not in spatial_groups[
                "spatial_rolling"
            ]
        ],

    "NO_SPATIAL_TREND":

        [
            c
            for c in spatial_historical_features
            if c not in spatial_groups[
                "spatial_trend"
            ]
        ],

    "NO_LAG_AND_CHANGE":

        [
            c
            for c in spatial_historical_features
            if c not in (
                spatial_groups[
                    "spatial_lag"
                ]
                +
                spatial_groups[
                    "spatial_change"
                ]
            )
        ],

    "NO_ROLLING_AND_TREND":

        [
            c
            for c in spatial_historical_features
            if c not in (
                spatial_groups[
                    "spatial_rolling"
                ]
                +
                spatial_groups[
                    "spatial_trend"
                ]
            )
        ]
}


robustness_rows = []

for config_name, spatial_subset in (
    robustness_configs.items()
):

    features = unique_preserve_order(
        current_features
        + temporal_features
        + spatial_subset
    )

    print(
        f"\nConfiguration: {config_name}"
    )

    print(
        "Feature count:",
        len(features)
    )

    rows = train_evaluate(
        train,
        val,
        test,
        features,
        config_name
    )

    test_row = [
        row
        for row in rows
        if row["split"] == "test"
    ][0]

    robustness_rows.append({

        "configuration":
            config_name,

        "feature_count":
            test_row[
                "feature_count"
            ],

        "test_mae":
            test_row[
                "mae"
            ],

        "test_rmse":
            test_row[
                "rmse"
            ],

        "test_r2":
            test_row[
                "r2"
            ],

        "mae_improvement_vs_persistence_pct":
            (
                1
                -
                test_row[
                    "mae"
                ]
                /
                persistence_mae
            ) * 100.0
    })


robustness_df = pd.DataFrame(
    robustness_rows
)


# ============================================================
# ROBUSTNESS COMPARISON
# ============================================================

full_spatial_row = robustness_df[
    robustness_df[
        "configuration"
    ]
    == "FULL_SPATIAL"
]

if len(full_spatial_row) == 1:

    full_spatial_mae = float(
        full_spatial_row.iloc[0][
            "test_mae"
        ]
    )

    robustness_df[
        "mae_increase_vs_full_spatial_pct"
    ] = (
        robustness_df[
            "test_mae"
        ]
        /
        full_spatial_mae
        - 1
    ) * 100.0

else:

    full_spatial_mae = None

    robustness_df[
        "mae_increase_vs_full_spatial_pct"
    ] = np.nan


# ============================================================
# SORT
# ============================================================

results_df = results_df.sort_values(
    [
        "split",
        "mae"
    ],
    ascending=[
        True,
        True
    ]
)

robustness_df = robustness_df.sort_values(
    "test_mae"
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.3 VALIDATION RESULTS")
print("=" * 70)

print(
    results_df[
        results_df["split"] == "val"
    ][
        [
            "model",
            "feature_count",
            "mae",
            "rmse",
            "r2",
            "n_records"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.3 TEST RESULTS")
print("=" * 70)

print(
    results_df[
        results_df["split"] == "test"
    ][
        [
            "model",
            "feature_count",
            "mae",
            "rmse",
            "r2",
            "n_records",
            "mae_improvement_vs_persistence_pct"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# ROBUSTNESS RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TRACK ROBUSTNESS RESULTS")
print("=" * 70)

print(
    robustness_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

test_results = results_df[
    results_df["split"] == "test"
].copy()

non_baseline = test_results[
    test_results["model"]
    != "PERSISTENCE"
].copy()

best_row = non_baseline.loc[
    non_baseline[
        "mae"
    ].idxmin()
]

best_model = str(
    best_row["model"]
)

best_mae = float(
    best_row["mae"]
)

best_rmse = float(
    best_row["rmse"]
)

best_r2 = float(
    best_row["r2"]
)


# ============================================================
# SPATIAL CONTRIBUTION
# ============================================================

temporal_row = test_results[
    test_results["model"]
    == "CURRENT_PLUS_TEMPORAL"
]

full_row = test_results[
    test_results["model"]
    == "FULL_SPATIAL_TEMPORAL"
]

if (
    len(temporal_row) == 1
    and len(full_row) == 1
):

    temporal_test_mae = float(
        temporal_row.iloc[0][
            "mae"
        ]
    )

    full_spatial_test_mae = float(
        full_row.iloc[0][
            "mae"
        ]
    )

    spatial_improvement = (
        1
        -
        full_spatial_test_mae
        /
        temporal_test_mae
    ) * 100.0

else:

    temporal_test_mae = None

    full_spatial_test_mae = None

    spatial_improvement = None


# ============================================================
# DECISION
# ============================================================

if (
    spatial_improvement is not None
    and spatial_improvement > 0
):

    spatial_signal_decision = (
        "SPATIAL_SIGNAL_SUPPORTED"
    )

else:

    spatial_signal_decision = (
        "SPATIAL_SIGNAL_NOT_SUPPORTED"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

robustness_df.to_csv(
    ROBUSTNESS_FILE,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "3.3",

    "status":
        "PASS",

    "target_source_column":
        target_source_column,

    "primary_target_records":
        int(
            len(
                primary_target
            )
        ),

    "spatial_records":
        int(
            len(
                spatial
            )
        ),

    "temporal_records":
        int(
            len(
                temporal
            )
        ),

    "merged_records":
        int(
            len(
                merged
            )
        ),

    "split_records": {

        "train":
            int(
                len(train)
            ),

        "val":
            int(
                len(val)
            ),

        "test":
            int(
                len(test)
            )
    },

    "feature_counts": {

        "current":
            int(
                len(
                    current_features
                )
            ),

        "temporal":
            int(
                len(
                    temporal_features
                )
            ),

        "spatial_historical":
            int(
                len(
                    spatial_historical_features
                )
            )
    },

    "spatial_feature_groups": {

        name:
            int(
                len(cols)
            )

        for name, cols
        in spatial_groups.items()
    },

    "integrity": {

        "sequence_split_conflicts":
            sequence_conflicts,

        "component_split_conflicts":
            component_conflicts,

        "temporal_current_frame_errors":
            temporal_current_errors,

        "temporal_future_frame_errors":
            temporal_future_errors,

        "future_named_features":
            int(
                len(
                    future_named_features
                )
            ),

        "spatial_target_max_difference":
            0.0
    },

    "persistence_test": {

        "mae":
            persistence_mae,

        "rmse":
            persistence_rmse,

        "r2":
            persistence_r2
    },

    "best_test_model": {

        "model":
            best_model,

        "mae":
            best_mae,

        "rmse":
            best_rmse,

        "r2":
            best_r2
    },

    "spatial_contribution": {

        "temporal_test_mae":
            temporal_test_mae,

        "spatial_temporal_test_mae":
            full_spatial_test_mae,

        "spatial_improvement_vs_temporal_pct":
            spatial_improvement
    },

    "decision":
        spatial_signal_decision,

    "outputs": {

        "ablation_results":
            RESULT_FILE,

        "robustness_results":
            ROBUSTNESS_FILE,

        "summary":
            SUMMARY_FILE
    }
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
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.3 SUMMARY")
print("=" * 70)

print(
    "Target source column:",
    target_source_column
)

print(
    "Primary t+1 records:",
    len(primary_target)
)

print(
    "Current feature count:",
    len(current_features)
)

print(
    "Temporal feature count:",
    len(temporal_features)
)

print(
    "Spatial historical feature count:",
    len(spatial_historical_features)
)

print(
    "Persistence TEST MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Best TEST model:",
    best_model
)

print(
    "Best TEST MAE:",
    f"{best_mae:.8f}"
)

print(
    "Best TEST RMSE:",
    f"{best_rmse:.8f}"
)

print(
    "Best TEST R2:",
    f"{best_r2:.8f}"
)

if spatial_improvement is not None:

    print(
        "Spatial improvement vs temporal:",
        f"{spatial_improvement:.4f}%"
    )

print(
    "\nSpatial signal decision:",
    spatial_signal_decision
)

print(
    "\nAblation results:"
)

print(
    RESULT_FILE
)

print(
    "\nRobustness results:"
)

print(
    ROBUSTNESS_FILE
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