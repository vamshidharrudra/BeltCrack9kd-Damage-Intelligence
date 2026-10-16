import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STAGE 3.2
# SPATIAL + TEMPORAL ML EXPERIMENT
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

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

OUTPUT_RESULTS = os.path.join(
    AUDIT_DIR,
    "stage_3_2_spatial_temporal_results.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    AUDIT_DIR,
    "stage_3_2_spatial_temporal_predictions.csv"
)

OUTPUT_COMPARISON = os.path.join(
    AUDIT_DIR,
    "stage_3_2_model_comparison.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_2_spatial_temporal_summary.json"
)


# ============================================================
# SETTINGS
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


# ============================================================
# KEYS
# ============================================================

SPATIAL_KEY = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"
]

TEMPORAL_KEY = [
    "sequence",
    "component",
    "split",
    "frame_number"
]

TARGET_KEY = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"
]


# ============================================================
# HELPERS
# ============================================================

def check_columns(
    dataframe,
    columns,
    name
):

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            name
            +
            " is missing required columns:\n"
            +
            "\n".join(
                missing
            )
        )


def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return (
        float(mae),
        float(rmse),
        float(r2)
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.2 - SPATIAL + TEMPORAL ML EXPERIMENT"
)
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = {
    "Stage 3.1 spatial features":
        SPATIAL_FILE,

    "Stage 2.2 temporal features":
        TEMPORAL_FILE,

    "Stage 1.3 prediction targets":
        TARGET_FILE
}


for name, path in required_files.items():

    print()
    print(
        "Checking "
        +
        name
        +
        "..."
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            name
            +
            " not found:\n"
            +
            path
        )

    print(
        name
        +
        ": FOUND"
    )


# ============================================================
# LOAD
# ============================================================

print()
print(
    "Loading Stage 3.1 spatial features..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial feature records: "
    +
    str(
        len(spatial)
    )
)


print()
print(
    "Loading Stage 2.2 temporal features..."
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records: "
    +
    str(
        len(temporal)
    )
)


print()
print(
    "Loading Stage 1.3 prediction targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records: "
    +
    str(
        len(targets)
    )
)


# ============================================================
# VERIFY SCHEMAS
# ============================================================

check_columns(
    spatial,
    SPATIAL_KEY,
    "Stage 3.1 spatial file"
)

check_columns(
    temporal,
    TEMPORAL_KEY,
    "Stage 2.2 temporal file"
)

check_columns(
    targets,
    TARGET_KEY,
    "Stage 1.3 target file"
)


# ============================================================
# FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "target_t_plus_1",
    "target_crack_area_ratio",
    "future_crack_area_ratio"
]

target_column = None

for candidate in target_candidates:

    if candidate in targets.columns:

        target_column = candidate

        break


if target_column is None:

    raise ValueError(
        "Could not find t+1 target column."
    )


# ============================================================
# CHECK TARGET DUPLICATES
# ============================================================

target_duplicate_count = int(
    targets.duplicated(
        subset=TARGET_KEY
    ).sum()
)


if target_duplicate_count != 0:

    raise ValueError(
        "Duplicate target keys detected: "
        +
        str(
            target_duplicate_count
        )
    )


# ============================================================
# BUILD VERIFIED TARGET TABLE
# ============================================================

target_table = targets[
    TARGET_KEY
    +
    [
        target_column
    ]
].copy()


target_table = target_table.rename(
    columns={
        target_column:
            "verified_target_t_plus_1"
    }
)


# ============================================================
# VERIFY SPATIAL TARGET ALIGNMENT
# ============================================================

print()
print(
    "Verifying spatial target alignment..."
)


spatial_target = spatial.merge(
    target_table,
    on=TARGET_KEY,
    how="inner",
    validate="one_to_one"
)


print(
    "Matched spatial-target records: "
    +
    str(
        len(spatial_target)
    )
)


if (
    "target_t_plus_1"
    in spatial_target.columns
):

    difference = (
        spatial_target[
            "target_t_plus_1"
        ]
        -
        spatial_target[
            "verified_target_t_plus_1"
        ]
    ).abs()

    target_mismatches = int(
        (
            difference
            >
            1e-12
        ).sum()
    )

    max_target_difference = float(
        difference.max()
    )

else:

    target_mismatches = 0

    max_target_difference = 0.0


print(
    "Target mismatches: "
    +
    str(
        target_mismatches
    )
)

print(
    "Maximum target difference: "
    +
    f"{max_target_difference:.12f}"
)


if target_mismatches != 0:

    raise ValueError(
        "Target verification failed."
    )


# ============================================================
# BUILD TEMPORAL FRAME TABLE
# ============================================================

# Stage 2.2 uses:
#
# frame_number
# target_current_frame
# target_future_frame
#
# rather than current_frame/future_frame.
#
# We explicitly map frame_number -> current_frame
# and frame_number + 1 -> future_frame.

print()
print(
    "Preparing Stage 2.2 temporal features..."
)


temporal = temporal.copy()


temporal[
    "current_frame"
] = pd.to_numeric(
    temporal[
        "frame_number"
    ],
    errors="coerce"
)


temporal[
    "future_frame"
] = (
    temporal[
        "current_frame"
    ]
    +
    1
)


# ============================================================
# TEMPORAL KEY VALIDATION
# ============================================================

temporal_key_duplicate_count = int(
    temporal.duplicated(
        subset=SPATIAL_KEY
    ).sum()
)


if temporal_key_duplicate_count != 0:

    raise ValueError(
        "Duplicate temporal keys detected: "
        +
        str(
            temporal_key_duplicate_count
        )
    )


# ============================================================
# VERIFY TEMPORAL TARGET FRAME
# ============================================================

if (
    "target_current_frame"
    in temporal.columns
):

    target_current_difference = (
        temporal[
            "target_current_frame"
        ]
        -
        temporal[
            "current_frame"
        ]
    ).abs()

    target_current_errors = int(
        (
            target_current_difference
            >
            0
        ).sum()
    )

else:

    target_current_errors = 0


if (
    "target_future_frame"
    in temporal.columns
):

    target_future_difference = (
        temporal[
            "target_future_frame"
        ]
        -
        temporal[
            "future_frame"
        ]
    ).abs()

    target_future_errors = int(
        (
            target_future_difference
            >
            0
        ).sum()
    )

else:

    target_future_errors = 0


print(
    "Temporal current-frame errors: "
    +
    str(
        target_current_errors
    )
)

print(
    "Temporal future-frame errors: "
    +
    str(
        target_future_errors
    )
)


if target_current_errors != 0:

    raise ValueError(
        "Temporal current-frame mapping failed."
    )


if target_future_errors != 0:

    raise ValueError(
        "Temporal future-frame mapping failed."
    )


# ============================================================
# TEMPORAL FEATURE IDENTIFICATION
# ============================================================

TEMPORAL_PREFIXES = [
    "lag1_",
    "lag2_",
    "lag3_",
    "delta1_",
    "delta2_",
    "accel_",
    "rolling3_",
    "recent_"
]


temporal_feature_columns = []

for column in temporal.columns:

    if any(
        column.startswith(prefix)
        for prefix in TEMPORAL_PREFIXES
    ):

        temporal_feature_columns.append(
            column
        )


# ============================================================
# IMPORTANT TEMPORAL CURRENT FEATURES
# ============================================================

# The Stage 2.2 file contains current physical damage
# features. We keep them separate from historical temporal
# features so that the experiment remains interpretable.

CURRENT_PHYSICAL_FEATURES = [
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "annotation_shape_count",
    "image_area_px"
]


current_features_available = [
    column
    for column in CURRENT_PHYSICAL_FEATURES
    if column in temporal.columns
]


if (
    "current_crack_area_ratio"
    in temporal.columns
):

    current_features_available.append(
        "current_crack_area_ratio"
    )


# Preserve order and remove duplicates.

current_features_available = list(
    dict.fromkeys(
        current_features_available
    )
)


# ============================================================
# MERGE TEMPORAL FEATURES
# ============================================================

temporal_subset_columns = (
    SPATIAL_KEY
    +
    current_features_available
    +
    temporal_feature_columns
)


temporal_subset_columns = list(
    dict.fromkeys(
        temporal_subset_columns
    )
)


temporal_subset = temporal[
    temporal_subset_columns
].copy()


print(
    "Temporal historical feature count: "
    +
    str(
        len(
            temporal_feature_columns
        )
    )
)


# ============================================================
# MERGE SPATIAL + TEMPORAL
# ============================================================

print()
print(
    "Merging spatial and temporal features..."
)


data = spatial.merge(
    temporal_subset,
    on=SPATIAL_KEY,
    how="inner",
    validate="one_to_one"
)


print(
    "Spatial-temporal merged records: "
    +
    str(
        len(data)
    )
)


# ============================================================
# MERGE VERIFIED TARGET
# ============================================================

data = data.merge(
    target_table,
    on=TARGET_KEY,
    how="inner",
    validate="one_to_one"
)


print(
    "Records after target merge: "
    +
    str(
        len(data)
    )
)


# ============================================================
# TARGET COMPLETENESS
# ============================================================

missing_target_count = int(
    data[
        "verified_target_t_plus_1"
    ].isna().sum()
)


if missing_target_count != 0:

    raise ValueError(
        "Missing target values: "
        +
        str(
            missing_target_count
        )
    )


# ============================================================
# SPATIAL FUTURE-DEPENDENT COLUMNS
# ============================================================

# These values in Stage 3.1 were computed by comparing
# current and future frames. They are therefore NOT allowed
# as forecasting features for the same t+1 target.

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


# ============================================================
# SPATIAL HISTORICAL FEATURE IDENTIFICATION
# ============================================================

SPATIAL_PREFIXES = [
    "spatial_lag",
    "spatial_delta",
    "spatial_accel",
    "spatial_rolling3",
    "spatial_recent"
]


spatial_feature_columns = []

for column in spatial.columns:

    if any(
        column.startswith(prefix)
        for prefix in SPATIAL_PREFIXES
    ):

        spatial_feature_columns.append(
            column
        )


spatial_feature_columns = [
    column
    for column in spatial_feature_columns
    if column
    not in future_dependent_spatial
]


# ============================================================
# REMOVE IDENTIFIERS
# ============================================================

identifier_columns = set(
    SPATIAL_KEY
    +
    [
        "frame_number",
        "target_current_frame",
        "target_future_frame",
        "verified_target_t_plus_1",
        "target_t_plus_1",
        "continuity_class",
        "valid_temporal_direction",
        "frame_gap"
    ]
)


current_features = [
    column
    for column in current_features_available
    if column
    not in identifier_columns
]


temporal_features = [
    column
    for column in temporal_feature_columns
    if column
    not in identifier_columns
]


spatial_features = [
    column
    for column in spatial_feature_columns
    if column
    not in identifier_columns
]


# ============================================================
# REMOVE CURRENT DUPLICATES FROM TEMPORAL GROUP
# ============================================================

# Stage 2.2 historical temporal features already include
# some current-prefixed information. Keep the explicit
# historical features only for the temporal experiment.

temporal_features = list(
    dict.fromkeys(
        temporal_features
    )
)

current_features = list(
    dict.fromkeys(
        current_features
    )
)

spatial_features = list(
    dict.fromkeys(
        spatial_features
    )
)


# ============================================================
# CHECK FEATURES EXIST
# ============================================================

missing_current = [
    column
    for column in current_features
    if column not in data.columns
]

missing_temporal = [
    column
    for column in temporal_features
    if column not in data.columns
]

missing_spatial = [
    column
    for column in spatial_features
    if column not in data.columns
]


if missing_current:

    raise ValueError(
        "Missing current features:\n"
        +
        "\n".join(
            missing_current
        )
    )


if missing_temporal:

    raise ValueError(
        "Missing temporal features:\n"
        +
        "\n".join(
            missing_temporal
        )
    )


if missing_spatial:

    raise ValueError(
        "Missing spatial features:\n"
        +
        "\n".join(
            missing_spatial
        )
    )


# ============================================================
# FUTURE-NAMED FEATURE AUDIT
# ============================================================

candidate_feature_columns = (
    current_features
    +
    temporal_features
    +
    spatial_features
)


candidate_feature_columns = list(
    dict.fromkeys(
        candidate_feature_columns
    )
)


future_named_features = []

for column in candidate_feature_columns:

    lower = column.lower()

    if (
        "future" in lower
        or "target" in lower
        or "t_plus" in lower
        or "tplus" in lower
    ):

        future_named_features.append(
            column
        )


if future_named_features:

    raise ValueError(
        "Future-named model features detected:\n"
        +
        "\n".join(
            future_named_features
        )
    )


# ============================================================
# SPLIT INTEGRITY
# ============================================================

sequence_split_conflicts = 0

for sequence_name, group in data.groupby(
    "sequence"
):

    if group[
        "split"
    ].nunique() > 1:

        sequence_split_conflicts += 1


component_split_conflicts = 0

for component_name, group in data.groupby(
    "component"
):

    if group[
        "split"
    ].nunique() > 1:

        component_split_conflicts += 1


if sequence_split_conflicts != 0:

    raise ValueError(
        "Sequence split conflicts detected."
    )


if component_split_conflicts != 0:

    raise ValueError(
        "Component split conflicts detected."
    )


# ============================================================
# FRAME DIRECTION
# ============================================================

data[
    "frame_gap"
] = (
    data[
        "future_frame"
    ]
    -
    data[
        "current_frame"
    ]
)


invalid_frame_gap = int(
    (
        data[
            "frame_gap"
        ]
        !=
        1
    ).sum()
)


if invalid_frame_gap != 0:

    raise ValueError(
        "Invalid current-to-future frame mappings: "
        +
        str(
            invalid_frame_gap
        )
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

all_model_features = list(
    dict.fromkeys(
        candidate_feature_columns
    )
)


for column in all_model_features:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# COMPLETE RECORDS
# ============================================================

required_for_model = (
    all_model_features
    +
    [
        "verified_target_t_plus_1"
    ]
)


complete_mask = (
    data[
        required_for_model
    ]
    .notna()
    .all(
        axis=1
    )
)


complete_data = data.loc[
    complete_mask
].copy()


print()
print(
    "Complete ML records: "
    +
    str(
        len(complete_data)
    )
)


# ============================================================
# SPLIT
# ============================================================

train_data = complete_data[
    complete_data[
        "split"
    ]
    ==
    "train"
].copy()


val_data = complete_data[
    complete_data[
        "split"
    ]
    ==
    "val"
].copy()


test_data = complete_data[
    complete_data[
        "split"
    ]
    ==
    "test"
].copy()


print()
print(
    "ML split sizes:"
)

print(
    "TRAIN: "
    +
    str(
        len(train_data)
    )
)

print(
    "VAL:   "
    +
    str(
        len(val_data)
    )
)

print(
    "TEST:  "
    +
    str(
        len(test_data)
    )
)


# ============================================================
# TARGET
# ============================================================

target_name = (
    "verified_target_t_plus_1"
)


y_train = train_data[
    target_name
].values

y_val = val_data[
    target_name
].values

y_test = test_data[
    target_name
].values


# ============================================================
# MODEL GROUPS
# ============================================================

MODEL_GROUPS = {

    "CURRENT_ONLY":
        current_features,

    "CURRENT_PLUS_TEMPORAL":
        (
            current_features
            +
            temporal_features
        ),

    "CURRENT_PLUS_TEMPORAL_PLUS_SPATIAL":
        (
            current_features
            +
            temporal_features
            +
            spatial_features
        )
}


for model_name in MODEL_GROUPS:

    MODEL_GROUPS[
        model_name
    ] = list(
        dict.fromkeys(
            MODEL_GROUPS[
                model_name
            ]
        )
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print(
    "TRAINING GRADIENT BOOSTING MODELS"
)
print("=" * 70)


result_rows = []
prediction_rows = []


for model_name, feature_columns in MODEL_GROUPS.items():

    print()
    print(
        "Model: "
        +
        model_name
    )

    print(
        "Feature count: "
        +
        str(
            len(
                feature_columns
            )
        )
    )

    X_train = train_data[
        feature_columns
    ]

    X_val = val_data[
        feature_columns
    ]

    X_test = test_data[
        feature_columns
    ]


    model = GradientBoostingRegressor(
        **MODEL_PARAMS
    )


    print(
        "Training..."
    )

    model.fit(
        X_train,
        y_train
    )


    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    prediction_sets = [
        (
            "train",
            y_train,
            train_pred
        ),
        (
            "val",
            y_val,
            val_pred
        ),
        (
            "test",
            y_test,
            test_pred
        )
    ]


    for split_name, actual, predicted in prediction_sets:

        mae, rmse, r2 = calculate_metrics(
            actual,
            predicted
        )

        result_rows.append(
            {
                "model":
                    model_name,

                "split":
                    split_name,

                "feature_count":
                    len(
                        feature_columns
                    ),

                "mae":
                    mae,

                "rmse":
                    rmse,

                "r2":
                    r2,

                "n_records":
                    len(actual)
            }
        )


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_prediction_table = test_data[
        SPATIAL_KEY
        +
        [
            "current_crack_count",
            "current_total_area",
            target_name
        ]
    ].copy()


    test_prediction_table[
        "model"
    ] = model_name


    test_prediction_table[
        "prediction"
    ] = test_pred


    test_prediction_table[
        "absolute_error"
    ] = np.abs(
        test_prediction_table[
            target_name
        ]
        -
        test_prediction_table[
            "prediction"
        ]
    )


    test_prediction_table[
        "residual"
    ] = (
        test_prediction_table[
            "prediction"
        ]
        -
        test_prediction_table[
            target_name
        ]
    )


    prediction_rows.append(
        test_prediction_table
    )


# ============================================================
# RESULT DATAFRAME
# ============================================================

results = pd.DataFrame(
    result_rows
)


predictions = pd.concat(
    prediction_rows,
    ignore_index=True
)


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

print()
print(
    "Calculating persistence baseline..."
)


if "current_crack_area_ratio" in test_data.columns:

    persistence_pred = (
        test_data[
            "current_crack_area_ratio"
        ].values
    )

elif "crack_area_ratio" in test_data.columns:

    persistence_pred = (
        test_data[
            "crack_area_ratio"
        ].values
    )

else:

    # This should not normally be needed because Stage 2.2
    # contains current_crack_area_ratio.

    persistence_source = temporal[
        TEMPORAL_KEY
        +
        [
            "current_crack_area_ratio"
        ]
    ].copy()

    persistence_source = (
        persistence_source
        .drop_duplicates(
            subset=TEMPORAL_KEY
        )
    )


    test_persistence = test_data[
        SPATIAL_KEY
        +
        [
            target_name
        ]
    ].copy()


    test_persistence = test_persistence.merge(
        persistence_source,
        left_on=SPATIAL_KEY,
        right_on=[
            "sequence",
            "component",
            "split",
            "frame_number"
        ],
        how="left"
    )


    persistence_pred = (
        test_persistence[
            "current_crack_area_ratio"
        ].values
    )


persistence_actual = (
    test_data[
        target_name
    ].values
)


persistence_mae, persistence_rmse, persistence_r2 = (
    calculate_metrics(
        persistence_actual,
        persistence_pred
    )
)


results = pd.concat(
    [
        results,

        pd.DataFrame(
            [
                {
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
                        len(
                            persistence_actual
                        )
                }
            ]
        )
    ],
    ignore_index=True
)


# ============================================================
# TEST RESULTS
# ============================================================

test_results = results[
    results[
        "split"
    ]
    ==
    "test"
].copy()


test_results = test_results.sort_values(
    "mae",
    ascending=True
).reset_index(
    drop=True
)


# ============================================================
# PERSISTENCE IMPROVEMENT
# ============================================================

test_results[
    "mae_improvement_vs_persistence_pct"
] = (
    (
        persistence_mae
        -
        test_results[
            "mae"
        ]
    )
    /
    persistence_mae
    *
    100.0
)


# ============================================================
# TEMPORAL VS SPATIAL
# ============================================================

temporal_rows = test_results[
    test_results[
        "model"
    ]
    ==
    "CURRENT_PLUS_TEMPORAL"
]


spatial_rows = test_results[
    test_results[
        "model"
    ]
    ==
    "CURRENT_PLUS_TEMPORAL_PLUS_SPATIAL"
]


if (
    len(temporal_rows) == 1
    and
    len(spatial_rows) == 1
):

    temporal_mae = float(
        temporal_rows[
            "mae"
        ].iloc[0]
    )

    spatial_mae = float(
        spatial_rows[
            "mae"
        ].iloc[0]
    )

    spatial_improvement = (
        (
            temporal_mae
            -
            spatial_mae
        )
        /
        temporal_mae
        *
        100.0
    )

else:

    temporal_mae = np.nan
    spatial_mae = np.nan
    spatial_improvement = np.nan


# ============================================================
# BEST MODEL
# ============================================================

best_row = (
    test_results
    .sort_values(
        "mae"
    )
    .iloc[0]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_RESULTS,
    index=False
)


predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


test_results.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.2",

    "status":
        "PASS",

    "experiment":
        "Current vs Current+Temporal vs Current+Temporal+Spatial",

    "model":
        "GradientBoostingRegressor",

    "model_parameters":
        MODEL_PARAMS,

    "records":
        {
            "spatial_input":
                int(
                    len(spatial)
                ),

            "temporal_input":
                int(
                    len(temporal)
                ),

            "target_input":
                int(
                    len(targets)
                ),

            "merged":
                int(
                    len(data)
                ),

            "complete":
                int(
                    len(complete_data)
                ),

            "train":
                int(
                    len(train_data)
                ),

            "val":
                int(
                    len(val_data)
                ),

            "test":
                int(
                    len(test_data)
                )
        },

    "feature_counts":
        {
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

            "spatial":
                int(
                    len(
                        spatial_features
                    )
                ),

            "current_plus_temporal":
                int(
                    len(
                        MODEL_GROUPS[
                            "CURRENT_PLUS_TEMPORAL"
                        ]
                    )
                ),

            "current_plus_temporal_plus_spatial":
                int(
                    len(
                        MODEL_GROUPS[
                            "CURRENT_PLUS_TEMPORAL_PLUS_SPATIAL"
                        ]
                    )
                )
        },

    "target_verification":
        {
            "duplicate_target_keys":
                int(
                    target_duplicate_count
                ),

            "mismatches":
                int(
                    target_mismatches
                ),

            "maximum_difference":
                float(
                    max_target_difference
                )
        },

    "temporal_mapping_verification":
        {
            "current_frame_errors":
                int(
                    target_current_errors
                ),

            "future_frame_errors":
                int(
                    target_future_errors
                )
        },

    "split_integrity":
        {
            "sequence_conflicts":
                int(
                    sequence_split_conflicts
                ),

            "component_conflicts":
                int(
                    component_split_conflicts
                ),

            "invalid_frame_gaps":
                int(
                    invalid_frame_gap
                )
        },

    "leakage_audit":
        {
            "future_named_features":
                future_named_features,

            "future_dependent_spatial_columns_excluded":
                future_dependent_spatial
        },

    "persistence_test":
        {
            "mae":
                float(
                    persistence_mae
                ),

            "rmse":
                float(
                    persistence_rmse
                ),

            "r2":
                float(
                    persistence_r2
                )
        },

    "best_test_model":
        {
            "model":
                str(
                    best_row[
                        "model"
                    ]
                ),

            "mae":
                float(
                    best_row[
                        "mae"
                    ]
                ),

            "rmse":
                float(
                    best_row[
                        "rmse"
                    ]
                ),

            "r2":
                float(
                    best_row[
                        "r2"
                    ]
                )
        },

    "temporal_test_mae":
        float(
            temporal_mae
        )
        if pd.notna(
            temporal_mae
        )
        else None,

    "spatial_temporal_test_mae":
        float(
            spatial_mae
        )
        if pd.notna(
            spatial_mae
        )
        else None,

    "spatial_improvement_vs_temporal_pct":
        float(
            spatial_improvement
        )
        if pd.notna(
            spatial_improvement
        )
        else None,

    "outputs":
        {
            "results":
                OUTPUT_RESULTS,

            "predictions":
                OUTPUT_PREDICTIONS,

            "comparison":
                OUTPUT_COMPARISON,

            "summary":
                OUTPUT_SUMMARY
        }
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 3.2 VALIDATION RESULTS"
)
print("=" * 70)


validation_results = results[
    results[
        "split"
    ]
    ==
    "val"
].sort_values(
    "mae",
    ascending=True
)


print(
    validation_results.to_string(
        index=False
    )
)


# ============================================================
# PRINT TEST RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 3.2 TEST RESULTS"
)
print("=" * 70)


print(
    test_results.to_string(
        index=False
    )
)


# ============================================================
# SPATIAL CONTRIBUTION
# ============================================================

print()
print("=" * 70)
print(
    "SPATIAL CONTRIBUTION"
)
print("=" * 70)


print(
    "Temporal model test MAE: "
    +
    f"{temporal_mae:.8f}"
)


print(
    "Spatial + temporal test MAE: "
    +
    f"{spatial_mae:.8f}"
)


print(
    "Spatial improvement vs temporal: "
    +
    f"{spatial_improvement:.4f}%"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 3.2 SUMMARY"
)
print("=" * 70)


print(
    "Merged records: "
    +
    str(
        len(data)
    )
)

print(
    "Complete ML records: "
    +
    str(
        len(complete_data)
    )
)

print(
    "TRAIN: "
    +
    str(
        len(train_data)
    )
)

print(
    "VAL:   "
    +
    str(
        len(val_data)
    )
)

print(
    "TEST:  "
    +
    str(
        len(test_data)
    )
)

print()
print(
    "Current feature count: "
    +
    str(
        len(current_features)
    )
)

print(
    "Temporal feature count: "
    +
    str(
        len(temporal_features)
    )
)

print(
    "Spatial feature count: "
    +
    str(
        len(spatial_features)
    )
)

print()
print(
    "Target mismatches: "
    +
    str(
        target_mismatches
    )
)

print(
    "Maximum target difference: "
    +
    f"{max_target_difference:.12f}"
)

print(
    "Temporal current-frame errors: "
    +
    str(
        target_current_errors
    )
)

print(
    "Temporal future-frame errors: "
    +
    str(
        target_future_errors
    )
)

print(
    "Sequence split conflicts: "
    +
    str(
        sequence_split_conflicts
    )
)

print(
    "Component split conflicts: "
    +
    str(
        component_split_conflicts
    )
)

print(
    "Invalid frame gaps: "
    +
    str(
        invalid_frame_gap
    )
)

print()
print(
    "Future-named features: "
    +
    str(
        len(
            future_named_features
        )
    )
)

print()
print(
    "Persistence TEST MAE: "
    +
    f"{persistence_mae:.8f}"
)

print(
    "Best TEST model: "
    +
    str(
        best_row[
            "model"
        ]
    )
)

print(
    "Best TEST MAE: "
    +
    f"{float(best_row['mae']):.8f}"
)

print(
    "Best TEST RMSE: "
    +
    f"{float(best_row['rmse']):.8f}"
)

print(
    "Best TEST R2: "
    +
    f"{float(best_row['r2']):.8f}"
)

print()
print(
    "Spatial improvement vs temporal: "
    +
    f"{spatial_improvement:.4f}%"
)

print()
print(
    "Results:"
)

print(
    OUTPUT_RESULTS
)

print()
print(
    "Predictions:"
)

print(
    OUTPUT_PREDICTIONS
)

print()
print(
    "Comparison:"
)

print(
    OUTPUT_COMPARISON
)

print()
print(
    "Summary:"
)

print(
    OUTPUT_SUMMARY
)

print()
print(
    "FINAL STATUS: PASS"
)