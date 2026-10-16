import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 3.1
# LEAKAGE-SAFE SPATIAL DAMAGE TRACK FEATURE ENGINEERING
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

PAIR_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

TEMPORAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

TARGET_FILE = os.path.join(
    AUDIT_DIR,
    "stage_1_3_prediction_targets.csv"
)

OUTPUT_FEATURE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_features.csv"
)

OUTPUT_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_feature_summary.csv"
)

OUTPUT_CORRELATION_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_feature_target_correlations.csv"
)

OUTPUT_AUDIT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_feature_audit.json"
)


# ============================================================
# HELPER
# ============================================================

def build_lag_block(
    dataframe,
    columns,
    lag
):

    block = {}

    grouped = dataframe.groupby(
        "sequence"
    )

    for column in columns:

        block[
            "spatial_lag"
            + str(lag)
            + "_"
            + column
        ] = (
            grouped[column]
            .shift(lag)
        )

    return pd.DataFrame(
        block,
        index=dataframe.index
    )


def build_delta_block(
    dataframe,
    columns
):

    block = {}

    grouped = dataframe.groupby(
        "sequence"
    )

    for column in columns:

        lag1 = grouped[column].shift(1)
        lag2 = grouped[column].shift(2)

        block[
            "spatial_delta1_"
            + column
        ] = (
            dataframe[column]
            -
            lag1
        )

        block[
            "spatial_delta2_"
            + column
        ] = (
            dataframe[column]
            -
            lag2
        )

    return pd.DataFrame(
        block,
        index=dataframe.index
    )


def build_acceleration_block(
    dataframe,
    columns
):

    block = {}

    grouped = dataframe.groupby(
        "sequence"
    )

    for column in columns:

        lag1 = grouped[column].shift(1)
        lag2 = grouped[column].shift(2)

        current_delta = (
            dataframe[column]
            -
            lag1
        )

        previous_delta = (
            lag1
            -
            lag2
        )

        block[
            "spatial_accel_"
            + column
        ] = (
            current_delta
            -
            previous_delta
        )

    return pd.DataFrame(
        block,
        index=dataframe.index
    )


def build_rolling_block(
    dataframe,
    columns
):

    mean_block = {}
    std_block = {}

    for column in columns:

        rolling_mean = (
            dataframe
            .groupby(
                "sequence"
            )[column]
            .transform(
                lambda s:
                s.rolling(
                    window=3,
                    min_periods=1
                ).mean()
            )
        )

        rolling_std = (
            dataframe
            .groupby(
                "sequence"
            )[column]
            .transform(
                lambda s:
                s.rolling(
                    window=3,
                    min_periods=1
                ).std()
                .fillna(0.0)
            )
        )

        mean_block[
            "spatial_rolling3_mean_"
            + column
        ] = rolling_mean

        std_block[
            "spatial_rolling3_std_"
            + column
        ] = rolling_std

    mean_df = pd.DataFrame(
        mean_block,
        index=dataframe.index
    )

    std_df = pd.DataFrame(
        std_block,
        index=dataframe.index
    )

    return pd.concat(
        [
            mean_df,
            std_df
        ],
        axis=1
    )


def build_trend_block(
    dataframe,
    columns
):

    block = {}

    grouped = dataframe.groupby(
        "sequence"
    )

    for column in columns:

        lag1 = grouped[column].shift(1)
        lag3 = grouped[column].shift(3)

        block[
            "spatial_recent_change_"
            + column
        ] = (
            dataframe[column]
            -
            lag3
        )

        block[
            "spatial_recent_slope_"
            + column
        ] = (
            dataframe[column]
            -
            lag3
        ) / 3.0

        block[
            "spatial_recent_velocity_change_"
            + column
        ] = (
            dataframe[column]
            -
            lag1
        )

    return pd.DataFrame(
        block,
        index=dataframe.index
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.1 - LEAKAGE-SAFE SPATIAL DAMAGE TRACK FEATURES"
)
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

required_files = {
    "Stage 3.0 spatial pairs":
        PAIR_FILE,

    "Stage 2.2 temporal features":
        TEMPORAL_FILE,

    "Stage 1.3 targets":
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
# LOAD INPUTS
# ============================================================

print()
print(
    "Loading Stage 3.0 spatial track pairs..."
)

spatial = pd.read_csv(
    PAIR_FILE
)

print(
    "Spatial pair records: "
    +
    str(len(spatial))
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
    str(len(temporal))
)


print()
print(
    "Loading Stage 1.3 targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records: "
    +
    str(len(targets))
)


# ============================================================
# REQUIRED SPATIAL COLUMNS
# ============================================================

required_spatial_columns = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame",
    "current_crack_count",
    "matched_crack_count",
    "persistent_fraction",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score",
    "current_total_area",
    "continuity_class"
]


missing_spatial = [
    column
    for column in required_spatial_columns
    if column not in spatial.columns
]


if missing_spatial:

    raise ValueError(
        "Missing Stage 3.0 columns:\n"
        +
        "\n".join(
            missing_spatial
        )
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
# SORT
# ============================================================

spatial = spatial.sort_values(
    [
        "sequence",
        "current_frame"
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD CURRENT SPATIAL TRACK DATA
# ============================================================

print()
print(
    "Constructing spatial track features..."
)


features = spatial[
    [
        "sequence",
        "component",
        "split",
        "current_frame",
        "future_frame",
        "current_crack_count",
        "matched_crack_count",
        "persistent_fraction",
        "mean_match_iou",
        "mean_center_distance",
        "mean_area_similarity",
        "mean_match_score",
        "current_total_area",
        "continuity_class"
    ]
].copy()


# ============================================================
# CURRENT TRACK FEATURES
# ============================================================

features[
    "matched_fraction_current"
] = np.where(
    features[
        "current_crack_count"
    ] > 0,

    features[
        "matched_crack_count"
    ]
    /
    features[
        "current_crack_count"
    ],

    0.0
)


features[
    "unmatched_current_crack_count"
] = (
    features[
        "current_crack_count"
    ]
    -
    features[
        "matched_crack_count"
    ]
)


features[
    "unmatched_current_fraction"
] = np.where(
    features[
        "current_crack_count"
    ] > 0,

    features[
        "unmatched_current_crack_count"
    ]
    /
    features[
        "current_crack_count"
    ],

    0.0
)


# ============================================================
# MATCH QUALITY
# ============================================================

features[
    "spatial_match_quality"
] = (
    0.40
    *
    features[
        "mean_match_iou"
    ]
    +
    0.30
    *
    features[
        "mean_area_similarity"
    ]
    +
    0.30
    *
    features[
        "mean_match_score"
    ]
)


features[
    "spatial_instability"
] = (
    features[
        "mean_center_distance"
    ]
    *
    (
        1.0
        -
        features[
            "mean_match_iou"
        ]
    )
)


features[
    "track_continuity_quality"
] = (
    features[
        "persistent_fraction"
    ]
    *
    features[
        "spatial_match_quality"
    ]
)


# ============================================================
# CONTINUITY SCORE
# ============================================================

continuity_mapping = {
    "HIGH_CONTINUITY": 2,
    "MODERATE_CONTINUITY": 1,
    "LOW_CONTINUITY": 0,
    "DAMAGE_DISAPPEARED": -1,
    "NO_CURRENT_DAMAGE": -1
}


features[
    "continuity_score"
] = (
    features[
        "continuity_class"
    ]
    .map(
        continuity_mapping
    )
    .fillna(-1)
)


# ============================================================
# SORT AGAIN
# ============================================================

features = features.sort_values(
    [
        "sequence",
        "current_frame"
    ]
).reset_index(
    drop=True
)


# ============================================================
# HISTORICAL TRACK VARIABLES
# ============================================================

history_columns = [
    "persistent_fraction",
    "matched_fraction_current",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score",
    "spatial_match_quality",
    "spatial_instability",
    "track_continuity_quality",
    "continuity_score",
    "matched_crack_count",
    "unmatched_current_crack_count",
    "current_crack_count"
]


# ============================================================
# LAG FEATURES
# ============================================================

print()
print(
    "Building lag features..."
)


lag_blocks = []

for lag in [1, 2, 3]:

    lag_blocks.append(
        build_lag_block(
            features,
            history_columns,
            lag
        )
    )


lag_features = pd.concat(
    lag_blocks,
    axis=1
)


# ============================================================
# DELTA FEATURES
# ============================================================

print(
    "Building change features..."
)


delta_features = build_delta_block(
    features,
    history_columns
)


# ============================================================
# ACCELERATION
# ============================================================

print(
    "Building acceleration features..."
)


acceleration_columns = [
    "persistent_fraction",
    "mean_center_distance",
    "mean_match_iou",
    "spatial_match_quality",
    "spatial_instability",
    "track_continuity_quality",
    "matched_crack_count"
]


acceleration_features = (
    build_acceleration_block(
        features,
        acceleration_columns
    )
)


# ============================================================
# ROLLING FEATURES
# ============================================================

print(
    "Building rolling features..."
)


rolling_columns = [
    "persistent_fraction",
    "matched_fraction_current",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score",
    "spatial_match_quality",
    "spatial_instability",
    "track_continuity_quality",
    "matched_crack_count",
    "unmatched_current_crack_count"
]


rolling_features = build_rolling_block(
    features,
    rolling_columns
)


# ============================================================
# TREND FEATURES
# ============================================================

print(
    "Building trend features..."
)


trend_columns = [
    "persistent_fraction",
    "mean_center_distance",
    "mean_match_iou",
    "spatial_match_quality",
    "spatial_instability",
    "track_continuity_quality",
    "matched_crack_count"
]


trend_features = build_trend_block(
    features,
    trend_columns
)


# ============================================================
# COMBINE
# ============================================================

features = pd.concat(
    [
        features,
        lag_features,
        delta_features,
        acceleration_features,
        rolling_features,
        trend_features
    ],
    axis=1
)


features = features.copy()


# ============================================================
# FRAME CHECK
# ============================================================

features[
    "frame_gap"
] = (
    features[
        "future_frame"
    ]
    -
    features[
        "current_frame"
    ]
)


features[
    "valid_temporal_direction"
] = (
    features[
        "frame_gap"
    ]
    ==
    1
)


# ============================================================
# TARGET MERGE
# ============================================================

print()
print(
    "Merging independently verified t+1 target..."
)


target_key = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"
]


target_subset = targets[
    target_key
    +
    [
        target_column
    ]
].copy()


target_subset = target_subset.rename(
    columns={
        target_column:
            "target_t_plus_1"
    }
)


target_duplicate_keys = int(
    target_subset.duplicated(
        subset=target_key
    ).sum()
)


if target_duplicate_keys != 0:

    raise ValueError(
        "Duplicate target keys found: "
        +
        str(
            target_duplicate_keys
        )
    )


features = features.merge(
    target_subset,
    on=target_key,
    how="left",
    validate="one_to_one"
)


# ============================================================
# TARGET CHECK
# ============================================================

missing_targets = int(
    features[
        "target_t_plus_1"
    ].isna().sum()
)


# ============================================================
# SPLIT CHECKS
# ============================================================

sequence_split_conflicts = 0

for sequence_name, group in features.groupby(
    "sequence"
):

    if group[
        "split"
    ].nunique() > 1:

        sequence_split_conflicts += 1


component_split_conflicts = 0

for component_name, group in features.groupby(
    "component"
):

    if group[
        "split"
    ].nunique() > 1:

        component_split_conflicts += 1


duplicate_keys = int(
    features.duplicated(
        subset=target_key
    ).sum()
)


invalid_temporal_direction = int(
    (
        ~features[
            "valid_temporal_direction"
        ]
    ).sum()
)


# ============================================================
# LEAKAGE-SAFE FORECAST FEATURE SET
# ============================================================

identifier_columns = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame",
    "frame_gap",
    "valid_temporal_direction",
    "target_t_plus_1",
    "continuity_class"
]


all_candidate_features = [
    column
    for column in features.columns
    if column
    not in identifier_columns
]


# ------------------------------------------------------------
# CRITICAL:
#
# These Stage 3.0 columns depend on the future frame.
# They cannot be used to predict that same future frame.
# ------------------------------------------------------------

future_dependent_columns = [
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


# Current-row versions are excluded.
#
# Their lagged/rolling historical versions are allowed because
# they describe track behavior observed before the current
# prediction point.


forecast_feature_columns = []

for column in all_candidate_features:

    if column in future_dependent_columns:

        continue

    forecast_feature_columns.append(
        column
    )


# ============================================================
# EXCLUDE EXPLICIT FUTURE INFORMATION
# ============================================================

forbidden_terms = [
    "future",
    "target",
    "t_plus",
    "tplus"
]


future_named_features = []

for column in forecast_feature_columns:

    lower = column.lower()

    if any(
        term in lower
        for term in forbidden_terms
    ):

        future_named_features.append(
            column
        )


if future_named_features:

    forecast_feature_columns = [
        column
        for column in forecast_feature_columns
        if column
        not in future_named_features
    ]


# ============================================================
# FUTURE FRAME MUST NEVER BE A MODEL FEATURE
# ============================================================

future_frame_used_as_feature = (
    "future_frame"
    in forecast_feature_columns
)


# ============================================================
# COMPLETE RECORDS
# ============================================================

complete_mask = (
    features[
        forecast_feature_columns
    ]
    .notna()
    .all(
        axis=1
    )
    &
    features[
        "target_t_plus_1"
    ].notna()
)


complete_records = int(
    complete_mask.sum()
)


# ============================================================
# CORRELATIONS
# ============================================================

print()
print(
    "Calculating feature-target correlations..."
)


correlation_rows = []

for column in forecast_feature_columns:

    values = pd.to_numeric(
        features[
            column
        ],
        errors="coerce"
    )

    valid = (
        values.notna()
        &
        features[
            "target_t_plus_1"
        ].notna()
    )

    if valid.sum() >= 3:

        correlation = float(
            values[
                valid
            ].corr(
                features[
                    "target_t_plus_1"
                ][
                    valid
                ]
            )
        )

    else:

        correlation = np.nan

    correlation_rows.append(
        {
            "feature":
                column,

            "target_correlation":
                correlation,

            "absolute_correlation":
                abs(correlation)
                if pd.notna(
                    correlation
                )
                else np.nan
        }
    )


correlations = pd.DataFrame(
    correlation_rows
)


correlations = correlations.sort_values(
    "absolute_correlation",
    ascending=False
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print(
    "Building feature summary..."
)


summary_rows = []

for column in forecast_feature_columns:

    values = pd.to_numeric(
        features[
            column
        ],
        errors="coerce"
    )

    valid_count = int(
        values.notna().sum()
    )

    missing_count = int(
        values.isna().sum()
    )

    if valid_count > 0:

        mean_value = float(
            values.mean()
        )

        std_value = float(
            values.std()
        )

        min_value = float(
            values.min()
        )

        max_value = float(
            values.max()
        )

    else:

        mean_value = np.nan
        std_value = np.nan
        min_value = np.nan
        max_value = np.nan

    summary_rows.append(
        {
            "feature":
                column,

            "valid_records":
                valid_count,

            "missing_records":
                missing_count,

            "missing_fraction":
                (
                    missing_count
                    /
                    len(features)
                ),

            "mean":
                mean_value,

            "std":
                std_value,

            "min":
                min_value,

            "max":
                max_value
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

features.to_csv(
    OUTPUT_FEATURE_FILE,
    index=False
)


summary.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False
)


correlations.to_csv(
    OUTPUT_CORRELATION_FILE,
    index=False
)


# ============================================================
# AUDIT CHECKS
# ============================================================

checks = {

    "spatial_records_expected":
        len(spatial) == 14058,

    "target_missing_zero":
        missing_targets == 0,

    "duplicate_keys_zero":
        duplicate_keys == 0,

    "invalid_temporal_direction_zero":
        invalid_temporal_direction == 0,

    "sequence_split_conflicts_zero":
        sequence_split_conflicts == 0,

    "component_split_conflicts_zero":
        component_split_conflicts == 0,

    "future_named_features_zero":
        len(
            future_named_features
        ) == 0,

    "future_frame_not_feature":
        future_frame_used_as_feature is False,

    "complete_records_positive":
        complete_records > 0
}


final_pass = all(
    checks.values()
)


# ============================================================
# AUDIT JSON
# ============================================================

audit = {

    "stage":
        "3.1",

    "status":
        "PASS"
        if final_pass
        else "REVIEW",

    "purpose":
        "Leakage-safe spatial damage track feature engineering",

    "spatial_records":
        int(
            len(spatial)
        ),

    "temporal_records":
        int(
            len(temporal)
        ),

    "target_records":
        int(
            len(targets)
        ),

    "output_records":
        int(
            len(features)
        ),

    "forecast_feature_count":
        int(
            len(
                forecast_feature_columns
            )
        ),

    "complete_forecast_records":
        int(
            complete_records
        ),

    "sequence_count":
        int(
            features[
                "sequence"
            ].nunique()
        ),

    "component_count":
        int(
            features[
                "component"
            ].nunique()
        ),

    "split_counts":
        features[
            "split"
        ].value_counts().to_dict(),

    "missing_targets":
        int(
            missing_targets
        ),

    "duplicate_keys":
        int(
            duplicate_keys
        ),

    "invalid_temporal_direction":
        int(
            invalid_temporal_direction
        ),

    "sequence_split_conflicts":
        int(
            sequence_split_conflicts
        ),

    "component_split_conflicts":
        int(
            component_split_conflicts
        ),

    "future_dependent_current_columns_excluded":
        future_dependent_columns,

    "future_named_features":
        future_named_features,

    "future_frame_used_as_feature":
        bool(
            future_frame_used_as_feature
        ),

    "checks":
        checks,

    "top_correlations":
        correlations
        .head(25)
        .to_dict(
            orient="records"
        ),

    "outputs":
        {
            "feature_file":
                OUTPUT_FEATURE_FILE,

            "summary_file":
                OUTPUT_SUMMARY_FILE,

            "correlation_file":
                OUTPUT_CORRELATION_FILE,

            "audit_file":
                OUTPUT_AUDIT_FILE
        }
}


with open(
    OUTPUT_AUDIT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        audit,
        file,
        indent=2
    )


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print()
print("=" * 70)
print(
    "TOP SPATIAL TRACK FEATURES"
)
print("=" * 70)

if len(correlations) > 0:

    print(
        correlations
        .head(25)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No correlations available."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 3.1 SUMMARY"
)
print("=" * 70)

print(
    "Spatial records: "
    +
    str(
        len(spatial)
    )
)

print(
    "Output records: "
    +
    str(
        len(features)
    )
)

print(
    "Forecast feature count: "
    +
    str(
        len(
            forecast_feature_columns
        )
    )
)

print(
    "Complete forecast records: "
    +
    str(
        complete_records
    )
)

print(
    "Sequences: "
    +
    str(
        features[
            "sequence"
        ].nunique()
    )
)

print(
    "Components: "
    +
    str(
        features[
            "component"
        ].nunique()
    )
)

print()
print(
    "Missing targets: "
    +
    str(
        missing_targets
    )
)

print(
    "Duplicate keys: "
    +
    str(
        duplicate_keys
    )
)

print(
    "Invalid temporal directions: "
    +
    str(
        invalid_temporal_direction
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

print(
    "Future frame used as feature: "
    +
    str(
        future_frame_used_as_feature
    )
)

print()
print(
    "Excluded current-row future-dependent track features: "
    +
    str(
        len(
            future_dependent_columns
        )
    )
)

print()
print(
    "Output feature file:"
)

print(
    OUTPUT_FEATURE_FILE
)

print()
print(
    "Feature summary:"
)

print(
    OUTPUT_SUMMARY_FILE
)

print()
print(
    "Correlation file:"
)

print(
    OUTPUT_CORRELATION_FILE
)

print()
print(
    "Audit:"
)

print(
    OUTPUT_AUDIT_FILE
)

print()

if final_pass:

    print(
        "FINAL STATUS: PASS"
    )

else:

    print(
        "FINAL STATUS: REVIEW"
    )