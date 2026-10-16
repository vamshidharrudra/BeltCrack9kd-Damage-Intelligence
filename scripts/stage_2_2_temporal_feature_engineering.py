import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 2.2
# TEMPORAL FEATURE ENGINEERING AUDIT
#
# Purpose:
#   Construct historical temporal features using ONLY frames
#   at or before the current frame t.
#
# Target:
#   crack_area_ratio at t+1
#
# Important:
#   No future frame is used to construct any feature.
#
# This stage does NOT train a model.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_0_damage_features.csv"
)

TARGET_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3_prediction_targets.csv"
)

OUTPUT_FEATURES = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_2_temporal_features.csv"
)

OUTPUT_AUDIT = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_2_temporal_feature_audit.json"
)

OUTPUT_SUMMARY = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_2_temporal_feature_summary.csv"
)


# ============================================================
# START
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.2 - TEMPORAL FEATURE ENGINEERING AUDIT")
print("=" * 70)
print("")


# ============================================================
# CHECK INPUT FILES
# ============================================================

for path, description in [
    (FEATURE_FILE, "Stage 1.0 feature file"),
    (TARGET_FILE, "Stage 1.3 target file")
]:

    if not os.path.exists(path):

        print(
            "ERROR:",
            description,
            "not found:"
        )

        print(path)

        raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading Stage 1.0 features..."
)

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature records:",
    len(features)
)


print("")
print(
    "Loading Stage 1.3 targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


# ============================================================
# DISPLAY SCHEMA
# ============================================================

print("")
print(
    "Feature columns available:",
    len(features.columns)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_feature_columns = [
    "record_id",
    "split",
    "component",
    "sequence",
    "source_image",
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "annotation_shape_count"
]


missing_features = [
    column
    for column in required_feature_columns
    if column not in features.columns
]


if missing_features:

    print("")
    print(
        "ERROR: Missing feature columns:"
    )

    for column in missing_features:
        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

print("")
print(
    "Detecting target columns..."
)


print(
    "Target columns:"
)

for column in targets.columns:

    print(
        "  -",
        column
    )


# Stage 1.3 may contain several horizon representations.
# We explicitly identify the primary t+1 target.

target_horizon_column = None

for candidate in [
    "horizon",
    "target_horizon",
    "future_horizon"
]:

    if candidate in targets.columns:

        target_horizon_column = candidate

        break


if target_horizon_column is None:

    print("")
    print(
        "ERROR: Could not find target horizon column."
    )

    raise SystemExit(1)


target_value_column = None

for candidate in [
    "target_crack_area_ratio",
    "future_crack_area_ratio",
    "target_value",
    "future_damage"
]:

    if candidate in targets.columns:

        target_value_column = candidate

        break


if target_value_column is None:

    print("")
    print(
        "ERROR: Could not identify target value column."
    )

    raise SystemExit(1)


print(
    "Horizon column:",
    target_horizon_column
)

print(
    "Target value column:",
    target_value_column
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_feature_columns = [
    "record_id",
    "component",
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "annotation_shape_count"
]


for column in numeric_feature_columns:

    features[column] = pd.to_numeric(
        features[column],
        errors="coerce"
    )


targets[target_horizon_column] = pd.to_numeric(
    targets[target_horizon_column],
    errors="coerce"
)


targets[target_value_column] = pd.to_numeric(
    targets[target_value_column],
    errors="coerce"
)


# ============================================================
# CHECK REQUIRED IDENTIFIERS
# ============================================================

identifier_columns = [
    "record_id",
    "sequence",
    "component",
    "source_image"
]


for column in identifier_columns:

    if features[column].isna().any():

        print(
            "ERROR: Missing values in",
            column
        )

        raise SystemExit(1)


# ============================================================
# EXTRACT FRAME NUMBER
# ============================================================

print("")
print(
    "Extracting chronological frame numbers..."
)


def extract_frame_number(path):

    if pd.isna(path):

        return np.nan

    filename = os.path.basename(
        str(path)
    )

    stem = os.path.splitext(
        filename
    )[0]

    try:

        return int(
            stem
        )

    except ValueError:

        return np.nan


features[
    "frame_number"
] = features[
    "source_image"
].apply(
    extract_frame_number
)


missing_frame_numbers = int(
    features[
        "frame_number"
    ]
    .isna()
    .sum()
)


print(
    "Missing frame numbers:",
    missing_frame_numbers
)


if missing_frame_numbers != 0:

    print(
        "ERROR: Could not determine all frame numbers."
    )

    raise SystemExit(1)


features[
    "frame_number"
] = features[
    "frame_number"
].astype(int)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

features = features.sort_values(
    by=[
        "sequence",
        "frame_number"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CHECK SEQUENCE STRUCTURE
# ============================================================

print("")
print("=" * 70)
print("SEQUENCE STRUCTURE CHECK")
print("=" * 70)


sequence_records = []

sequence_conflicts = 0
sequence_gap_count = 0


for sequence, group in features.groupby(
    "sequence",
    sort=False
):

    group = group.sort_values(
        "frame_number"
    )


    frames = group[
        "frame_number"
    ].to_numpy()


    if len(frames) > 1:

        gaps = np.diff(
            frames
        )

        invalid_gaps = int(
            np.sum(
                gaps != 1
            )
        )

    else:

        invalid_gaps = 0


    components = group[
        "component"
    ].nunique()

    splits = group[
        "split"
    ].nunique()


    if components != 1:

        sequence_conflicts += 1


    if splits != 1:

        sequence_conflicts += 1


    sequence_gap_count += invalid_gaps


    sequence_records.append({

        "sequence":
            sequence,

        "records":
            int(
                len(group)
            ),

        "first_frame":
            int(
                frames.min()
            ),

        "last_frame":
            int(
                frames.max()
            ),

        "frame_gaps":
            invalid_gaps,

        "components":
            int(
                components
            ),

        "splits":
            int(
                splits
            )
    })


sequence_summary = pd.DataFrame(
    sequence_records
)


print(
    "Sequences:",
    len(sequence_summary)
)

print(
    "Sequence/component/split conflicts:",
    sequence_conflicts
)

print(
    "Frame gaps:",
    sequence_gap_count
)


if sequence_conflicts != 0:

    print(
        "ERROR: Sequence integrity failed."
    )

    raise SystemExit(1)


if sequence_gap_count != 0:

    print(
        "ERROR: Frame gaps detected."
    )

    raise SystemExit(1)


# ============================================================
# TEMPORAL BASE FEATURES
# ============================================================

print("")
print("=" * 70)
print("CONSTRUCTING HISTORICAL FEATURES")
print("=" * 70)


# Features whose historical values will be used.

base_temporal_features = [

    "crack_area_ratio",

    "crack_count",

    "largest_crack_area_ratio",

    "largest_crack_area_px",

    "total_crack_area_px",

    "mean_crack_area_px",

    "total_crack_perimeter_px",

    "mean_crack_perimeter_px",

    "annotation_shape_count"
]


# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

temporal_parts = []


for sequence, group in features.groupby(
    "sequence",
    sort=False
):

    group = group.sort_values(
        "frame_number"
    ).copy()


    # --------------------------------------------------------
    # Current features
    # --------------------------------------------------------

    for column in base_temporal_features:

        group[
            "current_" + column
        ] = group[
            column
        ]


    # --------------------------------------------------------
    # Lag features
    #
    # lag_1 = t-1
    # lag_2 = t-2
    # lag_3 = t-3
    # --------------------------------------------------------

    for column in base_temporal_features:

        group[
            "lag1_" + column
        ] = group[
            column
        ].shift(1)


        group[
            "lag2_" + column
        ] = group[
            column
        ].shift(2)


        group[
            "lag3_" + column
        ] = group[
            column
        ].shift(3)


    # --------------------------------------------------------
    # First differences
    #
    # Current change:
    # t - (t-1)
    #
    # Previous change:
    # (t-1) - (t-2)
    # --------------------------------------------------------

    for column in base_temporal_features:

        current = group[
            column
        ]

        lag1 = group[
            column
        ].shift(1)

        lag2 = group[
            column
        ].shift(2)


        group[
            "delta1_" + column
        ] = (
            current
            -
            lag1
        )


        group[
            "delta2_" + column
        ] = (
            lag1
            -
            lag2
        )


    # --------------------------------------------------------
    # Acceleration-style feature
    #
    # delta(current) - delta(previous)
    #
    # This uses only t, t-1 and t-2.
    # --------------------------------------------------------

    for column in base_temporal_features:

        group[
            "accel_" + column
        ] = (
            group[
                "delta1_" + column
            ]
            -
            group[
                "delta2_" + column
            ]
        )


    # --------------------------------------------------------
    # Rolling statistics
    #
    # IMPORTANT:
    # rolling windows include only current/past frames.
    # No future frame is used.
    # --------------------------------------------------------

    for column in base_temporal_features:

        group[
            "rolling3_mean_" + column
        ] = (
            group[
                column
            ]
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )


        group[
            "rolling3_std_" + column
        ] = (
            group[
                column
            ]
            .rolling(
                window=3,
                min_periods=3
            )
            .std()
        )


    # --------------------------------------------------------
    # Recent slope for crack area ratio
    #
    # Slope over t-2, t-1, t
    #
    # Since frames are consecutive, this is equivalent to a
    # simple linear trend estimate.
    # --------------------------------------------------------

    x = np.array(
        [0.0, 1.0, 2.0]
    )

    x_mean = x.mean()

    x_centered = (
        x
        -
        x_mean
    )

    denominator = np.sum(
        x_centered ** 2
    )


    values = group[
        "crack_area_ratio"
    ].to_numpy(
        dtype=float
    )


    slope_values = np.full(
        len(values),
        np.nan
    )


    for i in range(
        2,
        len(values)
    ):

        y = values[
            i - 2:
            i + 1
        ]


        if np.isfinite(
            y
        ).all():

            y_mean = y.mean()

            slope_values[
                i
            ] = (
                np.sum(
                    x_centered
                    *
                    (
                        y
                        -
                        y_mean
                    )
                )
                /
                denominator
            )


    group[
        "recent_crack_area_slope"
    ] = slope_values


    # --------------------------------------------------------
    # Recent growth indicators
    # --------------------------------------------------------

    group[
        "recent_crack_area_growth"
    ] = (
        group[
            "crack_area_ratio"
        ]
        -
        group[
            "crack_area_ratio"
        ].shift(2)
    )


    group[
        "recent_crack_count_growth"
    ] = (
        group[
            "crack_count"
        ]
        -
        group[
            "crack_count"
        ].shift(2)
    )


    temporal_parts.append(
        group
    )


temporal_features = pd.concat(
    temporal_parts,
    ignore_index=True
)


# ============================================================
# MERGE PRIMARY T+1 TARGET
# ============================================================

print("")
print("=" * 70)
print("ATTACHING PRIMARY T+1 TARGET")
print("=" * 70)


primary_targets = targets[
    targets[
        target_horizon_column
    ]
    ==
    1
].copy()


print(
    "Primary target records:",
    len(primary_targets)
)


# Detect target frame column.

future_frame_column = None

for candidate in [
    "future_frame",
    "target_frame",
    "future_frame_number"
]:

    if candidate in primary_targets.columns:

        future_frame_column = candidate

        break


current_frame_column = None

for candidate in [
    "current_frame",
    "frame_number",
    "source_frame"
]:

    if candidate in primary_targets.columns:

        current_frame_column = candidate

        break


if future_frame_column is None:

    print(
        "ERROR: Future frame column not found."
    )

    raise SystemExit(1)


if current_frame_column is None:

    print(
        "ERROR: Current frame column not found."
    )

    raise SystemExit(1)


print(
    "Current target-frame column:",
    current_frame_column
)

print(
    "Future target-frame column:",
    future_frame_column
)


# ------------------------------------------------------------
# Build robust target key.
# ------------------------------------------------------------

target_key_columns = [
    "sequence",
    current_frame_column,
    future_frame_column
]


for column in target_key_columns:

    if column not in primary_targets.columns:

        print(
            "ERROR: Missing target key column:",
            column
        )

        raise SystemExit(1)


primary_targets[
    current_frame_column
] = pd.to_numeric(
    primary_targets[
        current_frame_column
    ],
    errors="coerce"
)


primary_targets[
    future_frame_column
] = pd.to_numeric(
    primary_targets[
        future_frame_column
    ],
    errors="coerce"
)


# Rename to stable names.

primary_targets = primary_targets.rename(
    columns={
        current_frame_column:
            "target_current_frame",

        future_frame_column:
            "target_future_frame",

        target_value_column:
            "target_t_plus_1"
    }
)


primary_targets[
    "target_current_frame"
] = primary_targets[
    "target_current_frame"
].astype(int)


primary_targets[
    "target_future_frame"
] = primary_targets[
    "target_future_frame"
].astype(int)


# ============================================================
# MERGE
# ============================================================

merged = temporal_features.merge(
    primary_targets[
        [
            "sequence",
            "target_current_frame",
            "target_future_frame",
            "target_t_plus_1"
        ]
    ],
    left_on=[
        "sequence",
        "frame_number"
    ],
    right_on=[
        "sequence",
        "target_current_frame"
    ],
    how="inner"
)


print(
    "Merged records:",
    len(merged)
)


# ============================================================
# TEMPORAL TARGET VALIDATION
# ============================================================

invalid_target_direction = int(
    (
        merged[
            "target_future_frame"
        ]
        <=
        merged[
            "frame_number"
        ]
    ).sum()
)


wrong_target_gap = int(
    (
        (
            merged[
                "target_future_frame"
            ]
            -
            merged[
                "frame_number"
            ]
        )
        !=
        1
    ).sum()
)


print(
    "Invalid target direction:",
    invalid_target_direction
)

print(
    "Invalid target frame gap:",
    wrong_target_gap
)


if invalid_target_direction != 0:

    print(
        "ERROR: Future target direction invalid."
    )

    raise SystemExit(1)


if wrong_target_gap != 0:

    print(
        "ERROR: Target is not t+1."
    )

    raise SystemExit(1)


# ============================================================
# FUTURE LEAKAGE AUDIT
# ============================================================

print("")
print("=" * 70)
print("FUTURE LEAKAGE AUDIT")
print("=" * 70)


# Columns explicitly containing future information are forbidden
# as model features.

forbidden_keywords = [
    "future",
    "target",
    "t_plus_1",
    "t_plus_3",
    "t_plus_5"
]


candidate_feature_columns = []


metadata_columns = [
    "record_id",
    "split",
    "component",
    "sequence",
    "source_image",
    "source_json",
    "original_image_filename",
    "original_json_filename",
    "canonical_image_filename",
    "canonical_json_filename",
    "frame_number",
    "target_current_frame",
    "target_future_frame",
    "target_t_plus_1"
]


for column in merged.columns:

    if column in metadata_columns:

        continue


    lowered = column.lower()


    if any(
        keyword in lowered
        for keyword in forbidden_keywords
    ):

        continue


    candidate_feature_columns.append(
        column
    )


print(
    "Candidate temporal feature count:",
    len(
        candidate_feature_columns
    )
)


future_keyword_features = []


for column in candidate_feature_columns:

    lowered = column.lower()

    if any(
        keyword in lowered
        for keyword in [
            "future",
            "target",
            "t_plus_1",
            "t_plus_3",
            "t_plus_5"
        ]
    ):

        future_keyword_features.append(
            column
        )


print(
    "Future-information feature names:",
    len(
        future_keyword_features
    )
)


if future_keyword_features:

    print(
        "ERROR: Future-information columns detected."
    )

    for column in future_keyword_features:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# CHECK LAG FEATURES AGAINST ORIGINAL DATA
# ============================================================

print("")
print(
    "Checking lag feature chronology..."
)


lag_validation_errors = 0


for sequence, group in features.groupby(
    "sequence",
    sort=False
):

    group = group.sort_values(
        "frame_number"
    ).reset_index(
        drop=True
    )


    for i in range(
        len(group)
    ):

        current_frame = int(
            group.loc[
                i,
                "frame_number"
            ]
        )


        if i >= 1:

            expected_lag1 = int(
                group.loc[
                    i - 1,
                    "frame_number"
                ]
            )

            if current_frame - expected_lag1 != 1:

                lag_validation_errors += 1


        if i >= 2:

            expected_lag2 = int(
                group.loc[
                    i - 2,
                    "frame_number"
                ]
            )

            if current_frame - expected_lag2 != 2:

                lag_validation_errors += 1


        if i >= 3:

            expected_lag3 = int(
                group.loc[
                    i - 3,
                    "frame_number"
                ]
            )

            if current_frame - expected_lag3 != 3:

                lag_validation_errors += 1


print(
    "Lag chronology errors:",
    lag_validation_errors
)


if lag_validation_errors != 0:

    print(
        "ERROR: Lag chronology validation failed."
    )

    raise SystemExit(1)


# ============================================================
# CHECK FEATURE AVAILABILITY
# ============================================================

print("")
print("=" * 70)
print("TEMPORAL FEATURE AVAILABILITY")
print("=" * 70)


availability_records = []


for column in candidate_feature_columns:

    missing_count = int(
        merged[
            column
        ]
        .isna()
        .sum()
    )


    available_count = (
        len(merged)
        -
        missing_count
    )


    availability_records.append({

        "feature":
            column,

        "records":
            int(
                len(merged)
            ),

        "available":
            available_count,

        "missing":
            missing_count,

        "availability_fraction":
            float(
                available_count
                /
                len(merged)
            )
    })


availability_df = pd.DataFrame(
    availability_records
)


# ============================================================
# CORRELATION WITH T+1 TARGET
# ============================================================

print("")
print("=" * 70)
print("TEMPORAL FEATURE-TARGET CORRELATION")
print("=" * 70)


correlation_records = []


for column in candidate_feature_columns:

    valid = merged[
        [
            column,
            "target_t_plus_1"
        ]
    ].dropna()


    if len(valid) < 3:

        correlation = np.nan

    elif valid[
        column
    ].nunique() <= 1:

        correlation = np.nan

    elif valid[
        "target_t_plus_1"
    ].nunique() <= 1:

        correlation = np.nan

    else:

        correlation = float(
            valid[
                column
            ].corr(
                valid[
                    "target_t_plus_1"
                ],
                method="spearman"
            )
        )


    correlation_records.append({

        "feature":
            column,

        "records":
            int(
                len(valid)
            ),

        "spearman_correlation":
            correlation,

        "absolute_spearman":
            (
                abs(correlation)
                if np.isfinite(
                    correlation
                )
                else np.nan
            )
    })


correlation_df = pd.DataFrame(
    correlation_records
)


correlation_df = correlation_df.sort_values(
    by="absolute_spearman",
    ascending=False,
    na_position="last"
).reset_index(
    drop=True
)


# ============================================================
# TEMPORAL FEATURES ONLY
# ============================================================

temporal_only_mask = (
    correlation_df[
        "feature"
    ].str.startswith(
        (
            "lag1_",
            "lag2_",
            "lag3_",
            "delta1_",
            "delta2_",
            "accel_",
            "rolling3_",
            "recent_"
        )
    )
)


temporal_correlations = correlation_df[
    temporal_only_mask
].copy()


print(
    "Top temporal features:"
)

print(
    temporal_correlations.head(
        20
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# CHECK CURRENT VS HISTORICAL FEATURE CORRELATIONS
# ============================================================

print("")
print("=" * 70)
print("CURRENT VS HISTORICAL FEATURE INFORMATION")
print("=" * 70)


comparison_features = [
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px"
]


current_history_records = []


for column in comparison_features:

    if column not in merged.columns:

        continue


    current_corr = correlation_df[
        correlation_df[
            "feature"
        ]
        ==
        column
    ]


    lag_name = (
        "lag1_"
        +
        column
    )


    lag_corr = correlation_df[
        correlation_df[
            "feature"
        ]
        ==
        lag_name
    ]


    current_value = np.nan
    lag_value = np.nan


    if len(
        current_corr
    ) > 0:

        current_value = float(
            current_corr.iloc[
                0
            ][
                "spearman_correlation"
            ]
        )


    if len(
        lag_corr
    ) > 0:

        lag_value = float(
            lag_corr.iloc[
                0
            ][
                "spearman_correlation"
            ]
        )


    current_history_records.append({

        "base_feature":
            column,

        "current_t_correlation":
            current_value,

        "lag1_t_minus_1_correlation":
            lag_value,

        "lag1_minus_current_absolute_correlation":
            (
                abs(lag_value)
                -
                abs(current_value)
                if np.isfinite(
                    lag_value
                )
                and np.isfinite(
                    current_value
                )
                else np.nan
            )
    })


current_history_df = pd.DataFrame(
    current_history_records
)


# ============================================================
# SAVE FEATURE DATA
# ============================================================

# Keep all original identifiers plus engineered features.
#
# This file is intentionally a feature-analysis table.
# Raw images and annotations are not copied or modified.

merged.to_csv(
    OUTPUT_FEATURES,
    index=False
)


# ============================================================
# SAVE FEATURE SUMMARY
# ============================================================

summary_df = availability_df.merge(
    correlation_df[
        [
            "feature",
            "spearman_correlation",
            "absolute_spearman"
        ]
    ],
    on="feature",
    how="left"
)


summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# FINAL AUDIT VALUES
# ============================================================

usable_records = int(
    len(merged)
)


rows_with_all_three_lags = int(
    merged[
        [
            "lag3_crack_area_ratio",
            "lag3_crack_count",
            "lag3_largest_crack_area_ratio"
        ]
    ]
    .notna()
    .all(
        axis=1
    )
    .sum()
)


rows_with_full_temporal_history = int(
    merged[
        [
            "lag3_crack_area_ratio",
            "lag3_crack_count",
            "lag3_largest_crack_area_ratio",
            "rolling3_mean_crack_area_ratio",
            "rolling3_std_crack_area_ratio",
            "recent_crack_area_slope"
        ]
    ]
    .notna()
    .all(
        axis=1
    )
    .sum()
)


# ============================================================
# BUILD AUDIT SUMMARY
# ============================================================

audit = {

    "stage":
        "2.2",

    "title":
        "Temporal Feature Engineering Audit",

    "status":
        "PASS",

    "input_feature_records":
        int(
            len(features)
        ),

    "primary_target_records":
        int(
            len(primary_targets)
        ),

    "merged_records":
        usable_records,

    "sequences":
        int(
            features[
                "sequence"
            ].nunique()
        ),

    "components":
        int(
            features[
                "component"
            ].nunique()
        ),

    "sequence_conflicts":
        sequence_conflicts,

    "frame_gaps":
        sequence_gap_count,

    "invalid_target_direction":
        invalid_target_direction,

    "wrong_target_frame_gap":
        wrong_target_gap,

    "lag_chronology_errors":
        lag_validation_errors,

    "candidate_temporal_feature_count":
        len(
            candidate_feature_columns
        ),

    "future_information_feature_count":
        len(
            future_keyword_features
        ),

    "records_with_all_three_lags":
        rows_with_all_three_lags,

    "records_with_full_temporal_history":
        rows_with_full_temporal_history,

    "top_temporal_features":
        temporal_correlations.head(
            20
        ).to_dict(
            orient="records"
        ),

    "current_vs_history":
        current_history_df.to_dict(
            orient="records"
        ),

    "outputs": {

        "temporal_features":
            OUTPUT_FEATURES,

        "feature_summary":
            OUTPUT_SUMMARY,

        "audit":
            OUTPUT_AUDIT
    }
}


# ============================================================
# SAVE AUDIT
# ============================================================

with open(
    OUTPUT_AUDIT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        audit,
        file,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.2 SUMMARY")
print("=" * 70)

print("")

print(
    "Input feature records:",
    len(features)
)

print(
    "Primary t+1 target records:",
    len(primary_targets)
)

print(
    "Merged records:",
    usable_records
)

print(
    "Sequences:",
    features[
        "sequence"
    ].nunique()
)

print(
    "Components:",
    features[
        "component"
    ].nunique()
)

print("")

print(
    "Sequence conflicts:",
    sequence_conflicts
)

print(
    "Frame gaps:",
    sequence_gap_count
)

print(
    "Invalid target direction:",
    invalid_target_direction
)

print(
    "Wrong target frame gap:",
    wrong_target_gap
)

print(
    "Lag chronology errors:",
    lag_validation_errors
)

print("")

print(
    "Candidate temporal features:",
    len(
        candidate_feature_columns
    )
)

print(
    "Future-information features:",
    len(
        future_keyword_features
    )
)

print(
    "Records with all three lags:",
    rows_with_all_three_lags
)

print(
    "Records with full temporal history:",
    rows_with_full_temporal_history
)

print("")

print(
    "Top temporal features by absolute Spearman correlation:"
)

print(
    temporal_correlations[
        [
            "feature",
            "spearman_correlation"
        ]
    ]
    .head(
        10
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)

print("")

print(
    "Temporal feature file:"
)

print(
    OUTPUT_FEATURES
)

print("")

print(
    "Feature summary:"
)

print(
    OUTPUT_SUMMARY
)

print("")

print(
    "Audit:"
)

print(
    OUTPUT_AUDIT
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")