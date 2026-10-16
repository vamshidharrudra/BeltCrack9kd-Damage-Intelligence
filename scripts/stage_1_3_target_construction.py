import os
import json
import re
import numpy as np
import pandas as pd


# ============================================================
# STAGE 1.3
# PREDICTION TARGET CONSTRUCTION AND TARGET AUDIT
#
# Primary target:
#   Future crack_area_ratio at t+1
#
# Secondary targets:
#   Future crack_area_ratio at t+3
#   Future crack_area_ratio at t+5
#
# No model training is performed here.
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_0_damage_features.csv"
)

PAIR_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_2_temporal_pairs.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

TARGET_FILE = os.path.join(
    OUTPUT_DIR,
    "stage_1_3_prediction_targets.csv"
)

SEQUENCE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_3_target_sequence_summary.csv"
)

SPLIT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_3_target_split_summary.csv"
)

JSON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_3_target_audit.json"
)


# ------------------------------------------------------------
# TARGET HORIZONS
# ------------------------------------------------------------

PRIMARY_HORIZON = 1

SECONDARY_HORIZONS = [
    3,
    5
]

ALL_HORIZONS = [
    PRIMARY_HORIZON
] + SECONDARY_HORIZONS


# ------------------------------------------------------------
# REQUIRED CURRENT FEATURES
# ------------------------------------------------------------

CURRENT_FEATURES = [
    "crack_count",
    "total_crack_area_px",
    "crack_area_ratio",
    "largest_crack_area_px",
    "largest_crack_area_ratio",
    "smallest_crack_area_px",
    "mean_crack_area_px",
    "median_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "max_crack_perimeter_px",
    "mean_crack_bbox_width_px",
    "mean_crack_bbox_height_px",
    "mean_crack_bbox_area_px",
    "max_crack_bbox_width_px",
    "max_crack_bbox_height_px",
    "mean_crack_centroid_x_px",
    "mean_crack_centroid_y_px",
    "mean_crack_centroid_x_norm",
    "mean_crack_centroid_y_norm",
    "crack_centroid_x_std_px",
    "crack_centroid_y_std_px",
    "crack_centroid_x_std_norm",
    "crack_centroid_y_std_norm",
    "crack_bbox_union_width_px",
    "crack_bbox_union_height_px"
]


# ------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------

def parse_frame_number(filename):

    if pd.isna(filename):
        return np.nan

    name = os.path.basename(
        str(filename)
    )

    stem = os.path.splitext(
        name
    )[0]

    match = re.search(
        r"(\d+)$",
        stem
    )

    if match:
        return int(
            match.group(1)
        )

    return np.nan


def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def percentile(values, q):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            q
        )
    )


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.3 - PREDICTION TARGET CONSTRUCTION AND AUDIT")
print("=" * 70)
print("")


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

if not os.path.exists(
    FEATURE_FILE
):

    print(
        "ERROR: Stage 1.0 feature file not found:"
    )

    print(
        FEATURE_FILE
    )

    raise SystemExit(1)


if not os.path.exists(
    PAIR_FILE
):

    print(
        "ERROR: Stage 1.2 temporal-pair file not found:"
    )

    print(
        PAIR_FILE
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

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
    "Loading Stage 1.2 temporal pairs..."
)

pairs = pd.read_csv(
    PAIR_FILE
)

print(
    "Temporal pair records:",
    len(pairs)
)


# ------------------------------------------------------------
# REQUIRED PAIR COLUMNS
# ------------------------------------------------------------

required_pair_columns = [
    "sequence",
    "split",
    "component",
    "horizon",
    "current_frame",
    "future_frame",
    "current_record_id",
    "future_record_id",
    "crack_area_ratio_current",
    "crack_area_ratio_future"
]


missing_pair_columns = [
    column
    for column in required_pair_columns
    if column not in pairs.columns
]


if missing_pair_columns:

    print("")
    print(
        "ERROR: Required Stage 1.2 columns are missing:"
    )

    for column in missing_pair_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ------------------------------------------------------------
# CURRENT FEATURE AVAILABILITY
# ------------------------------------------------------------

available_current_features = [
    feature
    for feature in CURRENT_FEATURES
    if feature in features.columns
]


print("")
print(
    "Current features available:",
    len(
        available_current_features
    )
)


for feature in available_current_features:

    print(
        "  -",
        feature
    )


# ------------------------------------------------------------
# FILTER REQUIRED HORIZONS
# ------------------------------------------------------------

target_pairs = pairs[
    pairs[
        "horizon"
    ].isin(
        ALL_HORIZONS
    )
].copy()


print("")
print(
    "Target horizons:",
    ALL_HORIZONS
)

print(
    "Filtered target pairs:",
    len(target_pairs)
)


# ------------------------------------------------------------
# VERIFY SEQUENCE INTEGRITY
# ------------------------------------------------------------

print("")
print(
    "Checking sequence integrity..."
)


sequence_conflicts = []


for sequence, group in target_pairs.groupby(
    "sequence"
):

    unique_splits = (
        group[
            "split"
        ]
        .dropna()
        .unique()
    )


    unique_components = (
        group[
            "component"
        ]
        .dropna()
        .unique()
    )


    if len(unique_splits) != 1:

        sequence_conflicts.append({
            "sequence":
                sequence,

            "problem":
                "multiple_splits",

            "values":
                [
                    str(value)
                    for value
                    in unique_splits
                ]
        })


    if len(unique_components) != 1:

        sequence_conflicts.append({
            "sequence":
                sequence,

            "problem":
                "multiple_components",

            "values":
                [
                    str(value)
                    for value
                    in unique_components
                ]
        })


print(
    "Sequence conflicts:",
    len(
        sequence_conflicts
    )
)


# ------------------------------------------------------------
# VERIFY FRAME DIRECTION
# ------------------------------------------------------------

print("")
print(
    "Checking temporal direction..."
)


wrong_direction = (
    target_pairs[
        "future_frame"
    ]
    <=
    target_pairs[
        "current_frame"
    ]
)


wrong_direction_count = int(
    wrong_direction.sum()
)


print(
    "Invalid future-frame direction:",
    wrong_direction_count
)


# ------------------------------------------------------------
# VERIFY EXPECTED FRAME GAP
# ------------------------------------------------------------

target_pairs[
    "actual_frame_gap"
] = (
    target_pairs[
        "future_frame"
    ]
    -
    target_pairs[
        "current_frame"
    ]
)


target_pairs[
    "frame_gap_valid"
] = (
    target_pairs[
        "actual_frame_gap"
    ]
    ==
    target_pairs[
        "horizon"
    ]
)


invalid_gap_count = int(
    (
        ~target_pairs[
            "frame_gap_valid"
        ]
    ).sum()
)


print(
    "Invalid frame gaps:",
    invalid_gap_count
)


# ------------------------------------------------------------
# VERIFY TARGET VALUES
# ------------------------------------------------------------

print("")
print(
    "Checking target values..."
)


target_pairs[
    "target_t1"
] = np.nan

target_pairs[
    "target_t3"
] = np.nan

target_pairs[
    "target_t5"
] = np.nan


for horizon, column in [
    (1, "target_t1"),
    (3, "target_t3"),
    (5, "target_t5")
]:

    mask = (
        target_pairs[
            "horizon"
        ]
        ==
        horizon
    )


    target_pairs.loc[
        mask,
        column
    ] = pd.to_numeric(
        target_pairs.loc[
            mask,
            "crack_area_ratio_future"
        ],
        errors="coerce"
    )


# ------------------------------------------------------------
# TARGET VALIDITY
# ------------------------------------------------------------

target_pairs[
    "target_valid"
] = (
    np.isfinite(
        target_pairs[
            "crack_area_ratio_future"
        ]
    )
    &
    (
        target_pairs[
            "crack_area_ratio_future"
        ] >= 0
    )
)


invalid_target_count = int(
    (
        ~target_pairs[
            "target_valid"
        ]
    ).sum()
)


print(
    "Invalid future damage targets:",
    invalid_target_count
)


# ------------------------------------------------------------
# NEGATIVE DAMAGE CHECK
# ------------------------------------------------------------

negative_target_count = int(
    (
        target_pairs[
            "crack_area_ratio_future"
        ]
        <
        0
    )
    .fillna(False)
    .sum()
)


print(
    "Negative damage targets:",
    negative_target_count
)


# ------------------------------------------------------------
# BUILD ML-READY TARGET TABLE
#
# One row represents:
#
#   current frame t
#       ->
#   future frame t+h
#
# Current features are copied from Stage 1.0.
# ------------------------------------------------------------

print("")
print(
    "Constructing prediction-target table..."
)


target_records = []


for _, row in target_pairs.iterrows():

    if not row[
        "target_valid"
    ]:

        continue


    record = {

        "sequence":
            row[
                "sequence"
            ],

        "split":
            row[
                "split"
            ],

        "component":
            int(
                row[
                    "component"
                ]
            ),

        "horizon":
            int(
                row[
                    "horizon"
                ]
            ),

        "current_frame":
            int(
                row[
                    "current_frame"
                ]
            ),

        "future_frame":
            int(
                row[
                    "future_frame"
                ]
            ),

        "current_record_id":
            int(
                row[
                    "current_record_id"
                ]
            ),

        "future_record_id":
            int(
                row[
                    "future_record_id"
                ]
            ),

        "target_crack_area_ratio":
            safe_float(
                row[
                    "crack_area_ratio_future"
                ]
            ),

        "current_crack_area_ratio":
            safe_float(
                row[
                    "crack_area_ratio_current"
                ]
            )
    }


    # --------------------------------------------------------
    # Current feature values
    # --------------------------------------------------------

    current_record = features[
        features[
            "record_id"
        ]
        ==
        row[
            "current_record_id"
        ]
    ]


    if len(
        current_record
    ) != 1:

        continue


    current_record = (
        current_record.iloc[0]
    )


    for feature in available_current_features:

        record[
            feature
        ] = safe_float(
            current_record[
                feature
            ]
        )


    # --------------------------------------------------------
    # Target change
    # --------------------------------------------------------

    current_damage = safe_float(
        row[
            "crack_area_ratio_current"
        ]
    )


    future_damage = safe_float(
        row[
            "crack_area_ratio_future"
        ]
    )


    if (
        np.isfinite(
            current_damage
        )
        and
        np.isfinite(
            future_damage
        )
    ):

        record[
            "target_damage_change"
        ] = (
            future_damage
            -
            current_damage
        )


        if abs(
            current_damage
        ) > 1e-12:

            record[
                "target_damage_relative_change"
            ] = (
                (
                    future_damage
                    -
                    current_damage
                )
                /
                abs(
                    current_damage
                )
            )

        else:

            record[
                "target_damage_relative_change"
            ] = np.nan

    else:

        record[
            "target_damage_change"
        ] = np.nan

        record[
            "target_damage_relative_change"
        ] = np.nan


    target_records.append(
        record
    )


targets = pd.DataFrame(
    target_records
)


# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

if not targets.empty:

    targets = targets.sort_values(
        by=[
            "split",
            "sequence",
            "current_frame",
            "horizon"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )


# ------------------------------------------------------------
# SAVE TARGET DATA
# ------------------------------------------------------------

targets.to_csv(
    TARGET_FILE,
    index=False
)


print(
    "Saved:",
    TARGET_FILE
)


# ------------------------------------------------------------
# SEQUENCE SUMMARY
# ------------------------------------------------------------

sequence_records = []


for sequence, group in targets.groupby(
    "sequence"
):

    for horizon in ALL_HORIZONS:

        h = group[
            group[
                "horizon"
            ]
            ==
            horizon
        ]


        if h.empty:
            continue


        values = pd.to_numeric(
            h[
                "target_crack_area_ratio"
            ],
            errors="coerce"
        )


        changes = pd.to_numeric(
            h[
                "target_damage_change"
            ],
            errors="coerce"
        )


        sequence_records.append({

            "sequence":
                sequence,

            "split":
                h[
                    "split"
                ].iloc[0],

            "component":
                int(
                    h[
                        "component"
                    ].iloc[0]
                ),

            "horizon":
                int(
                    horizon
                ),

            "target_records":
                int(
                    len(h)
                ),

            "target_mean":
                safe_float(
                    values.mean()
                ),

            "target_std":
                safe_float(
                    values.std()
                ),

            "target_min":
                safe_float(
                    values.min()
                ),

            "target_max":
                safe_float(
                    values.max()
                ),

            "target_p25":
                percentile(
                    values,
                    25
                ),

            "target_median":
                percentile(
                    values,
                    50
                ),

            "target_p75":
                percentile(
                    values,
                    75
                ),

            "mean_future_damage_change":
                safe_float(
                    changes.mean()
                ),

            "median_future_damage_change":
                safe_float(
                    changes.median()
                ),

            "positive_future_change_fraction":
                safe_float(
                    (
                        changes > 0
                    ).mean()
                ),

            "negative_future_change_fraction":
                safe_float(
                    (
                        changes < 0
                    ).mean()
                ),

            "zero_future_change_fraction":
                safe_float(
                    (
                        changes == 0
                    ).mean()
                )
        })


sequence_summary = pd.DataFrame(
    sequence_records
)


sequence_summary.to_csv(
    SEQUENCE_OUTPUT,
    index=False
)


print("")
print(
    "Sequence target summary saved:"
)

print(
    SEQUENCE_OUTPUT
)


# ------------------------------------------------------------
# SPLIT SUMMARY
# ------------------------------------------------------------

split_records = []


for split, group in targets.groupby(
    "split"
):

    for horizon in ALL_HORIZONS:

        h = group[
            group[
                "horizon"
            ]
            ==
            horizon
        ]


        if h.empty:
            continue


        target_values = pd.to_numeric(
            h[
                "target_crack_area_ratio"
            ],
            errors="coerce"
        )


        current_values = pd.to_numeric(
            h[
                "current_crack_area_ratio"
            ],
            errors="coerce"
        )


        split_records.append({

            "split":
                split,

            "horizon":
                int(
                    horizon
                ),

            "records":
                int(
                    len(h)
                ),

            "sequences":
                int(
                    h[
                        "sequence"
                    ].nunique()
                ),

            "components":
                int(
                    h[
                        "component"
                    ].nunique()
                ),

            "target_mean":
                safe_float(
                    target_values.mean()
                ),

            "target_std":
                safe_float(
                    target_values.std()
                ),

            "target_min":
                safe_float(
                    target_values.min()
                ),

            "target_max":
                safe_float(
                    target_values.max()
                ),

            "target_p25":
                percentile(
                    target_values,
                    25
                ),

            "target_median":
                percentile(
                    target_values,
                    50
                ),

            "target_p75":
                percentile(
                    target_values,
                    75
                ),

            "current_mean":
                safe_float(
                    current_values.mean()
                ),

            "mean_target_change":
                safe_float(
                    h[
                        "target_damage_change"
                    ].mean()
                )
        })


split_summary = pd.DataFrame(
    split_records
)


split_summary.to_csv(
    SPLIT_OUTPUT,
    index=False
)


print("")
print(
    "Split target summary saved:"
)

print(
    SPLIT_OUTPUT
)


# ------------------------------------------------------------
# PRIMARY TARGET SUMMARY
# ------------------------------------------------------------

primary = targets[
    targets[
        "horizon"
    ]
    ==
    PRIMARY_HORIZON
].copy()


primary_target = pd.to_numeric(
    primary[
        "target_crack_area_ratio"
    ],
    errors="coerce"
)


primary_current = pd.to_numeric(
    primary[
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


primary_change = pd.to_numeric(
    primary[
        "target_damage_change"
    ],
    errors="coerce"
)


# ------------------------------------------------------------
# SEQUENCE DISJOINTNESS
# ------------------------------------------------------------

sequence_split_map = (
    targets[
        [
            "sequence",
            "split"
        ]
    ]
    .drop_duplicates()
    .groupby(
        "sequence"
    )[
        "split"
    ]
    .nunique()
)


sequence_split_conflicts = int(
    (
        sequence_split_map > 1
    ).sum()
)


# ------------------------------------------------------------
# COMPONENT DISJOINTNESS
# ------------------------------------------------------------

component_split_map = (
    targets[
        [
            "component",
            "split"
        ]
    ]
    .drop_duplicates()
    .groupby(
        "component"
    )[
        "split"
    ]
    .nunique()
)


component_split_conflicts = int(
    (
        component_split_map > 1
    ).sum()
)


# ------------------------------------------------------------
# RECORD ID SELF TARGET CHECK
# ------------------------------------------------------------

self_target_count = int(
    (
        targets[
            "current_record_id"
        ]
        ==
        targets[
            "future_record_id"
        ]
    ).sum()
)


# ------------------------------------------------------------
# TARGET RANGE CHECK
# ------------------------------------------------------------

target_out_of_range = int(
    (
        (
            primary_target < 0
        )
        |
        (
            primary_target > 1
        )
    )
    .fillna(False)
    .sum()
)


# ------------------------------------------------------------
# CURRENT DAMAGE RANGE CHECK
# ------------------------------------------------------------

current_out_of_range = int(
    (
        (
            primary_current < 0
        )
        |
        (
            primary_current > 1
        )
    )
    .fillna(False)
    .sum()
)


# ------------------------------------------------------------
# TARGET SUMMARY
# ------------------------------------------------------------

primary_summary = {

    "records":
        int(
            len(primary)
        ),

    "sequences":
        int(
            primary[
                "sequence"
            ].nunique()
        ),

    "components":
        int(
            primary[
                "component"
            ].nunique()
        ),

    "target_mean":
        safe_float(
            primary_target.mean()
        ),

    "target_std":
        safe_float(
            primary_target.std()
        ),

    "target_min":
        safe_float(
            primary_target.min()
        ),

    "target_max":
        safe_float(
            primary_target.max()
        ),

    "target_p25":
        percentile(
            primary_target,
            25
        ),

    "target_median":
        percentile(
            primary_target,
            50
        ),

    "target_p75":
        percentile(
            primary_target,
            75
        ),

    "current_mean":
        safe_float(
            primary_current.mean()
        ),

    "future_mean":
        safe_float(
            primary_target.mean()
        ),

    "mean_damage_change":
        safe_float(
            primary_change.mean()
        ),

    "median_damage_change":
        safe_float(
            primary_change.median()
        ),

    "positive_change_fraction":
        safe_float(
            (
                primary_change > 0
            ).mean()
        ),

    "negative_change_fraction":
        safe_float(
            (
                primary_change < 0
            ).mean()
        ),

    "zero_change_fraction":
        safe_float(
            (
                primary_change == 0
            ).mean()
        )
}


# ------------------------------------------------------------
# AUDIT CHECKS
# ------------------------------------------------------------

checks = {

    "feature_file_exists":
        os.path.exists(
            FEATURE_FILE
        ),

    "pair_file_exists":
        os.path.exists(
            PAIR_FILE
        ),

    "sequence_conflicts":
        sequence_conflicts == 0,

    "wrong_temporal_direction":
        wrong_direction_count == 0,

    "invalid_frame_gaps":
        invalid_gap_count == 0,

    "invalid_targets":
        invalid_target_count == 0,

    "negative_targets":
        negative_target_count == 0,

    "primary_target_out_of_range":
        target_out_of_range == 0,

    "current_damage_out_of_range":
        current_out_of_range == 0,

    "self_target_pairs":
        self_target_count == 0,

    "sequence_split_conflicts":
        sequence_split_conflicts == 0,

    "component_split_conflicts":
        component_split_conflicts == 0,

    "primary_target_not_empty":
        len(primary) > 0,

    "all_primary_sequences_present":
        primary[
            "sequence"
        ].nunique() == 29
}


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

final_status = all(
    checks.values()
)


# ------------------------------------------------------------
# JSON REPORT
# ------------------------------------------------------------

report = {

    "stage":
        "1.3",

    "title":
        "Prediction Target Construction and Target Audit",

    "status":
        "PASS"
        if final_status
        else "REVIEW",

    "purpose": [

        "Construct future physical-damage prediction targets.",

        "Use t+1 as the primary prediction horizon.",

        "Keep t+3 and t+5 as secondary horizons.",

        "Verify sequence boundaries and target validity.",

        "Verify train/validation/test isolation.",

        "Do not create artificial severity labels at this stage."
    ],


    "input": {

        "feature_file":
            FEATURE_FILE,

        "temporal_pair_file":
            PAIR_FILE,

        "feature_records":
            int(
                len(features)
            ),

        "temporal_pairs":
            int(
                len(pairs)
            )
    },


    "target_definition": {

        "primary":
            "crack_area_ratio at t+1",

        "secondary": [
            "crack_area_ratio at t+3",
            "crack_area_ratio at t+5"
        ],

        "problem_type":
            "future physical-damage regression"
    },


    "primary_target_summary":
        primary_summary,


    "audit_counts": {

        "sequence_conflicts":
            len(
                sequence_conflicts
            ),

        "wrong_direction":
            wrong_direction_count,

        "invalid_frame_gaps":
            invalid_gap_count,

        "invalid_targets":
            invalid_target_count,

        "negative_targets":
            negative_target_count,

        "target_out_of_range":
            target_out_of_range,

        "current_out_of_range":
            current_out_of_range,

        "self_target_pairs":
            self_target_count,

        "sequence_split_conflicts":
            sequence_split_conflicts,

        "component_split_conflicts":
            component_split_conflicts
    },


    "checks":
        checks,


    "outputs": {

        "prediction_targets":
            TARGET_FILE,

        "sequence_summary":
            SEQUENCE_OUTPUT,

        "split_summary":
            SPLIT_OUTPUT,

        "json_report":
            JSON_OUTPUT
    },


    "next_stage":
        "Feature/target leakage audit before ML training."
}


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2,
        allow_nan=False
    )


# ------------------------------------------------------------
# CONSOLE SUMMARY
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.3 SUMMARY")
print("=" * 70)

print(
    "Total target records:",
    len(targets)
)

print(
    "Primary t+1 records:",
    len(primary)
)

print(
    "Sequences in primary target:",
    primary[
        "sequence"
    ].nunique()
)

print(
    "Components in primary target:",
    primary[
        "component"
    ].nunique()
)

print("")

print(
    "PRIMARY TARGET: crack_area_ratio at t+1"
)

print(
    "Target mean:",
    round(
        primary_summary[
            "target_mean"
        ],
        8
    )
)

print(
    "Target median:",
    round(
        primary_summary[
            "target_median"
        ],
        8
    )
)

print(
    "Target min:",
    round(
        primary_summary[
            "target_min"
        ],
        8
    )
)

print(
    "Target max:",
    round(
        primary_summary[
            "target_max"
        ],
        8
    )
)

print(
    "Mean future damage change:",
    round(
        primary_summary[
            "mean_damage_change"
        ],
        8
    )
)

print(
    "Positive future change fraction:",
    round(
        primary_summary[
            "positive_change_fraction"
        ],
        4
    )
)

print(
    "Negative future change fraction:",
    round(
        primary_summary[
            "negative_change_fraction"
        ],
        4
    )
)

print(
    "Zero future change fraction:",
    round(
        primary_summary[
            "zero_change_fraction"
        ],
        4
    )
)

print("")
print(
    "AUDIT CHECKS:"
)

for name, result in checks.items():

    print(
        "  ",
        name,
        ":",
        "PASS"
        if result
        else "FAIL"
    )

print("")

print(
    "Prediction targets:"
)

print(
    TARGET_FILE
)

print("")

print(
    "Sequence summary:"
)

print(
    SEQUENCE_OUTPUT
)

print("")

print(
    "Split summary:"
)

print(
    SPLIT_OUTPUT
)

print("")

print(
    "JSON report:"
)

print(
    JSON_OUTPUT
)

print("")

if final_status:

    print(
        "FINAL STATUS: PASS"
    )

else:

    print(
        "FINAL STATUS: REVIEW"
    )

print("")