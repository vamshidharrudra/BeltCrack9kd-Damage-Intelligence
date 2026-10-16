import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 1.3B
# PREDICTION TARGET INTEGRITY VERIFICATION
#
# Purpose:
#   Independently verify the Stage 1.3 prediction targets.
#
# Primary target:
#   crack_area_ratio at t+1
#
# Secondary targets:
#   crack_area_ratio at t+3
#   crack_area_ratio at t+5
#
# No model training is performed.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

TARGET_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3_prediction_targets.csv"
)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_0_damage_features.csv"
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3b_target_integrity.json"
)

OUTPUT_SEQUENCE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3b_sequence_integrity.csv"
)

OUTPUT_SPLIT = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3b_split_integrity.csv"
)


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_SEQUENCES = 29
EXPECTED_COMPONENTS = 20

EXPECTED_HORIZONS = [
    1,
    3,
    5
]


# ============================================================
# START
# ============================================================

print("")
print("=" * 70)
print("STAGE 1.3B - TARGET INTEGRITY VERIFICATION")
print("=" * 70)
print("")


# ============================================================
# CHECK INPUT FILES
# ============================================================

input_files = [
    (
        TARGET_FILE,
        "Stage 1.3 target file"
    ),
    (
        FEATURE_FILE,
        "Stage 1.0 feature file"
    ),
    (
        MANIFEST_FILE,
        "Canonical manifest"
    )
]


for path, name in input_files:

    if not os.path.exists(path):

        print(
            "ERROR:",
            name,
            "not found:"
        )

        print(path)

        raise SystemExit(1)


# ============================================================
# LOAD TARGET DATA
# ============================================================

print(
    "Loading Stage 1.3 target table..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


# ============================================================
# LOAD STAGE 1.0 FEATURES
# ============================================================

print("")
print(
    "Loading Stage 1.0 feature table..."
)

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature records:",
    len(features)
)


# ============================================================
# LOAD CANONICAL MANIFEST
# ============================================================

print("")
print(
    "Loading canonical manifest..."
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

print(
    "Manifest records:",
    len(manifest)
)


# ============================================================
# REQUIRED TARGET COLUMNS
# ============================================================

required_target_columns = [
    "sequence",
    "split",
    "component",
    "horizon",
    "current_frame",
    "future_frame",
    "current_record_id",
    "future_record_id",
    "current_crack_area_ratio",
    "target_crack_area_ratio"
]


missing_target_columns = [
    column
    for column in required_target_columns
    if column not in targets.columns
]


if missing_target_columns:

    print("")
    print(
        "ERROR: Missing required target columns:"
    )

    for column in missing_target_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# BASIC HORIZON CHECK
# ============================================================

print("")
print(
    "Checking target horizons..."
)

observed_horizons = sorted(
    targets[
        "horizon"
    ]
    .dropna()
    .unique()
    .tolist()
)

print(
    "Observed horizons:",
    observed_horizons
)

print(
    "Expected horizons:",
    EXPECTED_HORIZONS
)

horizon_check = (
    observed_horizons == EXPECTED_HORIZONS
)

print(
    "Horizon check:",
    "PASS"
    if horizon_check
    else "FAIL"
)


# ============================================================
# ACTUAL FRAME GAP
# ============================================================

print("")
print(
    "Checking temporal frame gaps..."
)

targets[
    "actual_frame_gap"
] = (
    targets[
        "future_frame"
    ]
    -
    targets[
        "current_frame"
    ]
)

targets[
    "gap_valid"
] = (
    targets[
        "actual_frame_gap"
    ]
    ==
    targets[
        "horizon"
    ]
)

invalid_gap_count = int(
    (
        ~targets[
            "gap_valid"
        ]
    ).sum()
)

print(
    "Invalid frame gaps:",
    invalid_gap_count
)


# ============================================================
# TEMPORAL DIRECTION
# ============================================================

print("")
print(
    "Checking temporal direction..."
)

invalid_direction_count = int(
    (
        targets[
            "future_frame"
        ]
        <=
        targets[
            "current_frame"
        ]
    ).sum()
)

print(
    "Invalid temporal direction:",
    invalid_direction_count
)


# ============================================================
# SELF TARGET CHECK
# ============================================================

print("")
print(
    "Checking current/future record identity..."
)

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

print(
    "Self-target pairs:",
    self_target_count
)


# ============================================================
# TARGET VALIDITY
# ============================================================

print("")
print(
    "Checking target values..."
)

target_values = pd.to_numeric(
    targets[
        "target_crack_area_ratio"
    ],
    errors="coerce"
)

current_values = pd.to_numeric(
    targets[
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


invalid_target_count = int(
    (
        ~np.isfinite(
            target_values
        )
    ).sum()
)


negative_target_count = int(
    (
        target_values < 0
    )
    .fillna(False)
    .sum()
)


target_above_one_count = int(
    (
        target_values > 1
    )
    .fillna(False)
    .sum()
)


invalid_current_count = int(
    (
        ~np.isfinite(
            current_values
        )
    ).sum()
)


negative_current_count = int(
    (
        current_values < 0
    )
    .fillna(False)
    .sum()
)


current_above_one_count = int(
    (
        current_values > 1
    )
    .fillna(False)
    .sum()
)


print(
    "Invalid targets:",
    invalid_target_count
)

print(
    "Negative targets:",
    negative_target_count
)

print(
    "Targets > 1:",
    target_above_one_count
)

print(
    "Invalid current values:",
    invalid_current_count
)

print(
    "Negative current values:",
    negative_current_count
)

print(
    "Current values > 1:",
    current_above_one_count
)


# ============================================================
# DUPLICATE TARGET PAIRS
# ============================================================

print("")
print(
    "Checking duplicate target pairs..."
)

pair_key_columns = [
    "sequence",
    "current_frame",
    "future_frame",
    "horizon",
    "current_record_id",
    "future_record_id"
]


duplicate_pair_mask = targets.duplicated(
    subset=pair_key_columns,
    keep=False
)


duplicate_pair_count = int(
    duplicate_pair_mask.sum()
)


duplicate_pair_groups = int(
    targets[
        duplicate_pair_mask
    ]
    .groupby(
        pair_key_columns,
        dropna=False
    )
    .ngroups
)


print(
    "Duplicate target rows:",
    duplicate_pair_count
)

print(
    "Duplicate target groups:",
    duplicate_pair_groups
)


# ============================================================
# SEQUENCE -> SPLIT INTEGRITY
#
# Important:
# A sequence may have many target rows because of different
# horizons. Therefore we first remove duplicate
# sequence/split/component assignments.
# ============================================================

print("")
print(
    "Checking sequence-to-split assignments..."
)

sequence_assignments = (
    targets[
        [
            "sequence",
            "split",
            "component"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        by=[
            "sequence",
            "split"
        ],
        kind="stable"
    )
    .reset_index(
        drop=True
    )
)


sequence_split_counts = (
    sequence_assignments
    .groupby(
        "sequence"
    )[
        "split"
    ]
    .nunique()
)


sequence_conflict_values = (
    sequence_split_counts[
        sequence_split_counts > 1
    ]
)


print(
    "Unique sequences:",
    len(
        sequence_split_counts
    )
)

print(
    "Sequence split conflicts:",
    len(
        sequence_conflict_values
    )
)


# ============================================================
# COMPONENT -> SPLIT INTEGRITY
# ============================================================

print("")
print(
    "Checking component-to-split assignments..."
)

component_assignments = (
    targets[
        [
            "component",
            "split",
            "sequence"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        by=[
            "component",
            "split"
        ],
        kind="stable"
    )
    .reset_index(
        drop=True
    )
)


component_split_counts = (
    component_assignments
    .groupby(
        "component"
    )[
        "split"
    ]
    .nunique()
)


component_conflict_values = (
    component_split_counts[
        component_split_counts > 1
    ]
)


print(
    "Unique components:",
    len(
        component_split_counts
    )
)

print(
    "Component split conflicts:",
    len(
        component_conflict_values
    )
)


# ============================================================
# SAVE SEQUENCE INTEGRITY
# ============================================================

sequence_output = sequence_assignments.copy()

sequence_output[
    "split_count"
] = (
    sequence_output[
        "sequence"
    ]
    .map(
        sequence_split_counts
    )
)

sequence_output[
    "conflict"
] = (
    sequence_output[
        "split_count"
    ]
    > 1
)

sequence_output.to_csv(
    OUTPUT_SEQUENCE,
    index=False
)


# ============================================================
# BUILD SPLIT SUMMARY
# ============================================================

print("")
print(
    "Building split summary..."
)

split_records = []


unique_splits = sorted(
    targets[
        "split"
    ]
    .dropna()
    .unique()
)


for split in unique_splits:

    split_group = targets[
        targets[
            "split"
        ]
        ==
        split
    ].copy()


    target = pd.to_numeric(
        split_group[
            "target_crack_area_ratio"
        ],
        errors="coerce"
    )


    split_records.append({

        "split":
            split,

        "records":
            int(
                len(
                    split_group
                )
            ),

        "sequences":
            int(
                split_group[
                    "sequence"
                ].nunique()
            ),

        "components":
            int(
                split_group[
                    "component"
                ].nunique()
            ),

        "horizon_1_records":
            int(
                (
                    split_group[
                        "horizon"
                    ]
                    ==
                    1
                ).sum()
            ),

        "horizon_3_records":
            int(
                (
                    split_group[
                        "horizon"
                    ]
                    ==
                    3
                ).sum()
            ),

        "horizon_5_records":
            int(
                (
                    split_group[
                        "horizon"
                    ]
                    ==
                    5
                ).sum()
            ),

        "target_mean":
            float(
                target.mean()
            ),

        "target_median":
            float(
                target.median()
            ),

        "target_min":
            float(
                target.min()
            ),

        "target_max":
            float(
                target.max()
            )
    })


split_df = pd.DataFrame(
    split_records
)


split_df.to_csv(
    OUTPUT_SPLIT,
    index=False
)


# ============================================================
# PRIMARY TARGET ANALYSIS
# ============================================================

primary = targets[
    targets[
        "horizon"
    ]
    ==
    1
].copy()


primary_target = pd.to_numeric(
    primary[
        "target_crack_area_ratio"
    ],
    errors="coerce"
)


primary_change = pd.to_numeric(
    primary[
        "target_damage_change"
    ],
    errors="coerce"
)


primary_current = pd.to_numeric(
    primary[
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


primary_summary = {

    "records":
        int(
            len(
                primary
            )
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

    "mean":
        float(
            primary_target.mean()
        ),

    "median":
        float(
            primary_target.median()
        ),

    "std":
        float(
            primary_target.std()
        ),

    "min":
        float(
            primary_target.min()
        ),

    "max":
        float(
            primary_target.max()
        ),

    "current_mean":
        float(
            primary_current.mean()
        ),

    "future_mean":
        float(
            primary_target.mean()
        ),

    "mean_change":
        float(
            primary_change.mean()
        ),

    "median_change":
        float(
            primary_change.median()
        ),

    "positive_change_fraction":
        float(
            (
                primary_change > 0
            ).mean()
        ),

    "negative_change_fraction":
        float(
            (
                primary_change < 0
            ).mean()
        ),

    "zero_change_fraction":
        float(
            (
                primary_change == 0
            ).mean()
        )
}


# ============================================================
# DATASET COVERAGE
# ============================================================

observed_sequence_count = int(
    targets[
        "sequence"
    ]
    .nunique()
)


observed_component_count = int(
    targets[
        "component"
    ]
    .nunique()
)


expected_sequence_count_check = (
    observed_sequence_count
    ==
    EXPECTED_SEQUENCES
)


expected_component_count_check = (
    observed_component_count
    ==
    EXPECTED_COMPONENTS
)


# ============================================================
# HORIZON RECORD COUNTS
# ============================================================

horizon_counts = (
    targets[
        "horizon"
    ]
    .value_counts()
    .sort_index()
)


horizon_count_dict = {
    str(int(horizon)): int(count)
    for horizon, count
    in horizon_counts.items()
}


# ============================================================
# AUDIT CHECKS
# ============================================================

checks = {

    "target_file_exists":
        os.path.exists(
            TARGET_FILE
        ),

    "feature_file_exists":
        os.path.exists(
            FEATURE_FILE
        ),

    "manifest_file_exists":
        os.path.exists(
            MANIFEST_FILE
        ),

    "expected_horizons":
        horizon_check,

    "invalid_frame_gaps":
        invalid_gap_count == 0,

    "invalid_temporal_direction":
        invalid_direction_count == 0,

    "self_target_pairs":
        self_target_count == 0,

    "invalid_targets":
        invalid_target_count == 0,

    "negative_targets":
        negative_target_count == 0,

    "target_above_one":
        target_above_one_count == 0,

    "invalid_current_values":
        invalid_current_count == 0,

    "negative_current":
        negative_current_count == 0,

    "current_above_one":
        current_above_one_count == 0,

    "duplicate_target_rows":
        duplicate_pair_count == 0,

    "sequence_split_conflicts":
        len(
            sequence_conflict_values
        ) == 0,

    "component_split_conflicts":
        len(
            component_conflict_values
        ) == 0,

    "expected_sequence_count":
        expected_sequence_count_check,

    "expected_component_count":
        expected_component_count_check,

    "primary_target_nonempty":
        len(
            primary
        ) > 0,

    "primary_has_all_sequences":
        primary[
            "sequence"
        ].nunique()
        ==
        EXPECTED_SEQUENCES
}


# ============================================================
# FINAL STATUS
# ============================================================

final_status = all(
    checks.values()
)


# ============================================================
# REPORT
# ============================================================

report = {

    "stage":
        "1.3B",

    "title":
        "Prediction Target Integrity Verification",

    "status":
        "PASS"
        if final_status
        else "REVIEW",

    "target_definition": {

        "primary":
            "crack_area_ratio at t+1",

        "secondary": [
            "crack_area_ratio at t+3",
            "crack_area_ratio at t+5"
        ]
    },

    "dataset": {

        "target_records":
            int(
                len(
                    targets
                )
            ),

        "primary_records":
            int(
                len(
                    primary
                )
            ),

        "sequences":
            observed_sequence_count,

        "components":
            observed_component_count,

        "horizon_counts":
            horizon_count_dict
    },

    "primary_target":
        primary_summary,

    "audit_counts": {

        "invalid_frame_gaps":
            invalid_gap_count,

        "invalid_direction":
            invalid_direction_count,

        "self_pairs":
            self_target_count,

        "invalid_targets":
            invalid_target_count,

        "negative_targets":
            negative_target_count,

        "target_above_one":
            target_above_one_count,

        "current_invalid":
            invalid_current_count,

        "current_negative":
            negative_current_count,

        "current_above_one":
            current_above_one_count,

        "duplicate_target_rows":
            duplicate_pair_count,

        "duplicate_target_groups":
            duplicate_pair_groups,

        "sequence_split_conflicts":
            len(
                sequence_conflict_values
            ),

        "component_split_conflicts":
            len(
                component_conflict_values
            )
    },

    "checks":
        checks,

    "outputs": {

        "sequence_integrity":
            OUTPUT_SEQUENCE,

        "split_integrity":
            OUTPUT_SPLIT,

        "report":
            OUTPUT_JSON
    },

    "next_stage":
        "Feature/target leakage audit before ML training."
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print("")
print("=" * 70)
print("STAGE 1.3B SUMMARY")
print("=" * 70)

print(
    "Target records:",
    len(
        targets
    )
)

print(
    "Primary t+1 records:",
    len(
        primary
    )
)

print(
    "Sequences:",
    observed_sequence_count
)

print(
    "Components:",
    observed_component_count
)

print("")

print(
    "Horizon counts:"
)

for horizon in sorted(
    horizon_count_dict.keys(),
    key=lambda x: int(x)
):

    print(
        "  t+" + horizon + ":",
        horizon_count_dict[
            horizon
        ]
    )

print("")

print(
    "Sequence split conflicts:",
    len(
        sequence_conflict_values
    )
)

print(
    "Component split conflicts:",
    len(
        component_conflict_values
    )
)

print(
    "Duplicate target rows:",
    duplicate_pair_count
)

print(
    "Duplicate target groups:",
    duplicate_pair_groups
)

print(
    "Invalid frame gaps:",
    invalid_gap_count
)

print(
    "Invalid temporal direction:",
    invalid_direction_count
)

print(
    "Self-target pairs:",
    self_target_count
)

print("")

print(
    "PRIMARY t+1 TARGET"
)

print(
    "Records:",
    primary_summary[
        "records"
    ]
)

print(
    "Mean:",
    round(
        primary_summary[
            "mean"
        ],
        8
    )
)

print(
    "Median:",
    round(
        primary_summary[
            "median"
        ],
        8
    )
)

print(
    "Min:",
    round(
        primary_summary[
            "min"
        ],
        8
    )
)

print(
    "Max:",
    round(
        primary_summary[
            "max"
        ],
        8
    )
)

print(
    "Mean change:",
    round(
        primary_summary[
            "mean_change"
        ],
        8
    )
)

print(
    "Positive change fraction:",
    round(
        primary_summary[
            "positive_change_fraction"
        ],
        4
    )
)

print(
    "Negative change fraction:",
    round(
        primary_summary[
            "negative_change_fraction"
        ],
        4
    )
)

print(
    "Zero change fraction:",
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
    "Sequence integrity file:"
)

print(
    OUTPUT_SEQUENCE
)

print("")

print(
    "Split integrity file:"
)

print(
    OUTPUT_SPLIT
)

print("")

print(
    "JSON report:"
)

print(
    OUTPUT_JSON
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