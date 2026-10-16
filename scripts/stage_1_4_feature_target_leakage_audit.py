import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 1.4
# FEATURE / TARGET LEAKAGE AUDIT
#
# Prediction:
#
#   Frame t physical features
#              |
#              v
#        ML regression
#              |
#              v
#   crack_area_ratio at t+1
#
# This stage does NOT train a model.
# ============================================================


# ============================================================
# PROJECT PATHS
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

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_4_feature_target_leakage_audit.json"
)

OUTPUT_FEATURE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_4_feature_leakage_analysis.csv"
)

OUTPUT_CORRELATION = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_4_target_feature_correlation.csv"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_SEQUENCES = 29
EXPECTED_COMPONENTS = 20


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

TARGET_COLUMN = "target_crack_area_ratio"

CURRENT_DAMAGE_COLUMN = "current_crack_area_ratio"

CURRENT_RECORD_COLUMN = "current_record_id"

FUTURE_RECORD_COLUMN = "future_record_id"

CURRENT_FRAME_COLUMN = "current_frame"

FUTURE_FRAME_COLUMN = "future_frame"

SEQUENCE_COLUMN = "sequence"

COMPONENT_COLUMN = "component"

SPLIT_COLUMN = "split"

HORIZON_COLUMN = "horizon"


# ============================================================
# START
# ============================================================

print("")
print("=" * 70)
print("STAGE 1.4 - FEATURE / TARGET LEAKAGE AUDIT")
print("=" * 70)
print("")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    (
        FEATURE_FILE,
        "Stage 1.0 feature file"
    ),
    (
        TARGET_FILE,
        "Stage 1.3 target file"
    ),
    (
        MANIFEST_FILE,
        "Canonical manifest"
    )
]


for path, description in required_files:

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
    TARGET_COLUMN,
    CURRENT_DAMAGE_COLUMN,
    CURRENT_RECORD_COLUMN,
    FUTURE_RECORD_COLUMN,
    CURRENT_FRAME_COLUMN,
    FUTURE_FRAME_COLUMN,
    SEQUENCE_COLUMN,
    COMPONENT_COLUMN,
    SPLIT_COLUMN,
    HORIZON_COLUMN
]


missing_target_columns = [
    column
    for column in required_target_columns
    if column not in targets.columns
]


if missing_target_columns:

    print("")
    print(
        "ERROR: Missing target columns:"
    )

    for column in missing_target_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# REQUIRED FEATURE COLUMNS
# ============================================================

required_feature_columns = [
    "record_id",
    "sequence",
    "component",
    "split",
    "crack_area_ratio"
]


missing_feature_columns = [
    column
    for column in required_feature_columns
    if column not in features.columns
]


if missing_feature_columns:

    print("")
    print(
        "ERROR: Missing Stage 1.0 feature columns:"
    )

    for column in missing_feature_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# PRIMARY TARGET
# ============================================================

primary = targets[
    targets[
        HORIZON_COLUMN
    ]
    ==
    1
].copy()


print("")
print(
    "Primary t+1 target records:",
    len(primary)
)


# ============================================================
# CHECK HORIZONS
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 1 - TARGET HORIZONS"
)

print(
    "=" * 70
)


observed_horizons = sorted(
    targets[
        HORIZON_COLUMN
    ]
    .dropna()
    .unique()
    .tolist()
)


expected_horizons = [
    1,
    3,
    5
]


horizon_check = (
    observed_horizons
    ==
    expected_horizons
)


print(
    "Observed horizons:",
    observed_horizons
)

print(
    "Expected horizons:",
    expected_horizons
)

print(
    "Horizon check:",
    "PASS"
    if horizon_check
    else "FAIL"
)


# ============================================================
# CHECK 2 - DIRECT TARGET LEAKAGE
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 2 - DIRECT TARGET LEAKAGE"
)

print(
    "=" * 70
)


target_keyword_columns = []


target_keywords = [
    "target",
    "future",
    "next",
    "t_plus"
]


for column in features.columns:

    lower_name = str(
        column
    ).lower()

    for keyword in target_keywords:

        if keyword in lower_name:

            target_keyword_columns.append(
                column
            )

            break


target_keyword_columns = sorted(
    set(
        target_keyword_columns
    )
)


direct_target_column_present = (
    TARGET_COLUMN
    in
    features.columns
)


print(
    "Feature columns containing target/future keywords:"
)

if target_keyword_columns:

    for column in target_keyword_columns:

        print(
            "  -",
            column
        )

else:

    print(
        "  None"
    )


print(
    "Exact target column present:",
    direct_target_column_present
)


# ============================================================
# CHECK 3 - FUTURE INFORMATION
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 3 - FUTURE INFORMATION"
)

print(
    "=" * 70
)


future_information_columns = []


future_keywords = [
    "future",
    "next",
    "t+1",
    "t+3",
    "t+5"
]


for column in features.columns:

    lower_name = str(
        column
    ).lower()

    for keyword in future_keywords:

        if keyword in lower_name:

            future_information_columns.append(
                column
            )

            break


future_information_columns = sorted(
    set(
        future_information_columns
    )
)


print(
    "Potential future-information columns:"
)

if future_information_columns:

    for column in future_information_columns:

        print(
            "  -",
            column
        )

else:

    print(
        "  None"
    )


# ============================================================
# CHECK 4 - FUTURE METADATA
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 4 - FUTURE METADATA"
)

print(
    "=" * 70
)


future_metadata_columns = []


future_metadata_keywords = [
    "future_record",
    "future_frame",
    "future_image",
    "future_json",
    "next_frame",
    "next_record"
]


for column in features.columns:

    lower_name = str(
        column
    ).lower()

    for keyword in future_metadata_keywords:

        if keyword in lower_name:

            future_metadata_columns.append(
                column
            )

            break


future_metadata_columns = sorted(
    set(
        future_metadata_columns
    )
)


print(
    "Potential future metadata columns:"
)

if future_metadata_columns:

    for column in future_metadata_columns:

        print(
            "  -",
            column
        )

else:

    print(
        "  None"
    )


# ============================================================
# CHECK 5 - PROPOSED MODEL FEATURES
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 5 - PROPOSED MODEL FEATURES"
)

print(
    "=" * 70
)


identifier_columns = [
    "record_id",
    "split",
    "component",
    "sequence",
    "original_image_filename",
    "original_json_filename",
    "canonical_image_filename",
    "canonical_json_filename",
    "source_image",
    "source_json"
]


future_columns = [
    "future_record_id",
    "future_frame",
    "future_crack_area_ratio",
    "future_image",
    "future_json",
    "target_crack_area_ratio",
    "target_damage_change",
    "target_damage_relative_change"
]


non_model_columns = (
    identifier_columns
    +
    future_columns
)


candidate_features = [
    column
    for column in features.columns
    if column not in non_model_columns
]


print(
    "Candidate physical features:"
)

for column in candidate_features:

    print(
        "  -",
        column
    )


print(
    "Candidate feature count:",
    len(
        candidate_features
    )
)


# ============================================================
# CHECK 6 - MERGE CURRENT FRAME
#
# IMPORTANT:
# Stage 1.3 and Stage 1.0 both contain several columns with
# the same names.
#
# Therefore we explicitly rename Stage 1.0 columns before
# merging rather than relying on pandas suffixes.
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 6 - CURRENT FRAME FEATURE ALIGNMENT"
)

print(
    "=" * 70
)


feature_current = features.copy()


feature_current = feature_current.rename(
    columns={
        "sequence":
            "feature_sequence",

        "component":
            "feature_component",

        "split":
            "feature_split",

        "crack_area_ratio":
            "feature_crack_area_ratio"
    }
)


merged = primary.merge(
    feature_current,
    left_on=CURRENT_RECORD_COLUMN,
    right_on="record_id",
    how="left"
)


print(
    "Merged primary records:",
    len(
        merged
    )
)


missing_current_features = int(
    merged[
        "record_id"
    ].isna().sum()
)


print(
    "Missing current feature rows:",
    missing_current_features
)


# ============================================================
# CHECK CURRENT RECORD IDENTITY
# ============================================================

print("")
print(
    "Checking current sequence identity..."
)


current_sequence_mismatch = int(
    (
        merged[
            SEQUENCE_COLUMN
        ]
        !=
        merged[
            "feature_sequence"
        ]
    ).sum()
)


current_component_mismatch = int(
    (
        merged[
            COMPONENT_COLUMN
        ]
        !=
        merged[
            "feature_component"
        ]
    ).sum()
)


current_split_mismatch = int(
    (
        merged[
            SPLIT_COLUMN
        ]
        !=
        merged[
            "feature_split"
        ]
    ).sum()
)


print(
    "Current sequence mismatches:",
    current_sequence_mismatch
)

print(
    "Current component mismatches:",
    current_component_mismatch
)

print(
    "Current split mismatches:",
    current_split_mismatch
)


# ============================================================
# CHECK CURRENT DAMAGE CONSISTENCY
# ============================================================

print("")
print(
    "Checking current damage consistency..."
)


feature_current_damage = pd.to_numeric(
    merged[
        "feature_crack_area_ratio"
    ],
    errors="coerce"
)


target_current_damage = pd.to_numeric(
    merged[
        CURRENT_DAMAGE_COLUMN
    ],
    errors="coerce"
)


current_difference = (
    feature_current_damage
    -
    target_current_damage
).abs()


current_damage_mismatch_count = int(
    (
        current_difference
        >
        1e-10
    ).sum()
)


max_current_damage_difference = float(
    current_difference.max()
)


print(
    "Current damage mismatches:",
    current_damage_mismatch_count
)

print(
    "Maximum current damage difference:",
    max_current_damage_difference
)


# ============================================================
# CHECK CURRENT FRAME
# ============================================================

print("")
print(
    "Checking current frame alignment..."
)


# The target table uses current_frame.
# Stage 1.0 uses the original image filename.
#
# The exact numeric frame is not stored as a separate Stage 1.0
# feature column, so we do not invent one here.
#
# Record ID identity is the authoritative alignment check.
#
current_record_unique = (
    primary[
        CURRENT_RECORD_COLUMN
    ].nunique()
)


print(
    "Unique current record IDs:",
    current_record_unique
)

print(
    "Primary target rows:",
    len(
        primary
    )
)


current_record_duplicate_count = (
    len(
        primary
    )
    -
    current_record_unique
)


print(
    "Repeated current record IDs:",
    current_record_duplicate_count
)


# ============================================================
# CHECK 7 - DIRECT NUMERICAL TARGET MATCH
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 7 - DIRECT NUMERICAL TARGET MATCH"
)

print(
    "=" * 70
)


target_values = pd.to_numeric(
    merged[
        TARGET_COLUMN
    ],
    errors="coerce"
)


direct_numerical_matches = []


for column in candidate_features:

    if column not in merged.columns:

        continue


    feature_values = pd.to_numeric(
        merged[
            column
        ],
        errors="coerce"
    )


    valid_mask = (
        feature_values.notna()
        &
        target_values.notna()
    )


    valid_count = int(
        valid_mask.sum()
    )


    if valid_count == 0:

        continue


    difference = (
        feature_values[
            valid_mask
        ]
        -
        target_values[
            valid_mask
        ]
    ).abs()


    match_count = int(
        (
            difference
            <=
            1e-12
        ).sum()
    )


    match_fraction = (
        match_count
        /
        valid_count
    )


    if match_fraction >= 0.999999:

        direct_numerical_matches.append({

            "feature":
                column,

            "valid_records":
                valid_count,

            "matching_records":
                match_count,

            "match_fraction":
                float(
                    match_fraction
                )
        })


print(
    "Features numerically identical to future target:"
)

if direct_numerical_matches:

    for item in direct_numerical_matches:

        print(
            "  -",
            item[
                "feature"
            ],
            "match fraction:",
            round(
                item[
                    "match_fraction"
                ],
                6
            )
        )

else:

    print(
        "  None"
    )


# ============================================================
# CHECK 8 - FEATURE/TARGET CORRELATION
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 8 - FEATURE/TARGET CORRELATION"
)

print(
    "=" * 70
)


correlation_records = []


for column in candidate_features:

    if column not in merged.columns:

        continue


    feature_values = pd.to_numeric(
        merged[
            column
        ],
        errors="coerce"
    )


    target_values_local = pd.to_numeric(
        merged[
            TARGET_COLUMN
        ],
        errors="coerce"
    )


    valid_mask = (
        feature_values.notna()
        &
        target_values_local.notna()
    )


    valid_count = int(
        valid_mask.sum()
    )


    if valid_count < 3:

        correlation = np.nan

    else:

        correlation = (
            feature_values[
                valid_mask
            ]
            .corr(
                target_values_local[
                    valid_mask
                ]
            )
        )


    correlation_records.append({

        "feature":
            column,

        "valid_records":
            valid_count,

        "pearson_correlation":
            float(
                correlation
            )
            if pd.notna(
                correlation
            )
            else np.nan
    })


correlation_df = pd.DataFrame(
    correlation_records
)


if len(
    correlation_df
) > 0:

    correlation_df[
        "absolute_correlation"
    ] = (
        correlation_df[
            "pearson_correlation"
        ]
        .abs()
    )

    correlation_df = (
        correlation_df
        .sort_values(
            by="absolute_correlation",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


correlation_df.to_csv(
    OUTPUT_CORRELATION,
    index=False
)


print(
    "Top feature correlations:"
)

for _, row in (
    correlation_df
    .head(10)
    .iterrows()
):

    print(
        "  ",
        row[
            "feature"
        ],
        ":",
        round(
            row[
                "pearson_correlation"
            ],
            6
        )
    )


# ============================================================
# CHECK 9 - FUTURE RECORD RESOLUTION
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 9 - FUTURE RECORD RESOLUTION"
)

print(
    "=" * 70
)


future_lookup = features[
    [
        "record_id",
        "sequence",
        "component",
        "split",
        "crack_area_ratio"
    ]
].copy()


future_lookup = future_lookup.rename(
    columns={
        "record_id":
            "future_record_id_lookup",

        "sequence":
            "future_sequence_lookup",

        "component":
            "future_component_lookup",

        "split":
            "future_split_lookup",

        "crack_area_ratio":
            "future_crack_area_ratio_lookup"
    }
)


future_join = primary.merge(
    future_lookup,
    left_on=FUTURE_RECORD_COLUMN,
    right_on="future_record_id_lookup",
    how="left"
)


future_missing_count = int(
    future_join[
        "future_record_id_lookup"
    ].isna().sum()
)


print(
    "Future records missing:",
    future_missing_count
)


# ============================================================
# FUTURE TARGET CONSISTENCY
# ============================================================

future_target_values = pd.to_numeric(
    future_join[
        TARGET_COLUMN
    ],
    errors="coerce"
)


future_actual_values = pd.to_numeric(
    future_join[
        "future_crack_area_ratio_lookup"
    ],
    errors="coerce"
)


future_difference = (
    future_target_values
    -
    future_actual_values
).abs()


future_target_mismatch_count = int(
    (
        future_difference
        >
        1e-10
    ).sum()
)


max_future_difference = float(
    future_difference.max()
)


print(
    "Future target mismatches:",
    future_target_mismatch_count
)

print(
    "Maximum future target difference:",
    max_future_difference
)


# ============================================================
# FUTURE SEQUENCE / COMPONENT / SPLIT CHECK
# ============================================================

future_sequence_mismatch_count = int(
    (
        future_join[
            "future_sequence_lookup"
        ]
        !=
        future_join[
            SEQUENCE_COLUMN
        ]
    ).sum()
)


future_component_mismatch_count = int(
    (
        future_join[
            "future_component_lookup"
        ]
        !=
        future_join[
            COMPONENT_COLUMN
        ]
    ).sum()
)


future_split_mismatch_count = int(
    (
        future_join[
            "future_split_lookup"
        ]
        !=
        future_join[
            SPLIT_COLUMN
        ]
    ).sum()
)


print(
    "Future sequence mismatches:",
    future_sequence_mismatch_count
)

print(
    "Future component mismatches:",
    future_component_mismatch_count
)

print(
    "Future split mismatches:",
    future_split_mismatch_count
)


# ============================================================
# CHECK 10 - SEQUENCE SPLIT INTEGRITY
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 10 - SEQUENCE / COMPONENT SPLIT INTEGRITY"
)

print(
    "=" * 70
)


sequence_assignments = (
    primary[
        [
            SEQUENCE_COLUMN,
            SPLIT_COLUMN
        ]
    ]
    .drop_duplicates()
)


sequence_split_counts = (
    sequence_assignments
    .groupby(
        SEQUENCE_COLUMN
    )[
        SPLIT_COLUMN
    ]
    .nunique()
)


sequence_split_conflicts = int(
    (
        sequence_split_counts
        >
        1
    ).sum()
)


component_assignments = (
    primary[
        [
            COMPONENT_COLUMN,
            SPLIT_COLUMN
        ]
    ]
    .drop_duplicates()
)


component_split_counts = (
    component_assignments
    .groupby(
        COMPONENT_COLUMN
    )[
        SPLIT_COLUMN
    ]
    .nunique()
)


component_split_conflicts = int(
    (
        component_split_counts
        >
        1
    ).sum()
)


print(
    "Sequence split conflicts:",
    sequence_split_conflicts
)

print(
    "Component split conflicts:",
    component_split_conflicts
)


# ============================================================
# CHECK 11 - SOURCE IMAGE SPLIT ISOLATION
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 11 - SOURCE IMAGE SPLIT ISOLATION"
)

print(
    "=" * 70
)


if (
    "source_image"
    in
    manifest.columns
    and
    "split"
    in
    manifest.columns
):

    source_assignments = (
        manifest[
            [
                "source_image",
                "split"
            ]
        ]
        .drop_duplicates()
    )


    source_split_counts = (
        source_assignments
        .groupby(
            "source_image"
        )[
            "split"
        ]
        .nunique()
    )


    source_split_conflicts = int(
        (
            source_split_counts
            >
            1
        ).sum()
    )

else:

    source_split_conflicts = -1


print(
    "Source image split conflicts:",
    source_split_conflicts
)


# ============================================================
# CHECK 12 - DUPLICATE IMAGE HASH SPLIT ISOLATION
#
# Stage 0.9D.2 already verified this.
# Here we verify that the current canonical manifest does not
# accidentally place the same source path into multiple splits.
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 12 - SOURCE PATH DUPLICATION"
)

print(
    "=" * 70
)


if (
    "source_image"
    in
    manifest.columns
):

    duplicate_source_paths = int(
        manifest[
            "source_image"
        ]
        .duplicated()
        .sum()
    )

else:

    duplicate_source_paths = -1


print(
    "Duplicate source paths:",
    duplicate_source_paths
)


# ============================================================
# CHECK 13 - MODEL FEATURE SET
# ============================================================

print("")
print(
    "=" * 70
)

print(
    "CHECK 13 - FINAL MODEL FEATURE SET"
)

print(
    "=" * 70
)


model_features = []


for column in candidate_features:

    if column in future_columns:

        continue

    if column in future_information_columns:

        continue

    if column in future_metadata_columns:

        continue

    model_features.append(
        column
    )


print(
    "Final model feature count:",
    len(
        model_features
    )
)

print("")

for column in model_features:

    print(
        "  -",
        column
    )


# ============================================================
# IMPORTANT CONCEPTUAL CHECK
#
# current_crack_area_ratio is valid.
#
# It describes the physical state at frame t.
#
# The target describes frame t+1.
#
# Therefore current damage is an input feature, not leakage.
# ============================================================


# ============================================================
# FINAL AUDIT CHECKS
# ============================================================

checks = {

    "feature_file_exists":
        os.path.exists(
            FEATURE_FILE
        ),

    "target_file_exists":
        os.path.exists(
            TARGET_FILE
        ),

    "manifest_file_exists":
        os.path.exists(
            MANIFEST_FILE
        ),

    "expected_horizons":
        horizon_check,

    "primary_target_nonempty":
        len(
            primary
        ) > 0,

    "expected_sequence_count":
        primary[
            SEQUENCE_COLUMN
        ].nunique()
        ==
        EXPECTED_SEQUENCES,

    "expected_component_count":
        primary[
            COMPONENT_COLUMN
        ].nunique()
        ==
        EXPECTED_COMPONENTS,

    "direct_target_column_absent":
        not direct_target_column_present,

    "future_information_absent":
        len(
            future_information_columns
        ) == 0,

    "future_metadata_absent":
        len(
            future_metadata_columns
        ) == 0,

    "current_feature_rows_complete":
        missing_current_features == 0,

    "current_sequence_alignment":
        current_sequence_mismatch == 0,

    "current_component_alignment":
        current_component_mismatch == 0,

    "current_split_alignment":
        current_split_mismatch == 0,

    "current_damage_consistent":
        current_damage_mismatch_count == 0,

    "direct_numerical_target_matches_absent":
        len(
            direct_numerical_matches
        ) == 0,

    "future_records_resolved":
        future_missing_count == 0,

    "future_target_consistent":
        future_target_mismatch_count == 0,

    "future_sequence_consistent":
        future_sequence_mismatch_count == 0,

    "future_component_consistent":
        future_component_mismatch_count == 0,

    "future_split_consistent":
        future_split_mismatch_count == 0,

    "sequence_split_conflicts":
        sequence_split_conflicts == 0,

    "component_split_conflicts":
        component_split_conflicts == 0,

    "source_image_split_conflicts":
        source_split_conflicts == 0,

    "duplicate_source_paths":
        duplicate_source_paths == 0
}


# ============================================================
# FINAL STATUS
# ============================================================

final_status = all(
    checks.values()
)


# ============================================================
# FEATURE ANALYSIS TABLE
# ============================================================

feature_analysis_records = []


for column in features.columns:

    feature_analysis_records.append({

        "feature":
            column,

        "candidate_physical_feature":
            column in candidate_features,

        "proposed_model_feature":
            column in model_features,

        "identifier_column":
            column in identifier_columns,

        "future_column":
            column in future_information_columns,

        "future_metadata":
            column in future_metadata_columns,

        "target_keyword":
            column in target_keyword_columns,

        "exact_target_column":
            column == TARGET_COLUMN
    })


feature_analysis_df = pd.DataFrame(
    feature_analysis_records
)


feature_analysis_df.to_csv(
    OUTPUT_FEATURE,
    index=False
)


# ============================================================
# BUILD REPORT
# ============================================================

report = {

    "stage":
        "1.4",

    "title":
        "Feature / Target Leakage Audit",

    "status":
        "PASS"
        if final_status
        else "REVIEW",

    "prediction_problem": {

        "input":
            "Physical damage measurements available at frame t",

        "target":
            "crack_area_ratio at frame t+1",

        "primary_horizon":
            1
    },

    "dataset": {

        "feature_records":
            int(
                len(
                    features
                )
            ),

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
            int(
                primary[
                    SEQUENCE_COLUMN
                ].nunique()
            ),

        "components":
            int(
                primary[
                    COMPONENT_COLUMN
                ].nunique()
            )
    },

    "alignment": {

        "missing_current_features":
            missing_current_features,

        "current_sequence_mismatches":
            current_sequence_mismatch,

        "current_component_mismatches":
            current_component_mismatch,

        "current_split_mismatches":
            current_split_mismatch,

        "current_damage_mismatches":
            current_damage_mismatch_count,

        "future_records_missing":
            future_missing_count,

        "future_target_mismatches":
            future_target_mismatch_count,

        "future_sequence_mismatches":
            future_sequence_mismatch_count,

        "future_component_mismatches":
            future_component_mismatch_count,

        "future_split_mismatches":
            future_split_mismatch_count
    },

    "leakage": {

        "direct_target_column_present":
            direct_target_column_present,

        "target_keyword_columns":
            target_keyword_columns,

        "future_information_columns":
            future_information_columns,

        "future_metadata_columns":
            future_metadata_columns,

        "direct_numerical_target_matches":
            direct_numerical_matches
    },

    "split_integrity": {

        "sequence_split_conflicts":
            sequence_split_conflicts,

        "component_split_conflicts":
            component_split_conflicts,

        "source_image_split_conflicts":
            source_split_conflicts,

        "duplicate_source_paths":
            duplicate_source_paths
    },

    "model_features":
        model_features,

    "checks":
        checks,

    "outputs": {

        "feature_analysis":
            OUTPUT_FEATURE,

        "correlation_analysis":
            OUTPUT_CORRELATION,

        "report":
            OUTPUT_JSON
    },

    "next_stage":
        "Stage 2 baseline ML regression modeling."
}


# ============================================================
# SAVE OUTPUTS
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
print("STAGE 1.4 SUMMARY")
print("=" * 70)

print(
    "Feature records:",
    len(
        features
    )
)

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
    "Final model features:",
    len(
        model_features
    )
)

print("")

print(
    "Direct target column present:",
    direct_target_column_present
)

print(
    "Future information columns:",
    len(
        future_information_columns
    )
)

print(
    "Future metadata columns:",
    len(
        future_metadata_columns
    )
)

print(
    "Direct numerical target matches:",
    len(
        direct_numerical_matches
    )
)

print("")

print(
    "Current feature rows missing:",
    missing_current_features
)

print(
    "Current sequence mismatches:",
    current_sequence_mismatch
)

print(
    "Current component mismatches:",
    current_component_mismatch
)

print(
    "Current split mismatches:",
    current_split_mismatch
)

print(
    "Current damage mismatches:",
    current_damage_mismatch_count
)

print("")

print(
    "Future records missing:",
    future_missing_count
)

print(
    "Future target mismatches:",
    future_target_mismatch_count
)

print(
    "Future sequence mismatches:",
    future_sequence_mismatch_count
)

print(
    "Future component mismatches:",
    future_component_mismatch_count
)

print(
    "Future split mismatches:",
    future_split_mismatch_count
)

print("")

print(
    "Sequence split conflicts:",
    sequence_split_conflicts
)

print(
    "Component split conflicts:",
    component_split_conflicts
)

print(
    "Source image split conflicts:",
    source_split_conflicts
)

print(
    "Duplicate source paths:",
    duplicate_source_paths
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
    "Feature analysis:"
)

print(
    OUTPUT_FEATURE
)

print("")

print(
    "Target correlation analysis:"
)

print(
    OUTPUT_CORRELATION
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

    print("")
    print(
        "No feature/target leakage was detected."
    )

    print(
        "The dataset is ready for baseline ML regression."
    )

else:

    print(
        "FINAL STATUS: REVIEW"
    )

    print("")
    print(
        "Potential leakage or alignment issues were detected."
    )

print("")