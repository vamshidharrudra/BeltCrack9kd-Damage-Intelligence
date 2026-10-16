import os
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

SPATIAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

SPATIAL_FEATURE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_features.csv"
)

TEMPORAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

DAMAGE_FEATURE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_1_0_damage_features.csv"
)

OUTPUT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_physical_damage_progression.csv"
)

SEQUENCE_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_sequence_progression_summary.csv"
)

STATE_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_progression_state_summary.csv"
)

SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_physical_damage_progression_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns, name):

    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing)
        )


def percentile_value(series, q):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            q
        )
    )


def unique_preserve_order(values):

    return list(
        dict.fromkeys(values)
    )


def first_existing_column(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def calculate_streak(values, positive=True):

    result = []
    streak = 0

    for value in values:

        if pd.isna(value):

            streak = 0
            result.append(0)
            continue

        if positive:
            condition = value > 0
        else:
            condition = value < 0

        if condition:
            streak += 1
        else:
            streak = 0

        result.append(streak)

    return result


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 3.4 - PHYSICAL DAMAGE PROGRESSION MODELING")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

files_to_check = [
    (
        SPATIAL_FILE,
        "Stage 3.0 spatial track pairs"
    ),
    (
        SPATIAL_FEATURE_FILE,
        "Stage 3.1 spatial features"
    ),
    (
        TEMPORAL_FILE,
        "Stage 2.2 temporal features"
    ),
    (
        DAMAGE_FEATURE_FILE,
        "Stage 1.0 damage features"
    )
]

for path, label in files_to_check:

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
    "\nLoading Stage 3.0 spatial tracks..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial track records:",
    len(spatial)
)


print(
    "\nLoading Stage 1.0 damage features..."
)

damage = pd.read_csv(
    DAMAGE_FEATURE_FILE
)

print(
    "Damage feature records:",
    len(damage)
)


print(
    "\nLoading Stage 2.2 temporal features..."
)

# Copy immediately to avoid pandas fragmentation warnings.
temporal = pd.read_csv(
    TEMPORAL_FILE
).copy()

print(
    "Temporal feature records:",
    len(temporal)
)


# ============================================================
# REQUIRED SPATIAL COLUMNS
# ============================================================

required_spatial = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame",
    "current_total_area",
    "future_total_area"
]

require_columns(
    spatial,
    required_spatial,
    "Stage 3.0 spatial track pairs"
)


# ============================================================
# NORMALIZE SPATIAL FRAMES
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
).copy()

spatial["current_frame"] = (
    spatial["current_frame"]
    .astype(int)
)

spatial["future_frame"] = (
    spatial["future_frame"]
    .astype(int)
)


# ============================================================
# VERIFY CONSECUTIVE FRAMES
# ============================================================

invalid_frame_pairs = (
    spatial["future_frame"]
    !=
    spatial["current_frame"] + 1
)

invalid_count = int(
    invalid_frame_pairs.sum()
)

print(
    "\nInvalid spatial frame transitions:",
    invalid_count
)

if invalid_count > 0:

    raise RuntimeError(
        "Stage 3.0 contains non-consecutive frame pairs."
    )


# ============================================================
# NORMALIZE TEMPORAL FRAME INFORMATION
# ============================================================

if "frame_number" in temporal.columns:

    temporal["current_frame"] = pd.to_numeric(
        temporal["frame_number"],
        errors="coerce"
    )

elif "current_frame" in temporal.columns:

    temporal["current_frame"] = pd.to_numeric(
        temporal["current_frame"],
        errors="coerce"
    )

else:

    raise RuntimeError(
        "Stage 2.2 contains neither frame_number "
        "nor current_frame."
    )


if "future_frame" in temporal.columns:

    temporal["future_frame"] = pd.to_numeric(
        temporal["future_frame"],
        errors="coerce"
    )

else:

    temporal["future_frame"] = (
        temporal["current_frame"] + 1
    )


temporal = temporal.dropna(
    subset=[
        "current_frame",
        "future_frame"
    ]
).copy()

temporal["current_frame"] = (
    temporal["current_frame"]
    .astype(int)
)

temporal["future_frame"] = (
    temporal["future_frame"]
    .astype(int)
)


# ============================================================
# BUILD VALIDATED FRAME LOOKUP
# ============================================================

print(
    "\nConstructing validated Stage 2.2 frame mapping..."
)

frame_lookup_columns = [
    "sequence",
    "component",
    "split",
    "current_frame"
]

possible_identity_columns = [
    "record_id",
    "original_image_filename",
    "original_json_filename",
    "canonical_image_filename",
    "canonical_json_filename"
]

for col in possible_identity_columns:

    if col in temporal.columns:

        frame_lookup_columns.append(
            col
        )


frame_lookup_columns = unique_preserve_order(
    frame_lookup_columns
)

frame_lookup = temporal[
    frame_lookup_columns
].copy()

frame_lookup = frame_lookup.drop_duplicates(
    keep="first"
)

print(
    "Frame lookup records:",
    len(frame_lookup)
)


# ============================================================
# MAP STAGE 1.0 TO FRAME NUMBERS
# ============================================================

print(
    "\nConstructing Stage 1.0 damage frame mapping..."
)

print(
    "Available Stage 1.0 identity columns:"
)

for col in [
    "record_id",
    "sequence",
    "component",
    "split",
    "original_image_filename",
    "original_json_filename",
    "canonical_image_filename",
    "canonical_json_filename"
]:

    if col in damage.columns:

        print(
            " ",
            col
        )


# ------------------------------------------------------------
# Preferred mapping: record_id
# ------------------------------------------------------------

if (
    "record_id" in damage.columns
    and
    "record_id" in frame_lookup.columns
):

    print(
        "\nUsing record_id to map Stage 1.0 "
        "damage records to validated frames."
    )

    damage_mapping = frame_lookup[
        [
            "record_id",
            "sequence",
            "component",
            "split",
            "current_frame"
        ]
    ].drop_duplicates(
        subset=["record_id"],
        keep="first"
    )

    damage = damage.merge(
        damage_mapping,
        on="record_id",
        how="left",
        suffixes=(
            "",
            "_mapped"
        )
    )

    for col in [
        "sequence",
        "component",
        "split"
    ]:

        mapped_col = (
            col + "_mapped"
        )

        if mapped_col in damage.columns:

            if col not in damage.columns:

                damage[col] = damage[
                    mapped_col
                ]

            else:

                damage[col] = (
                    damage[col]
                    .fillna(
                        damage[
                            mapped_col
                        ]
                    )
                )

            damage.drop(
                columns=[mapped_col],
                inplace=True
            )


# ------------------------------------------------------------
# Fallback: canonical JSON filename
# ------------------------------------------------------------

elif (
    "canonical_json_filename"
    in damage.columns
    and
    "canonical_json_filename"
    in frame_lookup.columns
):

    print(
        "\nUsing canonical_json_filename "
        "for frame mapping."
    )

    damage_mapping = frame_lookup[
        [
            "canonical_json_filename",
            "sequence",
            "component",
            "split",
            "current_frame"
        ]
    ].drop_duplicates(
        subset=[
            "canonical_json_filename"
        ],
        keep="first"
    )

    damage = damage.merge(
        damage_mapping,
        on="canonical_json_filename",
        how="left",
        suffixes=(
            "",
            "_mapped"
        )
    )

    for col in [
        "sequence",
        "component",
        "split"
    ]:

        mapped_col = (
            col + "_mapped"
        )

        if mapped_col in damage.columns:

            if col not in damage.columns:

                damage[col] = damage[
                    mapped_col
                ]

            else:

                damage[col] = (
                    damage[col]
                    .fillna(
                        damage[
                            mapped_col
                        ]
                    )
                )

            damage.drop(
                columns=[mapped_col],
                inplace=True
            )


# ------------------------------------------------------------
# Fallback: original JSON filename
# ------------------------------------------------------------

elif (
    "original_json_filename"
    in damage.columns
    and
    "original_json_filename"
    in frame_lookup.columns
):

    print(
        "\nUsing original_json_filename "
        "for frame mapping."
    )

    damage_mapping = frame_lookup[
        [
            "original_json_filename",
            "sequence",
            "component",
            "split",
            "current_frame"
        ]
    ].drop_duplicates(
        subset=[
            "original_json_filename"
        ],
        keep="first"
    )

    damage = damage.merge(
        damage_mapping,
        on="original_json_filename",
        how="left",
        suffixes=(
            "",
            "_mapped"
        )
    )

    for col in [
        "sequence",
        "component",
        "split"
    ]:

        mapped_col = (
            col + "_mapped"
        )

        if mapped_col in damage.columns:

            if col not in damage.columns:

                damage[col] = damage[
                    mapped_col
                ]

            else:

                damage[col] = (
                    damage[col]
                    .fillna(
                        damage[
                            mapped_col
                        ]
                    )
                )

            damage.drop(
                columns=[mapped_col],
                inplace=True
            )


else:

    raise RuntimeError(
        "Could not map Stage 1.0 damage records "
        "to validated frame numbers."
    )


# ============================================================
# VERIFY FRAME MAPPING
# ============================================================

if "current_frame" not in damage.columns:

    raise RuntimeError(
        "Stage 1.0 still has no validated current_frame."
    )

damage["current_frame"] = pd.to_numeric(
    damage["current_frame"],
    errors="coerce"
)

damage = damage.dropna(
    subset=[
        "current_frame"
    ]
).copy()

damage["current_frame"] = (
    damage["current_frame"]
    .astype(int)
)

print(
    "Mapped Stage 1.0 records:",
    len(damage)
)


# ============================================================
# REQUIRED DAMAGE COLUMNS
# ============================================================

damage_columns = [
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

missing_damage_columns = [
    col
    for col in damage_columns
    if col not in damage.columns
]

if missing_damage_columns:

    raise RuntimeError(
        "Stage 1.0 is missing required damage columns:\n"
        + "\n".join(
            missing_damage_columns
        )
    )


# ============================================================
# DAMAGE KEY
# ============================================================

damage_key = [
    "sequence",
    "component",
    "split",
    "current_frame"
]


duplicate_damage_keys = int(
    damage.duplicated(
        subset=damage_key
    ).sum()
)

print(
    "Duplicate Stage 1.0 frame keys:",
    duplicate_damage_keys
)

if duplicate_damage_keys > 0:

    raise RuntimeError(
        "Duplicate Stage 1.0 sequence/frame keys found."
    )


# ============================================================
# CURRENT DAMAGE TABLE
# ============================================================

damage_current = damage[
    damage_key
    + damage_columns
].copy()

damage_current = damage_current.rename(
    columns={
        "crack_area_ratio":
            "damage_current",

        "crack_count":
            "crack_count_current",

        "largest_crack_area_ratio":
            "largest_crack_ratio_current",

        "largest_crack_area_px":
            "largest_crack_area_current",

        "total_crack_area_px":
            "total_crack_area_current",

        "mean_crack_area_px":
            "mean_crack_area_current",

        "total_crack_perimeter_px":
            "total_perimeter_current",

        "mean_crack_perimeter_px":
            "mean_perimeter_current",

        "annotation_shape_count":
            "annotation_count_current",

        "image_area_px":
            "image_area_current"
    }
)


# ============================================================
# FUTURE DAMAGE TABLE
# ============================================================

damage_future = damage[
    damage_key
    + damage_columns
].copy()

damage_future = damage_future.rename(
    columns={
        "current_frame":
            "future_frame",

        "crack_area_ratio":
            "damage_future",

        "crack_count":
            "crack_count_future",

        "largest_crack_area_ratio":
            "largest_crack_ratio_future",

        "largest_crack_area_px":
            "largest_crack_area_future",

        "total_crack_area_px":
            "total_crack_area_future",

        "mean_crack_area_px":
            "mean_crack_area_future",

        "total_crack_perimeter_px":
            "total_perimeter_future",

        "mean_crack_perimeter_px":
            "mean_perimeter_future",

        "annotation_shape_count":
            "annotation_count_future",

        "image_area_px":
            "image_area_future"
    }
)


# ============================================================
# CURRENT DAMAGE MERGE
# ============================================================

progression = spatial.merge(
    damage_current,
    on=[
        "sequence",
        "component",
        "split",
        "current_frame"
    ],
    how="left"
)


# ============================================================
# FUTURE DAMAGE MERGE
# ============================================================

progression = progression.merge(
    damage_future,
    on=[
        "sequence",
        "component",
        "split",
        "future_frame"
    ],
    how="left"
)


print(
    "\nRecords after damage mapping:",
    len(progression)
)


# ============================================================
# IMPORTANT:
# THE LAST FRAME OF EACH SEQUENCE HAS NO FUTURE FRAME.
#
# Stage 3.0 contains 14,058 transitions from 14,087 frames:
#
# 14,087 - 29 sequences = 14,058 transitions
#
# Therefore exactly 29 missing future mappings are expected.
# ============================================================

missing_current_mask = progression[
    "damage_current"
].isna()

missing_future_mask = progression[
    "damage_future"
].isna()

missing_current = int(
    missing_current_mask.sum()
)

missing_future = int(
    missing_future_mask.sum()
)

print(
    "Missing current damage mappings:",
    missing_current
)

print(
    "Missing future damage mappings:",
    missing_future
)


# ============================================================
# CURRENT DAMAGE MUST NEVER BE MISSING
# ============================================================

if missing_current > 0:

    raise RuntimeError(
        "Unexpected missing current damage mappings."
    )


# ============================================================
# EXPECTED FINAL-FRAME CHECK
# ============================================================

if missing_future > 0:

    missing_future_rows = progression[
        missing_future_mask
    ].copy()

    expected_final_frames = []

    for sequence, group_df in damage.groupby(
        "sequence"
    ):

        expected_final_frames.append({
            "sequence":
                sequence,

            "final_frame":
                int(
                    group_df[
                        "current_frame"
                    ].max()
                )
        })

    expected_final_frames = pd.DataFrame(
        expected_final_frames
    )

    missing_future_rows = missing_future_rows.merge(
        expected_final_frames,
        on="sequence",
        how="left"
    )

    correct_final_frame_count = int(
        (
            missing_future_rows[
                "future_frame"
            ]
            ==
            missing_future_rows[
                "final_frame"
            ]
            +
            1
        ).sum()
    )

    print(
        "Missing future mappings that are expected "
        "sequence-end transitions:",
        correct_final_frame_count
    )

    if (
        correct_final_frame_count
        !=
        missing_future
    ):

        raise RuntimeError(
            "Some missing future mappings are not "
            "expected final-frame cases."
        )

    # Remove only the final frames because they cannot
    # participate in a t -> t+1 transition.
    progression = progression[
        ~missing_future_mask
    ].copy()


print(
    "Valid progression transitions:",
    len(progression)
)


# ============================================================
# EXPECTED TRANSITION COUNT
# ============================================================

expected_transitions = (
    damage[
        "sequence"
    ].nunique()
)

expected_transitions = (
    len(damage)
    -
    expected_transitions
)

print(
    "Expected consecutive transitions:",
    expected_transitions
)

if len(progression) != expected_transitions:

    raise RuntimeError(
        "Unexpected progression transition count."
    )


# ============================================================
# BASIC DAMAGE CHANGE FEATURES
# ============================================================

progression[
    "absolute_damage_change"
] = (
    progression[
        "damage_future"
    ]
    -
    progression[
        "damage_current"
    ]
)


progression[
    "relative_damage_change"
] = (
    (
        progression[
            "damage_future"
        ]
        -
        progression[
            "damage_current"
        ]
    )
    /
    progression[
        "damage_current"
    ]
    .abs()
    .replace(
        0,
        np.nan
    )
)


progression[
    "damage_growth_rate"
] = (
    progression[
        "absolute_damage_change"
    ]
)


progression[
    "crack_count_change"
] = (
    progression[
        "crack_count_future"
    ]
    -
    progression[
        "crack_count_current"
    ]
)


progression[
    "largest_crack_change"
] = (
    progression[
        "largest_crack_ratio_future"
    ]
    -
    progression[
        "largest_crack_ratio_current"
    ]
)


progression[
    "largest_crack_area_change_px"
] = (
    progression[
        "largest_crack_area_future"
    ]
    -
    progression[
        "largest_crack_area_current"
    ]
)


progression[
    "total_area_change_px"
] = (
    progression[
        "total_crack_area_future"
    ]
    -
    progression[
        "total_crack_area_current"
    ]
)


progression[
    "perimeter_change_px"
] = (
    progression[
        "total_perimeter_future"
    ]
    -
    progression[
        "total_perimeter_current"
    ]
)


progression[
    "mean_crack_area_change_px"
] = (
    progression[
        "mean_crack_area_future"
    ]
    -
    progression[
        "mean_crack_area_current"
    ]
)


# ============================================================
# SPATIAL TRACK METRICS
# ============================================================

spatial_metric_candidates = {

    "persistent_fraction": [
        "persistent_fraction",
        "matched_fraction_current"
    ],

    "future_coverage": [
        "future_coverage",
        "matched_fraction_future"
    ],

    "match_iou": [
        "mean_match_iou"
    ],

    "center_distance": [
        "mean_center_distance"
    ],

    "area_similarity": [
        "mean_area_similarity"
    ],

    "match_score": [
        "mean_match_score"
    ],

    "track_continuity": [
        "track_continuity_quality",
        "continuity_score"
    ]
}


for output_name, candidates in (
    spatial_metric_candidates.items()
):

    source = first_existing_column(
        progression,
        candidates
    )

    if source is not None:

        progression[
            output_name
        ] = progression[
            source
        ]

    else:

        progression[
            output_name
        ] = np.nan


# ============================================================
# MATCHED AREA EVOLUTION
# ============================================================

if (
    "matched_current_area"
    in progression.columns
    and
    "matched_future_area"
    in progression.columns
):

    progression[
        "matched_area_change"
    ] = (
        progression[
            "matched_future_area"
        ]
        -
        progression[
            "matched_current_area"
        ]
    )

    progression[
        "matched_area_relative_change"
    ] = (
        (
            progression[
                "matched_future_area"
            ]
            -
            progression[
                "matched_current_area"
            ]
        )
        /
        progression[
            "matched_current_area"
        ]
        .abs()
        .replace(
            0,
            np.nan
        )
    )

else:

    progression[
        "matched_area_change"
    ] = np.nan

    progression[
        "matched_area_relative_change"
    ] = np.nan


# ============================================================
# DAMAGE CHANGE DIRECTION
# ============================================================

EPSILON = 1e-12

change = progression[
    "absolute_damage_change"
]

progression[
    "damage_change_direction"
] = np.select(
    [
        change > EPSILON,
        change < -EPSILON
    ],
    [
        "INCREASE",
        "DECREASE"
    ],
    default="STABLE"
)


# ============================================================
# EMPIRICAL THRESHOLDS
# ============================================================

absolute_change = (
    change.abs()
)

p90 = percentile_value(
    absolute_change,
    90
)

p95 = percentile_value(
    absolute_change,
    95
)

p99 = percentile_value(
    absolute_change,
    99
)


print(
    "\nEmpirical absolute damage-change thresholds:"
)

print(
    "P90:",
    f"{p90:.8f}"
)

print(
    "P95:",
    f"{p95:.8f}"
)

print(
    "P99:",
    f"{p99:.8f}"
)


# ============================================================
# PROGRESSION STATES
# ============================================================

progression[
    "progression_state"
] = np.select(
    [
        change >= p95,

        change <= -p95,

        change > EPSILON,

        change < -EPSILON
    ],
    [
        "RAPID_INCREASE",

        "RAPID_DECREASE",

        "INCREASING",

        "DECREASING"
    ],
    default="STABLE"
)


# ============================================================
# DAMAGE CHANGE MAGNITUDE
# ============================================================

progression[
    "damage_change_magnitude"
] = np.select(
    [
        absolute_change >= p99,

        absolute_change >= p95,

        absolute_change >= p90,

        absolute_change > EPSILON
    ],
    [
        "EXTREME",

        "VERY_LARGE",

        "LARGE",

        "SMALL"
    ],
    default="NONE"
)


# ============================================================
# SPATIAL CONTINUITY CATEGORY
# ============================================================

continuity = progression[
    "track_continuity"
]

progression[
    "spatial_continuity_category"
] = np.select(
    [
        continuity >= 0.75,

        continuity >= 0.50,

        continuity > 0
    ],
    [
        "HIGH",

        "MODERATE",

        "LOW"
    ],
    default="UNKNOWN"
)


# ============================================================
# PROGRESSION + SPATIAL PATTERN
# ============================================================

progression[
    "progression_spatial_pattern"
] = "UNKNOWN"


increase = progression[
    "progression_state"
].isin(
    [
        "INCREASING",
        "RAPID_INCREASE"
    ]
)

decrease = progression[
    "progression_state"
].isin(
    [
        "DECREASING",
        "RAPID_DECREASE"
    ]
)

high = (
    progression[
        "spatial_continuity_category"
    ]
    ==
    "HIGH"
)

moderate = (
    progression[
        "spatial_continuity_category"
    ]
    ==
    "MODERATE"
)


progression.loc[
    increase & high,
    "progression_spatial_pattern"
] = "PERSISTENT_DAMAGE_GROWTH"


progression.loc[
    decrease & high,
    "progression_spatial_pattern"
] = "PERSISTENT_DAMAGE_REDUCTION"


progression.loc[
    increase & moderate,
    "progression_spatial_pattern"
] = "GROWTH_WITH_SPATIAL_CHANGE"


progression.loc[
    decrease & moderate,
    "progression_spatial_pattern"
] = "REDUCTION_WITH_SPATIAL_CHANGE"


progression.loc[
    increase
    & ~high
    & ~moderate,
    "progression_spatial_pattern"
] = "INCREASE_WITH_LOW_CONTINUITY"


progression.loc[
    decrease
    & ~high
    & ~moderate,
    "progression_spatial_pattern"
] = "DECREASE_WITH_LOW_CONTINUITY"


progression.loc[
    progression[
        "progression_state"
    ]
    ==
    "STABLE",
    "progression_spatial_pattern"
] = "STABLE_DAMAGE"


# ============================================================
# SORT
# ============================================================

progression = progression.sort_values(
    [
        "sequence",
        "current_frame"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SEQUENTIAL DAMAGE DYNAMICS
# ============================================================

group = progression.groupby(
    "sequence"
)


progression[
    "previous_damage_change"
] = group[
    "absolute_damage_change"
].shift(1)


progression[
    "damage_change_acceleration"
] = (
    progression[
        "absolute_damage_change"
    ]
    -
    progression[
        "previous_damage_change"
    ]
)


progression[
    "previous_relative_damage_change"
] = group[
    "relative_damage_change"
].shift(1)


progression[
    "relative_change_acceleration"
] = (
    progression[
        "relative_damage_change"
    ]
    -
    progression[
        "previous_relative_damage_change"
    ]
)


# ============================================================
# CONSECUTIVE INCREASE / DECREASE
# ============================================================

progression[
    "consecutive_increase"
] = progression.groupby(
    "sequence"
)[
    "absolute_damage_change"
].transform(
    lambda x:
        calculate_streak(
            x,
            positive=True
        )
)


progression[
    "consecutive_decrease"
] = progression.groupby(
    "sequence"
)[
    "absolute_damage_change"
].transform(
    lambda x:
        calculate_streak(
            x,
            positive=False
        )
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

sequence_rows = []

for sequence, group_df in progression.groupby(
    "sequence"
):

    state_counts_local = (
        group_df[
            "progression_state"
        ]
        .value_counts()
        .to_dict()
    )

    pattern_counts_local = (
        group_df[
            "progression_spatial_pattern"
        ]
        .value_counts()
        .to_dict()
    )

    sequence_rows.append({

        "sequence":
            sequence,

        "component":
            group_df[
                "component"
            ].iloc[0],

        "split":
            group_df[
                "split"
            ].iloc[0],

        "transition_count":
            int(len(group_df)),

        "mean_damage_current":
            float(
                group_df[
                    "damage_current"
                ].mean()
            ),

        "mean_damage_future":
            float(
                group_df[
                    "damage_future"
                ].mean()
            ),

        "mean_damage_change":
            float(
                group_df[
                    "absolute_damage_change"
                ].mean()
            ),

        "mean_absolute_damage_change":
            float(
                group_df[
                    "absolute_damage_change"
                ]
                .abs()
                .mean()
            ),

        "max_damage_increase":
            float(
                group_df[
                    "absolute_damage_change"
                ].max()
            ),

        "max_damage_decrease":
            float(
                group_df[
                    "absolute_damage_change"
                ].min()
            ),

        "rapid_increase_count":
            int(
                state_counts_local.get(
                    "RAPID_INCREASE",
                    0
                )
            ),

        "rapid_decrease_count":
            int(
                state_counts_local.get(
                    "RAPID_DECREASE",
                    0
                )
            ),

        "increasing_count":
            int(
                state_counts_local.get(
                    "INCREASING",
                    0
                )
            ),

        "decreasing_count":
            int(
                state_counts_local.get(
                    "DECREASING",
                    0
                )
            ),

        "stable_count":
            int(
                state_counts_local.get(
                    "STABLE",
                    0
                )
            ),

        "mean_spatial_continuity":
            float(
                group_df[
                    "track_continuity"
                ].mean()
            ),

        "mean_match_iou":
            float(
                group_df[
                    "match_iou"
                ].mean()
            ),

        "mean_center_distance":
            float(
                group_df[
                    "center_distance"
                ].mean()
            ),

        "persistent_growth_count":
            int(
                pattern_counts_local.get(
                    "PERSISTENT_DAMAGE_GROWTH",
                    0
                )
            ),

        "persistent_reduction_count":
            int(
                pattern_counts_local.get(
                    "PERSISTENT_DAMAGE_REDUCTION",
                    0
                )
            )
    })


sequence_summary = pd.DataFrame(
    sequence_rows
)


# ============================================================
# STATE SUMMARY
# ============================================================

state_summary = (
    progression[
        [
            "progression_state",
            "damage_change_magnitude",
            "progression_spatial_pattern"
        ]
    ]
    .value_counts()
    .reset_index(
        name="record_count"
    )
)


state_summary[
    "percentage"
] = (
    state_summary[
        "record_count"
    ]
    /
    len(progression)
    *
    100.0
)


# ============================================================
# GLOBAL COUNTS
# ============================================================

state_counts = (
    progression[
        "progression_state"
    ]
    .value_counts()
    .to_dict()
)

pattern_counts = (
    progression[
        "progression_spatial_pattern"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# CORRELATIONS
# ============================================================

correlation_features = [
    "track_continuity",
    "persistent_fraction",
    "future_coverage",
    "match_iou",
    "center_distance",
    "area_similarity",
    "match_score",
    "matched_area_change",
    "matched_area_relative_change",
    "crack_count_change",
    "largest_crack_change",
    "total_area_change_px"
]

correlations = {}

for feature in correlation_features:

    if feature not in progression.columns:
        continue

    valid = progression[
        [
            feature,
            "absolute_damage_change"
        ]
    ].dropna()

    if len(valid) < 3:

        correlations[feature] = None

    else:

        correlations[feature] = float(
            valid[
                feature
            ].corr(
                valid[
                    "absolute_damage_change"
                ]
            )
        )


# ============================================================
# INTEGRITY CHECKS
# ============================================================

duplicate_keys = int(
    progression.duplicated(
        subset=[
            "sequence",
            "component",
            "split",
            "current_frame",
            "future_frame"
        ]
    ).sum()
)


invalid_direction = int(
    (
        progression[
            "future_frame"
        ]
        <=
        progression[
            "current_frame"
        ]
    ).sum()
)


sequence_split_conflicts = int(
    (
        progression.groupby(
            "sequence"
        )["split"].nunique()
        > 1
    ).sum()
)


component_split_conflicts = int(
    (
        progression.groupby(
            "component"
        )["split"].nunique()
        > 1
    ).sum()
)


# ============================================================
# SAVE
# ============================================================

progression.to_csv(
    OUTPUT_FILE,
    index=False
)

sequence_summary.to_csv(
    SEQUENCE_SUMMARY_FILE,
    index=False
)

state_summary.to_csv(
    STATE_SUMMARY_FILE,
    index=False
)


summary = {

    "stage":
        "3.4",

    "status":
        "PASS",

    "records":
        int(len(progression)),

    "sequences":
        int(
            progression[
                "sequence"
            ].nunique()
        ),

    "components":
        int(
            progression[
                "component"
            ].nunique()
        ),

    "invalid_frame_pairs":
        invalid_count,

    "missing_current_damage":
        missing_current,

    "missing_future_damage_before_expected_final_frame_removal":
        missing_future,

    "expected_final_frame_transitions_removed":
        missing_future,

    "duplicate_transition_keys":
        duplicate_keys,

    "invalid_temporal_direction":
        invalid_direction,

    "sequence_split_conflicts":
        sequence_split_conflicts,

    "component_split_conflicts":
        component_split_conflicts,

    "damage_change_thresholds": {

        "p90":
            p90,

        "p95":
            p95,

        "p99":
            p99
    },

    "progression_state_counts":
        {
            key:
                int(value)
            for key, value
            in state_counts.items()
        },

    "progression_state_percentages":
        {
            key:
                float(
                    value
                    /
                    len(progression)
                    *
                    100.0
                )
            for key, value
            in state_counts.items()
        },

    "spatial_pattern_counts":
        {
            key:
                int(value)
            for key, value
            in pattern_counts.items()
        },

    "mean_absolute_damage_change":
        float(
            progression[
                "absolute_damage_change"
            ]
            .abs()
            .mean()
        ),

    "median_absolute_damage_change":
        float(
            progression[
                "absolute_damage_change"
            ]
            .abs()
            .median()
        ),

    "mean_damage_change":
        float(
            progression[
                "absolute_damage_change"
            ].mean()
        ),

    "correlations_with_absolute_damage_change":
        correlations,

    "output_files": {

        "progression":
            OUTPUT_FILE,

        "sequence_summary":
            SEQUENCE_SUMMARY_FILE,

        "state_summary":
            STATE_SUMMARY_FILE,

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
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.4 SUMMARY")
print("=" * 70)

print(
    "Progression records:",
    len(progression)
)

print(
    "Sequences:",
    progression[
        "sequence"
    ].nunique()
)

print(
    "Components:",
    progression[
        "component"
    ].nunique()
)

print(
    "\nDamage-change thresholds:"
)

print(
    "  P90:",
    f"{p90:.8f}"
)

print(
    "  P95:",
    f"{p95:.8f}"
)

print(
    "  P99:",
    f"{p99:.8f}"
)

print(
    "\nProgression states:"
)

for state in [
    "RAPID_INCREASE",
    "RAPID_DECREASE",
    "INCREASING",
    "DECREASING",
    "STABLE"
]:

    count = state_counts.get(
        state,
        0
    )

    percentage = (
        count
        /
        len(progression)
        *
        100.0
    )

    print(
        f"  {state}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


print(
    "\nSpatial progression patterns:"
)

for pattern, count in sorted(
    pattern_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"  {pattern}: {count}"
    )


print(
    "\nMean absolute damage change:",
    f"{progression['absolute_damage_change'].abs().mean():.8f}"
)

print(
    "Median absolute damage change:",
    f"{progression['absolute_damage_change'].abs().median():.8f}"
)


print(
    "\nIntegrity checks:"
)

print(
    "  Invalid frame pairs:",
    invalid_count
)

print(
    "  Missing current damage:",
    missing_current
)

print(
    "  Missing future damage before final-frame removal:",
    missing_future
)

print(
    "  Duplicate transition keys:",
    duplicate_keys
)

print(
    "  Invalid temporal direction:",
    invalid_direction
)

print(
    "  Sequence split conflicts:",
    sequence_split_conflicts
)

print(
    "  Component split conflicts:",
    component_split_conflicts
)


print(
    "\nOutput progression file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nOutput sequence summary:"
)

print(
    SEQUENCE_SUMMARY_FILE
)

print(
    "\nOutput state summary:"
)

print(
    STATE_SUMMARY_FILE
)

print(
    "\nOutput summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nFINAL STATUS: PASS"
)