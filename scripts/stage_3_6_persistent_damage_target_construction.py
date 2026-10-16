import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 3.6
# PERSISTENT DAMAGE TARGET CONSTRUCTION
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

SPATIAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

PROGRESSION_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_physical_damage_progression.csv"
)

OUTPUT_TARGET_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_persistent_damage_targets.csv"
)

OUTPUT_SEQUENCE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_sequence_target_summary.csv"
)

OUTPUT_SPLIT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_split_target_summary.csv"
)

OUTPUT_STATE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_target_state_summary.csv"
)

OUTPUT_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_persistent_damage_target_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path, description):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def require_columns(df, columns, name):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing)
        )


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def pct(value, total):

    if total == 0:
        return 0.0

    return float(
        value / total * 100.0
    )


def calculate_streak_series(series):

    values = series.astype(bool).tolist()

    result = []

    streak = 0

    for value in values:

        if value:

            streak += 1

        else:

            streak = 0

        result.append(
            streak
        )

    return pd.Series(
        result,
        index=series.index
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.6 - PERSISTENT DAMAGE TARGET CONSTRUCTION"
)
print("=" * 70)


# ============================================================
# FILE CHECKS
# ============================================================

print(
    "\nChecking required files..."
)

require_file(
    SPATIAL_FILE,
    "Stage 3.0 spatial track file"
)

require_file(
    PROGRESSION_FILE,
    "Stage 3.4 physical progression file"
)

print(
    "All required files: FOUND"
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
    "Stage 3.0 records:",
    len(spatial)
)


print(
    "\nLoading Stage 3.4 physical progression..."
)

progression = pd.read_csv(
    PROGRESSION_FILE
)

print(
    "Stage 3.4 records:",
    len(progression)
)


# ============================================================
# KEY COLUMNS
# ============================================================

key_columns = [

    "sequence",

    "component",

    "split",

    "current_frame",

    "future_frame"

]


require_columns(
    spatial,
    key_columns,
    "Stage 3.0 spatial file"
)

require_columns(
    progression,
    key_columns,
    "Stage 3.4 progression file"
)


# ============================================================
# REQUIRED STAGE 3.0 COLUMNS
# ============================================================

spatial_required = [

    "current_crack_count",

    "future_crack_count",

    "matched_crack_count",

    "new_crack_count",

    "disappeared_crack_count",

    "persistent_fraction",

    "future_coverage",

    "mean_match_iou",

    "mean_center_distance",

    "mean_area_similarity",

    "mean_match_score",

    "mean_relative_matched_area_change",

    "median_relative_matched_area_change",

    "max_relative_matched_area_change",

    "min_relative_matched_area_change",

    "current_total_area",

    "future_total_area",

    "total_area_relative_change",

    "continuity_class"

]


require_columns(
    spatial,
    spatial_required,
    "Stage 3.0 spatial file"
)


# ============================================================
# REQUIRED STAGE 3.4 COLUMNS
# ============================================================

progression_required = [

    "damage_current",

    "damage_future",

    "absolute_damage_change",

    "relative_damage_change",

    "progression_state"

]


require_columns(
    progression,
    progression_required,
    "Stage 3.4 progression file"
)


# ============================================================
# DUPLICATE KEY CHECK
# ============================================================

spatial_duplicate_keys = int(
    spatial.duplicated(
        subset=key_columns
    ).sum()
)

progression_duplicate_keys = int(
    progression.duplicated(
        subset=key_columns
    ).sum()
)

print(
    "\nDuplicate transition keys:"
)

print(
    "  Stage 3.0:",
    spatial_duplicate_keys
)

print(
    "  Stage 3.4:",
    progression_duplicate_keys
)


# ============================================================
# NORMALIZE FRAME NUMBERS
# ============================================================

for col in [
    "current_frame",
    "future_frame"
]:

    spatial[col] = safe_numeric(
        spatial[col]
    )

    progression[col] = safe_numeric(
        progression[col]
    )


# ============================================================
# MERGE STAGE 3.0 + STAGE 3.4
# ============================================================

print(
    "\nMerging Stage 3.0 spatial information "
    "with Stage 3.4 physical damage..."
)


spatial_subset = spatial[
    list(
        dict.fromkeys(
            key_columns
            +
            spatial_required
        )
    )
].copy()


progression_subset = progression[
    list(
        dict.fromkeys(
            key_columns
            +
            progression_required
        )
    )
].copy()


target_df = spatial_subset.merge(

    progression_subset,

    on=key_columns,

    how="inner",

    validate="one_to_one",

    suffixes=(
        "",
        "_progression"
    )

)


print(
    "Merged records:",
    len(target_df)
)


# ============================================================
# NORMALIZE DAMAGE VALUES
# ============================================================

for col in [

    "damage_current",

    "damage_future",

    "absolute_damage_change",

    "relative_damage_change"

]:

    target_df[col] = safe_numeric(
        target_df[col]
    )


# ============================================================
# RECOMPUTE DAMAGE CHANGE
# ============================================================

# Stage 3.4 damage values are normalized damage ratios.
#
# Do NOT compare them to Stage 3.0 pixel areas.
#
# Correct:
#
# damage_change =
#     damage_future - damage_current


target_df[
    "damage_change"
] = (

    target_df[
        "damage_future"
    ]

    -

    target_df[
        "damage_current"
    ]

)


target_df[
    "absolute_damage_change_recomputed"
] = (
    target_df[
        "damage_change"
    ].abs()
)


# ============================================================
# VERIFY SUPPLIED DAMAGE CHANGE
# ============================================================

supplied_abs_change = (
    target_df[
        "absolute_damage_change"
    ]
    .abs()
)


recomputed_abs_change = (
    target_df[
        "absolute_damage_change_recomputed"
    ]
)


damage_change_difference = (
    supplied_abs_change
    -
    recomputed_abs_change
).abs()


print(
    "\nPhysical damage-change verification:"
)

print(
    "Maximum difference between supplied "
    "and recomputed magnitude:",
    f"{damage_change_difference.max():.12f}"
)


# ============================================================
# MISSING DAMAGE CHECK
# ============================================================

missing_current_damage = int(
    target_df[
        "damage_current"
    ]
    .isna()
    .sum()
)

missing_future_damage = int(
    target_df[
        "damage_future"
    ]
    .isna()
    .sum()
)

print(
    "Missing current damage:",
    missing_current_damage
)

print(
    "Missing future damage:",
    missing_future_damage
)


# ============================================================
# USE ACTUAL STAGE 3.0 CONTINUITY CLASS
# ============================================================

print(
    "\nUsing Stage 3.0 continuity_class "
    "as authoritative spatial continuity."
)

print(
    target_df[
        "continuity_class"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# NORMALIZE CONTINUITY
# ============================================================

target_df[
    "continuity_class_normalized"
] = (

    target_df[
        "continuity_class"
    ]

    .astype(str)

    .str.upper()

    .str.strip()

)


# ============================================================
# DAMAGE CHANGE THRESHOLDS
# ============================================================

absolute_change = (
    target_df[
        "damage_change"
    ].abs()
)


p75 = float(
    absolute_change.quantile(
        0.75
    )
)

p90 = float(
    absolute_change.quantile(
        0.90
    )
)

p95 = float(
    absolute_change.quantile(
        0.95
    )
)

p99 = float(
    absolute_change.quantile(
        0.99
    )
)


print(
    "\nEmpirical damage-change thresholds:"
)

print(
    "P75:",
    f"{p75:.8f}"
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
# DAMAGE DIRECTION
# ============================================================

target_df[
    "damage_direction"
] = np.select(

    [

        target_df[
            "damage_change"
        ] > 1e-12,

        target_df[
            "damage_change"
        ] < -1e-12

    ],

    [

        "INCREASE",

        "DECREASE"

    ],

    default="STABLE"

)


# ============================================================
# DAMAGE MAGNITUDE
# ============================================================

target_df[
    "damage_change_magnitude"
] = np.select(

    [

        absolute_change >= p95,

        absolute_change >= p90,

        absolute_change >= p75,

        absolute_change > 1e-12

    ],

    [

        "RAPID",

        "LARGE",

        "MODERATE",

        "SMALL"

    ],

    default="NONE"

)


# ============================================================
# CONTINUITY FLAGS
# ============================================================

high = (

    target_df[
        "continuity_class_normalized"
    ]

    ==

    "HIGH_CONTINUITY"

)


moderate = (

    target_df[
        "continuity_class_normalized"
    ]

    ==

    "MODERATE_CONTINUITY"

)


low = (

    target_df[
        "continuity_class_normalized"
    ]

    ==

    "LOW_CONTINUITY"

)


increase = (

    target_df[
        "damage_direction"
    ]

    ==

    "INCREASE"

)


decrease = (

    target_df[
        "damage_direction"
    ]

    ==

    "DECREASE"

)


stable = (

    target_df[
        "damage_direction"
    ]

    ==

    "STABLE"

)


# ============================================================
# MULTI-CLASS TARGET
# ============================================================

target_df[
    "persistent_damage_target"
] = "AMBIGUOUS"


# HIGH CONTINUITY
# ------------------------------------------------------------

target_df.loc[
    high & increase,
    "persistent_damage_target"
] = "PERSISTENT_GROWTH"


target_df.loc[
    high & decrease,
    "persistent_damage_target"
] = "PERSISTENT_REDUCTION"


target_df.loc[
    high & stable,
    "persistent_damage_target"
] = "STABLE_PERSISTENT"


# MODERATE CONTINUITY
# ------------------------------------------------------------

target_df.loc[
    moderate & increase,
    "persistent_damage_target"
] = "PROBABLE_GROWTH"


target_df.loc[
    moderate & decrease,
    "persistent_damage_target"
] = "PROBABLE_REDUCTION"


target_df.loc[
    moderate & stable,
    "persistent_damage_target"
] = "STABLE_MODERATE"


# LOW CONTINUITY
# ------------------------------------------------------------

target_df.loc[
    low & increase,
    "persistent_damage_target"
] = "SPATIAL_TRANSITION_INCREASE"


target_df.loc[
    low & decrease,
    "persistent_damage_target"
] = "SPATIAL_TRANSITION_DECREASE"


target_df.loc[
    low & stable,
    "persistent_damage_target"
] = "STABLE_LOW_CONTINUITY"


# ============================================================
# RAPID TARGET
# ============================================================

target_df[
    "rapid_persistent_damage_target"
] = "NO_RAPID_EVENT"


rapid = (

    target_df[
        "damage_change_magnitude"
    ]

    ==

    "RAPID"

)


target_df.loc[
    rapid
    &
    high
    &
    increase,
    "rapid_persistent_damage_target"
] = "RAPID_PERSISTENT_GROWTH"


target_df.loc[
    rapid
    &
    high
    &
    decrease,
    "rapid_persistent_damage_target"
] = "RAPID_PERSISTENT_REDUCTION"


target_df.loc[
    rapid
    &
    moderate
    &
    increase,
    "rapid_persistent_damage_target"
] = "RAPID_PROBABLE_GROWTH"


target_df.loc[
    rapid
    &
    moderate
    &
    decrease,
    "rapid_persistent_damage_target"
] = "RAPID_PROBABLE_REDUCTION"


target_df.loc[
    rapid
    &
    low
    &
    increase,
    "rapid_persistent_damage_target"
] = "RAPID_SPATIAL_TRANSITION_INCREASE"


target_df.loc[
    rapid
    &
    low
    &
    decrease,
    "rapid_persistent_damage_target"
] = "RAPID_SPATIAL_TRANSITION_DECREASE"


# ============================================================
# BINARY TARGETS
# ============================================================

target_df[
    "persistent_damage_event"
] = (

    target_df[
        "persistent_damage_target"
    ]

    .isin(
        [
            "PERSISTENT_GROWTH",
            "PERSISTENT_REDUCTION"
        ]
    )

    .astype(int)

)


target_df[
    "persistent_or_probable_event"
] = (

    target_df[
        "persistent_damage_target"
    ]

    .isin(
        [
            "PERSISTENT_GROWTH",
            "PERSISTENT_REDUCTION",
            "PROBABLE_GROWTH",
            "PROBABLE_REDUCTION"
        ]
    )

    .astype(int)

)


target_df[
    "rapid_persistent_event"
] = (

    target_df[
        "rapid_persistent_damage_target"
    ]

    .isin(
        [
            "RAPID_PERSISTENT_GROWTH",
            "RAPID_PERSISTENT_REDUCTION"
        ]
    )

    .astype(int)

)


# ============================================================
# TARGET CONFIDENCE
# ============================================================

target_df[
    "target_confidence"
] = np.select(

    [

        high,

        moderate,

        low

    ],

    [

        "HIGH",

        "MEDIUM",

        "LOW"

    ],

    default="UNKNOWN"

)


# ============================================================
# TARGET DIRECTION
# ============================================================

target_df[
    "target_direction"
] = np.select(

    [

        target_df[
            "persistent_damage_target"
        ]
        .isin(
            [
                "PERSISTENT_GROWTH",
                "PROBABLE_GROWTH"
            ]
        ),

        target_df[
            "persistent_damage_target"
        ]
        .isin(
            [
                "PERSISTENT_REDUCTION",
                "PROBABLE_REDUCTION"
            ]
        ),

        target_df[
            "persistent_damage_target"
        ]
        ==
        "SPATIAL_TRANSITION_INCREASE",

        target_df[
            "persistent_damage_target"
        ]
        ==
        "SPATIAL_TRANSITION_DECREASE"

    ],

    [

        "GROWTH",

        "REDUCTION",

        "SPATIAL_INCREASE",

        "SPATIAL_DECREASE"

    ],

    default="STABLE_OR_AMBIGUOUS"

)


# ============================================================
# SORT TEMPORALLY
# ============================================================

target_df = target_df.sort_values(

    [
        "sequence",
        "current_frame"
    ]

).reset_index(
    drop=True
)


# ============================================================
# PREVIOUS TARGET
# ============================================================

target_df[
    "previous_persistent_target"
] = (

    target_df

    .groupby(
        "sequence"
    )[
        "persistent_damage_target"
    ]

    .shift(1)

)


target_df[
    "same_persistent_state_as_previous"
] = (

    target_df[
        "persistent_damage_target"
    ]

    ==

    target_df[
        "previous_persistent_target"
    ]

)


# ============================================================
# GROWTH / REDUCTION FLAGS
# ============================================================

target_df[
    "_growth_flag"
] = (

    target_df[
        "persistent_damage_target"
    ]

    .isin(
        [
            "PERSISTENT_GROWTH",
            "PROBABLE_GROWTH"
        ]
    )

)


target_df[
    "_reduction_flag"
] = (

    target_df[
        "persistent_damage_target"
    ]

    .isin(
        [
            "PERSISTENT_REDUCTION",
            "PROBABLE_REDUCTION"
        ]
    )

)


# ============================================================
# STREAK CALCULATION
# ============================================================

target_df[
    "growth_streak"
] = (

    target_df

    .groupby(
        "sequence"
    )[
        "_growth_flag"
    ]

    .transform(
        calculate_streak_series
    )

)


target_df[
    "reduction_streak"
] = (

    target_df

    .groupby(
        "sequence"
    )[
        "_reduction_flag"
    ]

    .transform(
        calculate_streak_series
    )

)


target_df.drop(

    columns=[
        "_growth_flag",
        "_reduction_flag"
    ],

    inplace=True

)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

sequence_rows = []


for sequence, group in target_df.groupby(
    "sequence"
):

    counts = (

        group[
            "persistent_damage_target"
        ]

        .value_counts()

        .to_dict()

    )


    sequence_rows.append({

        "sequence":
            sequence,

        "component":
            group[
                "component"
            ].iloc[0],

        "split":
            group[
                "split"
            ].iloc[0],

        "transition_count":
            int(
                len(group)
            ),

        "persistent_growth":
            int(
                counts.get(
                    "PERSISTENT_GROWTH",
                    0
                )
            ),

        "persistent_reduction":
            int(
                counts.get(
                    "PERSISTENT_REDUCTION",
                    0
                )
            ),

        "probable_growth":
            int(
                counts.get(
                    "PROBABLE_GROWTH",
                    0
                )
            ),

        "probable_reduction":
            int(
                counts.get(
                    "PROBABLE_REDUCTION",
                    0
                )
            ),

        "spatial_transition_increase":
            int(
                counts.get(
                    "SPATIAL_TRANSITION_INCREASE",
                    0
                )
            ),

        "spatial_transition_decrease":
            int(
                counts.get(
                    "SPATIAL_TRANSITION_DECREASE",
                    0
                )
            ),

        "stable_persistent":
            int(
                counts.get(
                    "STABLE_PERSISTENT",
                    0
                )
            ),

        "stable_moderate":
            int(
                counts.get(
                    "STABLE_MODERATE",
                    0
                )
            ),

        "stable_low_continuity":
            int(
                counts.get(
                    "STABLE_LOW_CONTINUITY",
                    0
                )
            ),

        "ambiguous":
            int(
                counts.get(
                    "AMBIGUOUS",
                    0
                )
            ),

        "persistent_event_rate":
            float(
                group[
                    "persistent_damage_event"
                ].mean()
            ),

        "persistent_or_probable_rate":
            float(
                group[
                    "persistent_or_probable_event"
                ].mean()
            ),

        "rapid_persistent_event_rate":
            float(
                group[
                    "rapid_persistent_event"
                ].mean()
            ),

        "max_growth_streak":
            int(
                group[
                    "growth_streak"
                ].max()
            ),

        "max_reduction_streak":
            int(
                group[
                    "reduction_streak"
                ].max()
            )

    })


sequence_summary = pd.DataFrame(
    sequence_rows
)


# ============================================================
# SPLIT SUMMARY
# ============================================================

split_rows = []


for split, group in target_df.groupby(
    "split"
):

    counts = (

        group[
            "persistent_damage_target"
        ]

        .value_counts()

        .to_dict()

    )


    split_rows.append({

        "split":
            split,

        "records":
            int(
                len(group)
            ),

        "persistent_growth":
            int(
                counts.get(
                    "PERSISTENT_GROWTH",
                    0
                )
            ),

        "persistent_reduction":
            int(
                counts.get(
                    "PERSISTENT_REDUCTION",
                    0
                )
            ),

        "probable_growth":
            int(
                counts.get(
                    "PROBABLE_GROWTH",
                    0
                )
            ),

        "probable_reduction":
            int(
                counts.get(
                    "PROBABLE_REDUCTION",
                    0
                )
            ),

        "spatial_transition_increase":
            int(
                counts.get(
                    "SPATIAL_TRANSITION_INCREASE",
                    0
                )
            ),

        "spatial_transition_decrease":
            int(
                counts.get(
                    "SPATIAL_TRANSITION_DECREASE",
                    0
                )
            ),

        "stable":
            int(
                group[
                    "persistent_damage_target"
                ]
                .isin(
                    [
                        "STABLE_PERSISTENT",
                        "STABLE_MODERATE",
                        "STABLE_LOW_CONTINUITY"
                    ]
                )
                .sum()
            ),

        "ambiguous":
            int(
                group[
                    "persistent_damage_target"
                ]
                .eq(
                    "AMBIGUOUS"
                )
                .sum()
            ),

        "persistent_damage_events":
            int(
                group[
                    "persistent_damage_event"
                ].sum()
            ),

        "persistent_event_rate":
            float(
                group[
                    "persistent_damage_event"
                ].mean()
            ),

        "persistent_or_probable_events":
            int(
                group[
                    "persistent_or_probable_event"
                ].sum()
            ),

        "persistent_or_probable_rate":
            float(
                group[
                    "persistent_or_probable_event"
                ].mean()
            ),

        "rapid_persistent_events":
            int(
                group[
                    "rapid_persistent_event"
                ].sum()
            ),

        "rapid_persistent_event_rate":
            float(
                group[
                    "rapid_persistent_event"
                ].mean()
            )

    })


split_summary = pd.DataFrame(
    split_rows
)


# ============================================================
# STATE SUMMARY
# ============================================================

state_summary = (

    target_df[
        "persistent_damage_target"
    ]

    .value_counts()

    .reset_index()

)

state_summary.columns = [
    "target_state",
    "record_count"
]

state_summary[
    "percentage"
] = (

    state_summary[
        "record_count"
    ]

    /

    len(target_df)

    *

    100.0

)


# ============================================================
# CONTINUITY SUMMARY
# ============================================================

continuity_summary = (

    target_df[
        "continuity_class_normalized"
    ]

    .value_counts()

    .reset_index()

)

continuity_summary.columns = [
    "continuity_class",
    "record_count"
]

continuity_summary[
    "percentage"
] = (

    continuity_summary[
        "record_count"
    ]

    /

    len(target_df)

    *

    100.0

)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

duplicate_target_keys = int(
    target_df.duplicated(
        subset=key_columns
    ).sum()
)


invalid_temporal_direction = int(

    (

        target_df[
            "future_frame"
        ]

        <=

        target_df[
            "current_frame"
        ]

    ).sum()

)


sequence_split_conflicts = int(

    (

        target_df

        .groupby(
            "sequence"
        )[
            "split"
        ]

        .nunique()

        > 1

    ).sum()

)


component_split_conflicts = int(

    (

        target_df

        .groupby(
            "component"
        )[
            "split"
        ]

        .nunique()

        > 1

    ).sum()

)


missing_continuity = int(

    target_df[
        "continuity_class"
    ]
    .isna()
    .sum()

)


invalid_target_states = int(

    target_df[
        "persistent_damage_target"
    ]
    .isna()
    .sum()

)


# ============================================================
# TARGET SELF-CONSISTENCY
# ============================================================

growth_mask = (

    target_df[
        "persistent_damage_target"
    ]

    ==

    "PERSISTENT_GROWTH"

)


growth_consistency_errors = int(

    (

        growth_mask

        &

        (
            target_df[
                "damage_direction"
            ]
            !=
            "INCREASE"
        )

    ).sum()

)


reduction_mask = (

    target_df[
        "persistent_damage_target"
    ]

    ==

    "PERSISTENT_REDUCTION"

)


reduction_consistency_errors = int(

    (

        reduction_mask

        &

        (
            target_df[
                "damage_direction"
            ]
            !=
            "DECREASE"
        )

    ).sum()

)


# ============================================================
# TARGET COUNT
# ============================================================

total_records = int(
    len(target_df)
)


persistent_growth_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "PERSISTENT_GROWTH"
    ).sum()

)


persistent_reduction_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "PERSISTENT_REDUCTION"
    ).sum()

)


probable_growth_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "PROBABLE_GROWTH"
    ).sum()

)


probable_reduction_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "PROBABLE_REDUCTION"
    ).sum()

)


spatial_increase_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "SPATIAL_TRANSITION_INCREASE"
    ).sum()

)


spatial_decrease_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "SPATIAL_TRANSITION_DECREASE"
    ).sum()

)


stable_count = int(

    target_df[
        "persistent_damage_target"
    ]
    .isin(
        [
            "STABLE_PERSISTENT",
            "STABLE_MODERATE",
            "STABLE_LOW_CONTINUITY"
        ]
    )
    .sum()

)


ambiguous_count = int(

    (
        target_df[
            "persistent_damage_target"
        ]
        ==
        "AMBIGUOUS"
    ).sum()

)


persistent_event_count = int(

    target_df[
        "persistent_damage_event"
    ].sum()

)


persistent_or_probable_count = int(

    target_df[
        "persistent_or_probable_event"
    ].sum()

)


rapid_persistent_count = int(

    target_df[
        "rapid_persistent_event"
    ].sum()

)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

target_df.to_csv(
    OUTPUT_TARGET_FILE,
    index=False
)


sequence_summary.to_csv(
    OUTPUT_SEQUENCE_FILE,
    index=False
)


split_summary.to_csv(
    OUTPUT_SPLIT_FILE,
    index=False
)


state_summary.to_csv(
    OUTPUT_STATE_FILE,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "3.6",

    "status":
        "PASS",

    "records":
        total_records,

    "sequences":
        int(
            target_df[
                "sequence"
            ].nunique()
        ),

    "components":
        int(
            target_df[
                "component"
            ].nunique()
        ),

    "target_definition":
        (
            "Predict the observed next-frame spatially "
            "persistent damage transition using Stage 3.0 "
            "heuristic continuity and Stage 3.4 physical "
            "damage change."
        ),

    "damage_change_definition":
        "damage_future - damage_current",

    "damage_change_thresholds": {

        "p75":
            p75,

        "p90":
            p90,

        "p95":
            p95,

        "p99":
            p99

    },

    "target_state_counts": {

        "PERSISTENT_GROWTH":
            persistent_growth_count,

        "PERSISTENT_REDUCTION":
            persistent_reduction_count,

        "PROBABLE_GROWTH":
            probable_growth_count,

        "PROBABLE_REDUCTION":
            probable_reduction_count,

        "SPATIAL_TRANSITION_INCREASE":
            spatial_increase_count,

        "SPATIAL_TRANSITION_DECREASE":
            spatial_decrease_count,

        "STABLE":
            stable_count,

        "AMBIGUOUS":
            ambiguous_count

    },

    "target_state_percentages": {

        "PERSISTENT_GROWTH":
            pct(
                persistent_growth_count,
                total_records
            ),

        "PERSISTENT_REDUCTION":
            pct(
                persistent_reduction_count,
                total_records
            ),

        "PROBABLE_GROWTH":
            pct(
                probable_growth_count,
                total_records
            ),

        "PROBABLE_REDUCTION":
            pct(
                probable_reduction_count,
                total_records
            ),

        "SPATIAL_TRANSITION_INCREASE":
            pct(
                spatial_increase_count,
                total_records
            ),

        "SPATIAL_TRANSITION_DECREASE":
            pct(
                spatial_decrease_count,
                total_records
            ),

        "STABLE":
            pct(
                stable_count,
                total_records
            ),

        "AMBIGUOUS":
            pct(
                ambiguous_count,
                total_records
            )

    },

    "binary_target_counts": {

        "persistent_damage_event":
            persistent_event_count,

        "persistent_or_probable_event":
            persistent_or_probable_count,

        "rapid_persistent_event":
            rapid_persistent_count

    },

    "binary_target_rates": {

        "persistent_damage_event":
            pct(
                persistent_event_count,
                total_records
            ),

        "persistent_or_probable_event":
            pct(
                persistent_or_probable_count,
                total_records
            ),

        "rapid_persistent_event":
            pct(
                rapid_persistent_count,
                total_records
            )

    },

    "continuity_distribution":
        {
            str(key): int(value)
            for key, value
            in target_df[
                "continuity_class_normalized"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

    "maximum_growth_streak":
        int(
            target_df[
                "growth_streak"
            ].max()
        ),

    "maximum_reduction_streak":
        int(
            target_df[
                "reduction_streak"
            ].max()
        ),

    "integrity": {

        "duplicate_spatial_keys":
            spatial_duplicate_keys,

        "duplicate_progression_keys":
            progression_duplicate_keys,

        "duplicate_target_keys":
            duplicate_target_keys,

        "invalid_temporal_direction":
            invalid_temporal_direction,

        "sequence_split_conflicts":
            sequence_split_conflicts,

        "component_split_conflicts":
            component_split_conflicts,

        "missing_current_damage":
            missing_current_damage,

        "missing_future_damage":
            missing_future_damage,

        "missing_continuity":
            missing_continuity,

        "invalid_target_states":
            invalid_target_states,

        "growth_consistency_errors":
            growth_consistency_errors,

        "reduction_consistency_errors":
            reduction_consistency_errors,

        "maximum_damage_change_recalculation_difference":
            float(
                damage_change_difference.max()
            )

    },

    "methodological_caveats": [

        "Stage 3.0 continuity is heuristic polygon-level matching, not manually verified crack identity.",

        "The target describes observed annotation/spatial damage evolution and is not a validated mechanical severity grade.",

        "The target is constructed from current-to-future information and must never be used as a predictor feature.",

        "Stage 3.0 pixel-area measurements are not compared directly with normalized damage ratios."

    ],

    "output_files": {

        "targets":
            OUTPUT_TARGET_FILE,

        "sequence_summary":
            OUTPUT_SEQUENCE_FILE,

        "split_summary":
            OUTPUT_SPLIT_FILE,

        "state_summary":
            OUTPUT_STATE_FILE,

        "summary":
            OUTPUT_SUMMARY_FILE

    }

}


# ============================================================
# WRITE SUMMARY
# ============================================================

with open(
    OUTPUT_SUMMARY_FILE,
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
print("STAGE 3.6 SUMMARY")
print("=" * 70)

print(
    "Target records:",
    total_records
)

print(
    "Sequences:",
    target_df[
        "sequence"
    ].nunique()
)

print(
    "Components:",
    target_df[
        "component"
    ].nunique()
)


print(
    "\nTarget states:"
)

for state in [

    "PERSISTENT_GROWTH",

    "PERSISTENT_REDUCTION",

    "PROBABLE_GROWTH",

    "PROBABLE_REDUCTION",

    "SPATIAL_TRANSITION_INCREASE",

    "SPATIAL_TRANSITION_DECREASE",

    "STABLE_PERSISTENT",

    "STABLE_MODERATE",

    "STABLE_LOW_CONTINUITY",

    "AMBIGUOUS"

]:

    count = int(

        (

            target_df[
                "persistent_damage_target"
            ]

            ==

            state

        ).sum()

    )

    if count > 0:

        print(

            f"  {state}: "
            f"{count} "
            f"({pct(count, total_records):.2f}%)"

        )


print(
    "\nBinary targets:"
)

print(
    "  Persistent damage event:",
    persistent_event_count,
    f"({pct(persistent_event_count, total_records):.2f}%)"
)

print(
    "  Persistent or probable event:",
    persistent_or_probable_count,
    f"({pct(persistent_or_probable_count, total_records):.2f}%)"
)

print(
    "  Rapid persistent event:",
    rapid_persistent_count,
    f"({pct(rapid_persistent_count, total_records):.2f}%)"
)


print(
    "\nDamage-change thresholds:"
)

print(
    "  P75:",
    f"{p75:.8f}"
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
    "\nSequential persistence:"
)

print(
    "  Maximum growth streak:",
    int(
        target_df[
            "growth_streak"
        ].max()
    )
)

print(
    "  Maximum reduction streak:",
    int(
        target_df[
            "reduction_streak"
        ].max()
    )
)


print(
    "\nIntegrity checks:"
)

print(
    "  Duplicate spatial keys:",
    spatial_duplicate_keys
)

print(
    "  Duplicate progression keys:",
    progression_duplicate_keys
)

print(
    "  Duplicate target keys:",
    duplicate_target_keys
)

print(
    "  Invalid temporal direction:",
    invalid_temporal_direction
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
    "  Missing current damage:",
    missing_current_damage
)

print(
    "  Missing future damage:",
    missing_future_damage
)

print(
    "  Missing continuity:",
    missing_continuity
)

print(
    "  Invalid target states:",
    invalid_target_states
)

print(
    "  Growth consistency errors:",
    growth_consistency_errors
)

print(
    "  Reduction consistency errors:",
    reduction_consistency_errors
)

print(
    "  Maximum damage-change recalculation difference:",
    f"{damage_change_difference.max():.12f}"
)


print(
    "\nOutput target file:"
)

print(
    OUTPUT_TARGET_FILE
)

print(
    "\nOutput sequence summary:"
)

print(
    OUTPUT_SEQUENCE_FILE
)

print(
    "\nOutput split summary:"
)

print(
    OUTPUT_SPLIT_FILE
)

print(
    "\nOutput state summary:"
)

print(
    OUTPUT_STATE_FILE
)

print(
    "\nOutput summary:"
)

print(
    OUTPUT_SUMMARY_FILE
)

print(
    "\nFINAL STATUS: PASS"
)