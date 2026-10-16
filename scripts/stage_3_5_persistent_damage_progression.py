import os
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

PROGRESSION_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_4_physical_damage_progression.csv"
)

SPATIAL_TRACK_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

OUTPUT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_5_persistent_damage_progression.csv"
)

SEQUENCE_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_5_sequence_persistent_summary.csv"
)

STATE_SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_5_persistent_state_summary.csv"
)

SUMMARY_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_5_persistent_damage_summary.json"
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


def first_existing(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def safe_corr(df, x, y):

    if x not in df.columns or y not in df.columns:
        return None

    temp = df[
        [x, y]
    ].dropna()

    if len(temp) < 3:
        return None

    value = temp[x].corr(
        temp[y]
    )

    if pd.isna(value):
        return None

    return float(value)


def percentage(value, total):

    if total == 0:
        return 0.0

    return float(
        value / total * 100.0
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 3.5 - PERSISTENT PHYSICAL DAMAGE PROGRESSION ANALYSIS")
print("=" * 70)


# ============================================================
# FILE CHECKS
# ============================================================

print(
    "\nChecking Stage 3.4 progression file..."
)

if not os.path.exists(
    PROGRESSION_FILE
):

    raise FileNotFoundError(
        PROGRESSION_FILE
    )

print(
    "Stage 3.4 progression file: FOUND"
)


print(
    "\nChecking Stage 3.0 spatial track file..."
)

if not os.path.exists(
    SPATIAL_TRACK_FILE
):

    raise FileNotFoundError(
        SPATIAL_TRACK_FILE
    )

print(
    "Stage 3.0 spatial track file: FOUND"
)


# ============================================================
# LOAD
# ============================================================

print(
    "\nLoading Stage 3.4 progression..."
)

progression = pd.read_csv(
    PROGRESSION_FILE
)

print(
    "Progression records:",
    len(progression)
)


print(
    "\nLoading Stage 3.0 spatial tracks..."
)

spatial = pd.read_csv(
    SPATIAL_TRACK_FILE
)

print(
    "Spatial track records:",
    len(spatial)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_progression = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame",
    "damage_current",
    "damage_future",
    "absolute_damage_change",
    "relative_damage_change",
    "crack_count_change",
    "largest_crack_change",
    "progression_state"
]

require_columns(
    progression,
    required_progression,
    "Stage 3.4 progression"
)


# ============================================================
# SPATIAL COLUMN DISCOVERY
# ============================================================

print(
    "\nDetecting spatial tracking columns..."
)

persistent_candidates = [
    "persistent_fraction",
    "matched_fraction_current"
]

future_coverage_candidates = [
    "future_coverage",
    "matched_fraction_future"
]

iou_candidates = [
    "mean_match_iou"
]

center_candidates = [
    "mean_center_distance"
]

area_similarity_candidates = [
    "mean_area_similarity"
]

match_score_candidates = [
    "mean_match_score"
]

continuity_candidates = [
    "track_continuity_quality",
    "continuity_score"
]

matched_count_candidates = [
    "matched_crack_count"
]

current_count_candidates = [
    "current_crack_count"
]

future_count_candidates = [
    "future_crack_count"
]


persistent_col = first_existing(
    progression,
    persistent_candidates
)

future_coverage_col = first_existing(
    progression,
    future_coverage_candidates
)

iou_col = first_existing(
    progression,
    iou_candidates
)

center_col = first_existing(
    progression,
    center_candidates
)

area_similarity_col = first_existing(
    progression,
    area_similarity_candidates
)

match_score_col = first_existing(
    progression,
    match_score_candidates
)

continuity_col = first_existing(
    progression,
    continuity_candidates
)

matched_count_col = first_existing(
    progression,
    matched_count_candidates
)

current_count_col = first_existing(
    progression,
    current_count_candidates
)

future_count_col = first_existing(
    progression,
    future_count_candidates
)


print(
    "Persistent fraction column:",
    persistent_col
)

print(
    "Future coverage column:",
    future_coverage_col
)

print(
    "Match IoU column:",
    iou_col
)

print(
    "Center distance column:",
    center_col
)

print(
    "Area similarity column:",
    area_similarity_col
)

print(
    "Match score column:",
    match_score_col
)

print(
    "Continuity column:",
    continuity_col
)


# ============================================================
# FALLBACK: BUILD SPATIAL METRICS FROM STAGE 3.0
# ============================================================

if (
    persistent_col is None
    or
    iou_col is None
    or
    center_col is None
    or
    area_similarity_col is None
):

    print(
        "\nSome spatial metrics are missing from Stage 3.4."
    )

    print(
        "Attempting reconstruction from Stage 3.0..."
    )

    spatial_key = [
        "sequence",
        "component",
        "split",
        "current_frame",
        "future_frame"
    ]

    require_columns(
        spatial,
        spatial_key,
        "Stage 3.0 spatial tracks"
    )

    reconstruction_candidates = {

        "persistent_fraction": [
            "persistent_fraction",
            "matched_fraction_current"
        ],

        "future_coverage": [
            "future_coverage",
            "matched_fraction_future"
        ],

        "mean_match_iou": [
            "mean_match_iou"
        ],

        "mean_center_distance": [
            "mean_center_distance"
        ],

        "mean_area_similarity": [
            "mean_area_similarity"
        ],

        "mean_match_score": [
            "mean_match_score"
        ],

        "track_continuity_quality": [
            "track_continuity_quality",
            "continuity_score"
        ]
    }

    spatial_small = spatial[
        spatial_key
    ].copy()

    for target_name, candidates in (
        reconstruction_candidates.items()
    ):

        source = first_existing(
            spatial,
            candidates
        )

        if source is not None:

            spatial_small[
                target_name
            ] = spatial[
                source
            ].values

    progression = progression.merge(
        spatial_small,
        on=spatial_key,
        how="left",
        suffixes=(
            "",
            "_reconstructed"
        )
    )

    # Prefer reconstructed values only where original values
    # were unavailable.

    for col in [
        "persistent_fraction",
        "future_coverage",
        "mean_match_iou",
        "mean_center_distance",
        "mean_area_similarity",
        "mean_match_score",
        "track_continuity_quality"
    ]:

        reconstructed = (
            col + "_reconstructed"
        )

        if reconstructed in progression.columns:

            if col not in progression.columns:

                progression[col] = (
                    progression[
                        reconstructed
                    ]
                )

            else:

                progression[col] = (
                    progression[col]
                    .fillna(
                        progression[
                            reconstructed
                        ]
                    )
                )

            progression.drop(
                columns=[reconstructed],
                inplace=True
            )


# ============================================================
# NORMALIZE SPATIAL COLUMN NAMES
# ============================================================

if (
    "persistent_fraction"
    not in progression.columns
):

    progression[
        "persistent_fraction"
    ] = np.nan


if (
    "future_coverage"
    not in progression.columns
):

    progression[
        "future_coverage"
    ] = np.nan


if (
    "mean_match_iou"
    not in progression.columns
):

    progression[
        "mean_match_iou"
    ] = np.nan


if (
    "mean_center_distance"
    not in progression.columns
):

    progression[
        "mean_center_distance"
    ] = np.nan


if (
    "mean_area_similarity"
    not in progression.columns
):

    progression[
        "mean_area_similarity"
    ] = np.nan


if (
    "mean_match_score"
    not in progression.columns
):

    progression[
        "mean_match_score"
    ] = np.nan


if (
    "track_continuity_quality"
    not in progression.columns
):

    progression[
        "track_continuity_quality"
    ] = np.nan


# ============================================================
# NORMALIZE FRAMES
# ============================================================

progression[
    "current_frame"
] = pd.to_numeric(
    progression[
        "current_frame"
    ],
    errors="coerce"
)

progression[
    "future_frame"
] = pd.to_numeric(
    progression[
        "future_frame"
    ],
    errors="coerce"
)

progression = progression.dropna(
    subset=[
        "current_frame",
        "future_frame"
    ]
).copy()

progression[
    "current_frame"
] = (
    progression[
        "current_frame"
    ].astype(int)
)

progression[
    "future_frame"
] = (
    progression[
        "future_frame"
    ].astype(int)
)


# ============================================================
# SPATIAL CONTINUITY QUALITY
# ============================================================

# We intentionally keep this composite conservative.
#
# A transition is considered spatially persistent only when
# multiple spatial indicators support continuity.

persistent_fraction = pd.to_numeric(
    progression[
        "persistent_fraction"
    ],
    errors="coerce"
)

future_coverage = pd.to_numeric(
    progression[
        "future_coverage"
    ],
    errors="coerce"
)

match_iou = pd.to_numeric(
    progression[
        "mean_match_iou"
    ],
    errors="coerce"
)

center_distance = pd.to_numeric(
    progression[
        "mean_center_distance"
    ],
    errors="coerce"
)

area_similarity = pd.to_numeric(
    progression[
        "mean_area_similarity"
    ],
    errors="coerce"
)

match_score = pd.to_numeric(
    progression[
        "mean_match_score"
    ],
    errors="coerce"
)


# ============================================================
# INDIVIDUAL CONTINUITY SIGNALS
# ============================================================

progression[
    "persistent_fraction_signal"
] = (
    persistent_fraction >= 0.75
)


progression[
    "future_coverage_signal"
] = (
    future_coverage >= 0.75
)


progression[
    "iou_signal"
] = (
    match_iou >= 0.30
)


progression[
    "center_signal"
] = (
    center_distance <= 0.10
)


progression[
    "area_similarity_signal"
] = (
    area_similarity >= 0.70
)


progression[
    "match_score_signal"
] = (
    match_score >= 0.50
)


# ============================================================
# CONTINUITY EVIDENCE SCORE
# ============================================================

signal_columns = [
    "persistent_fraction_signal",
    "future_coverage_signal",
    "iou_signal",
    "center_signal",
    "area_similarity_signal",
    "match_score_signal"
]

progression[
    "spatial_evidence_score"
] = (
    progression[
        signal_columns
    ]
    .astype(int)
    .sum(axis=1)
)


# ============================================================
# SPATIAL EVIDENCE CLASS
# ============================================================

progression[
    "spatial_evidence_class"
] = np.select(
    [
        progression[
            "spatial_evidence_score"
        ] >= 5,

        progression[
            "spatial_evidence_score"
        ] >= 3,

        progression[
            "spatial_evidence_score"
        ] >= 1
    ],
    [
        "STRONG",

        "MODERATE",

        "WEAK"
    ],
    default="NONE"
)


# ============================================================
# DAMAGE CHANGE MAGNITUDE
# ============================================================

abs_change = (
    progression[
        "absolute_damage_change"
    ].abs()
)

p75 = float(
    np.percentile(
        abs_change.dropna(),
        75
    )
)

p90 = float(
    np.percentile(
        abs_change.dropna(),
        90
    )
)

p95 = float(
    np.percentile(
        abs_change.dropna(),
        95
    )

)


print(
    "\nDamage-change thresholds:"
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


# ============================================================
# CHANGE MAGNITUDE CLASS
# ============================================================

progression[
    "change_magnitude_class"
] = np.select(
    [
        abs_change >= p95,

        abs_change >= p90,

        abs_change >= p75,

        abs_change > 1e-12
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
# PERSISTENT DAMAGE PROGRESSION
# ============================================================

increase = (
    progression[
        "absolute_damage_change"
    ]
    > 1e-12
)

decrease = (
    progression[
        "absolute_damage_change"
    ]
    < -1e-12
)

stable = ~increase & ~decrease

strong_spatial = (
    progression[
        "spatial_evidence_class"
    ]
    ==
    "STRONG"
)

moderate_spatial = (
    progression[
        "spatial_evidence_class"
    ]
    ==
    "MODERATE"
)

weak_spatial = (
    progression[
        "spatial_evidence_class"
    ].isin(
        [
            "WEAK",
            "NONE"
        ]
    )
)


# ============================================================
# INITIAL STATE
# ============================================================

progression[
    "persistent_damage_state"
] = "AMBIGUOUS"


# Strong spatial continuity + increasing area
progression.loc[
    increase & strong_spatial,
    "persistent_damage_state"
] = "PERSISTENT_GROWTH"


# Strong spatial continuity + decreasing area
progression.loc[
    decrease & strong_spatial,
    "persistent_damage_state"
] = "PERSISTENT_REDUCTION"


# Stable with strong spatial continuity
progression.loc[
    stable & strong_spatial,
    "persistent_damage_state"
] = "STABLE_PERSISTENT_DAMAGE"


# Moderate spatial evidence
progression.loc[
    increase & moderate_spatial,
    "persistent_damage_state"
] = "PROBABLE_GROWTH"


progression.loc[
    decrease & moderate_spatial,
    "persistent_damage_state"
] = "PROBABLE_REDUCTION"


# Weak spatial evidence
progression.loc[
    increase & weak_spatial,
    "persistent_damage_state"
] = "SPATIAL_TRANSITION_INCREASE"


progression.loc[
    decrease & weak_spatial,
    "persistent_damage_state"
] = "SPATIAL_TRANSITION_DECREASE"


progression.loc[
    stable & moderate_spatial,
    "persistent_damage_state"
] = "STABLE_MODERATE_CONTINUITY"


progression.loc[
    stable & weak_spatial,
    "persistent_damage_state"
] = "STABLE_LOW_CONTINUITY"


# ============================================================
# RAPID EVENT SUBCLASSIFICATION
# ============================================================

rapid = (
    progression[
        "change_magnitude_class"
    ]
    ==
    "RAPID"
)


progression[
    "rapid_event_type"
] = "NONE"


progression.loc[
    rapid
    & increase
    & strong_spatial,
    "rapid_event_type"
] = "RAPID_PERSISTENT_GROWTH"


progression.loc[
    rapid
    & decrease
    & strong_spatial,
    "rapid_event_type"
] = "RAPID_PERSISTENT_REDUCTION"


progression.loc[
    rapid
    & increase
    & moderate_spatial,
    "rapid_event_type"
] = "RAPID_PROBABLE_GROWTH"


progression.loc[
    rapid
    & decrease
    & moderate_spatial,
    "rapid_event_type"
] = "RAPID_PROBABLE_REDUCTION"


progression.loc[
    rapid
    & increase
    & weak_spatial,
    "rapid_event_type"
] = "RAPID_SPATIAL_TRANSITION_INCREASE"


progression.loc[
    rapid
    & decrease
    & weak_spatial,
    "rapid_event_type"
] = "RAPID_SPATIAL_TRANSITION_DECREASE"


# ============================================================
# PHYSICAL PROGRESSION CONFIDENCE
# ============================================================

progression[
    "progression_confidence"
] = np.select(
    [
        strong_spatial
        & (
            abs_change >= p95
        ),

        strong_spatial,

        moderate_spatial,

        weak_spatial
    ],
    [
        "HIGH",

        "HIGH",

        "MEDIUM",

        "LOW"
    ],
    default="LOW"
)


# ============================================================
# PERSISTENT GROWTH FLAG
# ============================================================

progression[
    "persistent_growth_flag"
] = (
    progression[
        "persistent_damage_state"
    ]
    .isin(
        [
            "PERSISTENT_GROWTH",
            "PROBABLE_GROWTH"
        ]
    )
)


progression[
    "persistent_reduction_flag"
] = (
    progression[
        "persistent_damage_state"
    ]
    .isin(
        [
            "PERSISTENT_REDUCTION",
            "PROBABLE_REDUCTION"
        ]
    )
)


# ============================================================
# SEQUENTIAL PERSISTENCE
# ============================================================

progression = progression.sort_values(
    [
        "sequence",
        "current_frame"
    ]
).reset_index(
    drop=True
)


group = progression.groupby(
    "sequence"
)


progression[
    "previous_persistent_state"
] = group[
    "persistent_damage_state"
].shift(1)


progression[
    "previous_damage_change"
] = group[
    "absolute_damage_change"
].shift(1)


progression[
    "growth_followed_growth"
] = (
    progression[
        "persistent_growth_flag"
    ]
    &
    group[
        "persistent_growth_flag"
    ].shift(1).fillna(False)
)


progression[
    "reduction_followed_reduction"
] = (
    progression[
        "persistent_reduction_flag"
    ]
    &
    group[
        "persistent_reduction_flag"
    ].shift(1).fillna(False)
)


# ============================================================
# RUN LENGTH OF PERSISTENT GROWTH
# ============================================================

progression[
    "growth_streak"
] = 0

progression[
    "reduction_streak"
] = 0


def positive_streak(values):

    result = []

    streak = 0

    for value in values:

        if bool(value):

            streak += 1

        else:

            streak = 0

        result.append(streak)

    return result


progression[
    "growth_streak"
] = group[
    "persistent_growth_flag"
].transform(
    positive_streak
)


progression[
    "reduction_streak"
] = group[
    "persistent_reduction_flag"
].transform(
    positive_streak
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

sequence_rows = []

for sequence, group_df in progression.groupby(
    "sequence"
):

    total = len(group_df)

    state_counts = (
        group_df[
            "persistent_damage_state"
        ]
        .value_counts()
        .to_dict()
    )

    rapid_counts = (
        group_df[
            "rapid_event_type"
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
            int(total),

        "persistent_growth_count":
            int(
                state_counts.get(
                    "PERSISTENT_GROWTH",
                    0
                )
            ),

        "probable_growth_count":
            int(
                state_counts.get(
                    "PROBABLE_GROWTH",
                    0
                )
            ),

        "persistent_reduction_count":
            int(
                state_counts.get(
                    "PERSISTENT_REDUCTION",
                    0
                )
            ),

        "probable_reduction_count":
            int(
                state_counts.get(
                    "PROBABLE_REDUCTION",
                    0
                )
            ),

        "spatial_transition_increase_count":
            int(
                state_counts.get(
                    "SPATIAL_TRANSITION_INCREASE",
                    0
                )
            ),

        "spatial_transition_decrease_count":
            int(
                state_counts.get(
                    "SPATIAL_TRANSITION_DECREASE",
                    0
                )
            ),

        "ambiguous_count":
            int(
                state_counts.get(
                    "AMBIGUOUS",
                    0
                )
            ),

        "rapid_persistent_growth":
            int(
                rapid_counts.get(
                    "RAPID_PERSISTENT_GROWTH",
                    0
                )
            ),

        "rapid_persistent_reduction":
            int(
                rapid_counts.get(
                    "RAPID_PERSISTENT_REDUCTION",
                    0
                )
            ),

        "rapid_spatial_transition":
            int(
                rapid_counts.get(
                    "RAPID_SPATIAL_TRANSITION_INCREASE",
                    0
                )
                +
                rapid_counts.get(
                    "RAPID_SPATIAL_TRANSITION_DECREASE",
                    0
                )
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

        "mean_persistent_fraction":
            float(
                group_df[
                    "persistent_fraction"
                ].mean()
            ),

        "mean_match_iou":
            float(
                group_df[
                    "mean_match_iou"
                ].mean()
            ),

        "mean_center_distance":
            float(
                group_df[
                    "mean_center_distance"
                ].mean()
            ),

        "mean_area_similarity":
            float(
                group_df[
                    "mean_area_similarity"
                ].mean()
            ),

        "max_growth_streak":
            int(
                group_df[
                    "growth_streak"
                ].max()
            ),

        "max_reduction_streak":
            int(
                group_df[
                    "reduction_streak"
                ].max()
            )
    })


sequence_summary = pd.DataFrame(
    sequence_rows
)


# ============================================================
# GLOBAL STATE SUMMARY
# ============================================================

state_summary = (
    progression[
        [
            "persistent_damage_state",
            "rapid_event_type",
            "progression_confidence"
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

persistent_growth = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "PERSISTENT_GROWTH"
    )
    .sum()
)

probable_growth = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "PROBABLE_GROWTH"
    )
    .sum()
)

persistent_reduction = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "PERSISTENT_REDUCTION"
    )
    .sum()
)

probable_reduction = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "PROBABLE_REDUCTION"
    )
    .sum()
)

spatial_increase = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "SPATIAL_TRANSITION_INCREASE"
    )
    .sum()
)

spatial_decrease = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "SPATIAL_TRANSITION_DECREASE"
    )
    .sum()
)

ambiguous = int(
    progression[
        "persistent_damage_state"
    ]
    .eq(
        "AMBIGUOUS"
    )
    .sum()
)

stable_states = int(
    progression[
        "persistent_damage_state"
    ]
    .str.startswith(
        "STABLE"
    )
    .sum()
)


# ============================================================
# RAPID EVENT COUNTS
# ============================================================

rapid_total = int(
    progression[
        "change_magnitude_class"
    ]
    .eq(
        "RAPID"
    )
    .sum()
)

rapid_persistent_growth = int(
    progression[
        "rapid_event_type"
    ]
    .eq(
        "RAPID_PERSISTENT_GROWTH"
    )
    .sum()
)

rapid_persistent_reduction = int(
    progression[
        "rapid_event_type"
    ]
    .eq(
        "RAPID_PERSISTENT_REDUCTION"
    )
    .sum()
)

rapid_spatial_increase = int(
    progression[
        "rapid_event_type"
    ]
    .eq(
        "RAPID_SPATIAL_TRANSITION_INCREASE"
    )
    .sum()
)

rapid_spatial_decrease = int(
    progression[
        "rapid_event_type"
    ]
    .eq(
        "RAPID_SPATIAL_TRANSITION_DECREASE"
    )
    .sum()
)


# ============================================================
# CORRELATIONS
# ============================================================

correlation_targets = [
    "absolute_damage_change"
]

correlation_features = [
    "persistent_fraction",
    "future_coverage",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score",
    "track_continuity_quality"
]

correlations = {}

for feature in correlation_features:

    correlations[feature] = safe_corr(
        progression,
        feature,
        "absolute_damage_change"
    )


# ============================================================
# PERSISTENCE VS DAMAGE CHANGE
# ============================================================

growth_change = progression[
    progression[
        "persistent_growth_flag"
    ]
]

reduction_change = progression[
    progression[
        "persistent_reduction_flag"
    ]
]

spatial_change = progression[
    progression[
        "persistent_damage_state"
    ]
    .isin(
        [
            "SPATIAL_TRANSITION_INCREASE",
            "SPATIAL_TRANSITION_DECREASE"
        ]
    )
]


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


invalid_temporal_direction = int(
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
        "3.5",

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

    "thresholds": {

        "p75":
            p75,

        "p90":
            p90,

        "p95":
            p95
    },

    "state_counts": {

        "persistent_growth":
            persistent_growth,

        "probable_growth":
            probable_growth,

        "persistent_reduction":
            persistent_reduction,

        "probable_reduction":
            probable_reduction,

        "spatial_transition_increase":
            spatial_increase,

        "spatial_transition_decrease":
            spatial_decrease,

        "ambiguous":
            ambiguous,

        "stable":
            stable_states
    },

    "state_percentages": {

        "persistent_growth":
            percentage(
                persistent_growth,
                len(progression)
            ),

        "probable_growth":
            percentage(
                probable_growth,
                len(progression)
            ),

        "persistent_reduction":
            percentage(
                persistent_reduction,
                len(progression)
            ),

        "probable_reduction":
            percentage(
                probable_reduction,
                len(progression)
            ),

        "spatial_transition_increase":
            percentage(
                spatial_increase,
                len(progression)
            ),

        "spatial_transition_decrease":
            percentage(
                spatial_decrease,
                len(progression)
            ),

        "ambiguous":
            percentage(
                ambiguous,
                len(progression)
            ),

        "stable":
            percentage(
                stable_states,
                len(progression)
            )
    },

    "rapid_events": {

        "total":
            rapid_total,

        "persistent_growth":
            rapid_persistent_growth,

        "persistent_reduction":
            rapid_persistent_reduction,

        "spatial_transition_increase":
            rapid_spatial_increase,

        "spatial_transition_decrease":
            rapid_spatial_decrease
    },

    "rapid_event_percentages": {

        "persistent_growth":
            percentage(
                rapid_persistent_growth,
                rapid_total
            ),

        "persistent_reduction":
            percentage(
                rapid_persistent_reduction,
                rapid_total
            ),

        "spatial_transition":
            percentage(
                rapid_spatial_increase
                +
                rapid_spatial_decrease,
                rapid_total
            )
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

    "correlations_with_absolute_damage_change":
        correlations,

    "maximum_growth_streak":
        int(
            progression[
                "growth_streak"
            ].max()
        ),

    "maximum_reduction_streak":
        int(
            progression[
                "reduction_streak"
            ].max()
        ),

    "integrity": {

        "duplicate_keys":
            duplicate_keys,

        "invalid_temporal_direction":
            invalid_temporal_direction,

        "sequence_split_conflicts":
            sequence_split_conflicts,

        "component_split_conflicts":
            component_split_conflicts
    },

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
print("STAGE 3.5 SUMMARY")
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
    "\nSpatial evidence:"
)

print(
    "  STRONG:",
    int(
        (
            progression[
                "spatial_evidence_class"
            ]
            ==
            "STRONG"
        ).sum()
    )
)

print(
    "  MODERATE:",
    int(
        (
            progression[
                "spatial_evidence_class"
            ]
            ==
            "MODERATE"
        ).sum()
    )
)

print(
    "  WEAK:",
    int(
        (
            progression[
                "spatial_evidence_class"
            ]
            ==
            "WEAK"
        ).sum()
    )
)

print(
    "  NONE:",
    int(
        (
            progression[
                "spatial_evidence_class"
            ]
            ==
            "NONE"
        ).sum()
    )
)


print(
    "\nPersistent progression states:"
)

for state in [
    "PERSISTENT_GROWTH",
    "PROBABLE_GROWTH",
    "PERSISTENT_REDUCTION",
    "PROBABLE_REDUCTION",
    "SPATIAL_TRANSITION_INCREASE",
    "SPATIAL_TRANSITION_DECREASE",
    "AMBIGUOUS",
    "STABLE_PERSISTENT_DAMAGE",
    "STABLE_MODERATE_CONTINUITY",
    "STABLE_LOW_CONTINUITY"
]:

    count = int(
        (
            progression[
                "persistent_damage_state"
            ]
            ==
            state
        ).sum()
    )

    if count > 0:

        print(
            f"  {state}: "
            f"{count} "
            f"({percentage(count, len(progression)):.2f}%)"
        )


print(
    "\nRapid events:"
)

print(
    "  Total rapid events:",
    rapid_total
)

print(
    "  Rapid persistent growth:",
    rapid_persistent_growth
)

print(
    "  Rapid persistent reduction:",
    rapid_persistent_reduction
)

print(
    "  Rapid spatial increase:",
    rapid_spatial_increase
)

print(
    "  Rapid spatial decrease:",
    rapid_spatial_decrease
)


print(
    "\nSequential persistence:"
)

print(
    "  Maximum growth streak:",
    int(
        progression[
            "growth_streak"
        ].max()
    )
)

print(
    "  Maximum reduction streak:",
    int(
        progression[
            "reduction_streak"
        ].max()
    )
)


print(
    "\nCorrelations with absolute damage change:"
)

for feature, value in correlations.items():

    if value is None:

        print(
            f"  {feature}: N/A"
        )

    else:

        print(
            f"  {feature}: "
            f"{value:.6f}"
        )


print(
    "\nIntegrity checks:"
)

print(
    "  Duplicate transition keys:",
    duplicate_keys
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
    "\nOutput progression:"
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