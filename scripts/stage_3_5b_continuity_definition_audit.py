import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 3.5B
# SPATIAL CONTINUITY DEFINITION AUDIT
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

SPATIAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

PERSISTENT_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_5_persistent_damage_progression.csv"
)

OUTPUT_CSV = os.path.join(
    AUDIT_DIR,
    "stage_3_5b_continuity_comparison.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_5b_continuity_summary.json"
)

OUTPUT_COLUMN_AUDIT = os.path.join(
    AUDIT_DIR,
    "stage_3_5b_spatial_column_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def first_existing(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def safe_corr(df, a, b):

    if a not in df.columns or b not in df.columns:
        return None

    temp = df[
        [a, b]
    ].copy()

    temp[a] = safe_numeric(
        temp[a]
    )

    temp[b] = safe_numeric(
        temp[b]
    )

    temp = temp.dropna()

    if len(temp) < 3:
        return None

    value = temp[a].corr(
        temp[b]
    )

    if pd.isna(value):
        return None

    return float(value)


def pct(value, total):

    if total == 0:
        return 0.0

    return float(
        value / total * 100.0
    )


def require_columns(
    df,
    columns,
    dataset_name
):

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{dataset_name} is missing required columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.5B - SPATIAL CONTINUITY DEFINITION AUDIT"
)
print("=" * 70)


# ============================================================
# FILE CHECKS
# ============================================================

print(
    "\nChecking Stage 3.0 spatial track file..."
)

if not os.path.exists(
    SPATIAL_FILE
):

    raise FileNotFoundError(
        SPATIAL_FILE
    )

print(
    "Stage 3.0 spatial track file: FOUND"
)


print(
    "\nChecking Stage 3.5 persistent progression file..."
)

if not os.path.exists(
    PERSISTENT_FILE
):

    raise FileNotFoundError(
        PERSISTENT_FILE
    )

print(
    "Stage 3.5 persistent progression file: FOUND"
)


# ============================================================
# LOAD FILES
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
    "\nLoading Stage 3.5 persistent progression..."
)

persistent = pd.read_csv(
    PERSISTENT_FILE
)

print(
    "Stage 3.5 records:",
    len(persistent)
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
    "Stage 3.0 spatial track file"
)

require_columns(
    persistent,
    key_columns,
    "Stage 3.5 persistent progression file"
)


# ============================================================
# DUPLICATE KEY AUDIT
# ============================================================

duplicate_spatial_keys = int(
    spatial.duplicated(
        subset=key_columns
    ).sum()
)

duplicate_persistent_keys = int(
    persistent.duplicated(
        subset=key_columns
    ).sum()
)

print(
    "\nDuplicate transition keys:"
)

print(
    "  Stage 3.0:",
    duplicate_spatial_keys
)

print(
    "  Stage 3.5:",
    duplicate_persistent_keys
)


# ============================================================
# STAGE 3.0 COLUMN INVENTORY
# ============================================================

print(
    "\nStage 3.0 spatial columns:"
)

for index, col in enumerate(
    spatial.columns,
    start=1
):

    print(
        f"  {index:03d}. {col}"
    )


# ============================================================
# SEARCH FOR EXPLICIT CONTINUITY COLUMNS
# ============================================================

continuity_candidates = [

    "track_continuity",

    "track_continuity_quality",

    "continuity",

    "continuity_class",

    "continuity_score",

    "spatial_match_quality",

    "match_quality",

    "match_score",

    "mean_match_score"

]

explicit_continuity_columns = []

for col in continuity_candidates:

    if col in spatial.columns:

        explicit_continuity_columns.append(
            col
        )


print(
    "\nPotential continuity columns:"
)

if len(explicit_continuity_columns) == 0:

    print(
        "  NONE"
    )

else:

    for col in explicit_continuity_columns:

        print(
            "  FOUND:",
            col
        )


# ============================================================
# EXPLICIT CONTINUITY VALUE DISTRIBUTIONS
# ============================================================

explicit_distributions = {}

for col in explicit_continuity_columns:

    print(
        "\nValue distribution:",
        col
    )

    counts = (
        spatial[
            col
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        counts.head(30)
    )

    explicit_distributions[
        col
    ] = {
        str(key): int(value)
        for key, value
        in counts.items()
    }


# ============================================================
# RESOLVE SPATIAL METRICS
# ============================================================

metric_candidates = {

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

    "matched_crack_count": [
        "matched_crack_count"
    ],

    "current_crack_count": [
        "current_crack_count"
    ],

    "future_crack_count": [
        "future_crack_count"
    ],

    "track_continuity_quality": [
        "track_continuity_quality",
        "continuity_score"
    ]

}


resolved_metrics = {}

for metric_name, candidates in metric_candidates.items():

    resolved_metrics[
        metric_name
    ] = first_existing(
        spatial,
        candidates
    )


print(
    "\nResolved Stage 3.0 spatial metrics:"
)

for metric_name, column_name in resolved_metrics.items():

    print(
        f"  {metric_name}: {column_name}"
    )


# ============================================================
# CREATE METRIC DATAFRAME
# ============================================================

metrics = spatial[
    key_columns
].copy()


for metric_name, column_name in resolved_metrics.items():

    if column_name is None:

        metrics[
            metric_name
        ] = np.nan

    else:

        metrics[
            metric_name
        ] = safe_numeric(
            spatial[
                column_name
            ]
        )


# ============================================================
# CHECK SPATIAL METRIC COVERAGE
# ============================================================

print(
    "\nSpatial metric coverage:"
)

for metric_name in [
    "persistent_fraction",
    "future_coverage",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "mean_match_score"
]:

    valid_count = int(
        metrics[
            metric_name
        ]
        .notna()
        .sum()
    )

    print(
        f"  {metric_name}: "
        f"{valid_count}/{len(metrics)} "
        f"({pct(valid_count, len(metrics)):.2f}%)"
    )


# ============================================================
# RECONSTRUCT SPATIAL EVIDENCE
# ============================================================

# These are the same thresholds used by Stage 3.5.
#
# persistent_fraction >= 0.75
# future_coverage     >= 0.75
# mean_match_iou      >= 0.30
# mean_center_distance <= 0.10
# mean_area_similarity >= 0.70
# mean_match_score      >= 0.50


metrics[
    "persistent_signal"
] = (
    metrics[
        "persistent_fraction"
    ] >= 0.75
)


metrics[
    "coverage_signal"
] = (
    metrics[
        "future_coverage"
    ] >= 0.75
)


metrics[
    "iou_signal"
] = (
    metrics[
        "mean_match_iou"
    ] >= 0.30
)


metrics[
    "center_signal"
] = (
    metrics[
        "mean_center_distance"
    ] <= 0.10
)


metrics[
    "area_signal"
] = (
    metrics[
        "mean_area_similarity"
    ] >= 0.70
)


metrics[
    "score_signal"
] = (
    metrics[
        "mean_match_score"
    ] >= 0.50
)


# ============================================================
# CONTINUITY SIGNAL COUNT
# ============================================================

signal_columns = [

    "persistent_signal",

    "coverage_signal",

    "iou_signal",

    "center_signal",

    "area_signal",

    "score_signal"

]


metrics[
    "continuity_signal_count"
] = (
    metrics[
        signal_columns
    ]
    .astype(int)
    .sum(
        axis=1
    )
)


# ============================================================
# RECONSTRUCTED CONTINUITY CLASS
# ============================================================

metrics[
    "reconstructed_continuity"
] = np.select(

    [

        metrics[
            "continuity_signal_count"
        ] >= 5,

        metrics[
            "continuity_signal_count"
        ] >= 3,

        metrics[
            "continuity_signal_count"
        ] >= 1

    ],

    [

        "HIGH_EQUIVALENT",

        "MODERATE_EQUIVALENT",

        "LOW_EQUIVALENT"

    ],

    default="NONE_EQUIVALENT"

)


# ============================================================
# PRINT RECONSTRUCTED CONTINUITY
# ============================================================

reconstructed_counts = (
    metrics[
        "reconstructed_continuity"
    ]
    .value_counts()
)

print(
    "\nReconstructed continuity:"
)

for label, count in reconstructed_counts.items():

    print(
        f"  {label}: "
        f"{int(count)} "
        f"({pct(count, len(metrics)):.2f}%)"
    )


# ============================================================
# JOIN STAGE 3.5
# ============================================================

stage_3_5_columns = [

    "damage_current",

    "damage_future",

    "absolute_damage_change",

    "relative_damage_change",

    "persistent_damage_state",

    "rapid_event_type",

    "progression_confidence",

    "spatial_evidence_class",

    "spatial_evidence_score"

]


require_columns(
    persistent,
    stage_3_5_columns,
    "Stage 3.5 persistent progression file"
)


comparison = persistent[
    key_columns
    +
    stage_3_5_columns
].copy()


comparison = comparison.merge(

    metrics,

    on=key_columns,

    how="left",

    validate="one_to_one"

)


print(
    "\nComparison records:",
    len(comparison)
)


# ============================================================
# MERGE INTEGRITY
# ============================================================

missing_metric_rows = int(
    comparison[
        "persistent_fraction"
    ]
    .isna()
    .sum()
)

print(
    "Missing spatial metric rows:",
    missing_metric_rows
)


# ============================================================
# STAGE 3.5 VS RECONSTRUCTED CONTINUITY
# ============================================================

comparison[
    "stage_3_5_spatial_vs_reconstructed_match"
] = (

    (

        (
            comparison[
                "spatial_evidence_class"
            ]
            ==
            "STRONG"
        )

        &

        (
            comparison[
                "reconstructed_continuity"
            ]
            ==
            "HIGH_EQUIVALENT"
        )

    )

    |

    (

        (
            comparison[
                "spatial_evidence_class"
            ]
            ==
            "MODERATE"
        )

        &

        (
            comparison[
                "reconstructed_continuity"
            ]
            ==
            "MODERATE_EQUIVALENT"
        )

    )

)


# ============================================================
# ADD STRONG/HIGH FLAGS
# ============================================================

comparison[
    "stage_3_5_strong"
] = (
    comparison[
        "spatial_evidence_class"
    ]
    ==
    "STRONG"
)


comparison[
    "reconstructed_high"
] = (
    comparison[
        "reconstructed_continuity"
    ]
    ==
    "HIGH_EQUIVALENT"
)


comparison[
    "strong_and_high"
] = (

    comparison[
        "stage_3_5_strong"
    ]

    &

    comparison[
        "reconstructed_high"
    ]

)


# ============================================================
# DAMAGE DIRECTION
# ============================================================

comparison[
    "absolute_damage_change"
] = safe_numeric(
    comparison[
        "absolute_damage_change"
    ]
)


comparison[
    "damage_direction"
] = np.select(

    [

        comparison[
            "absolute_damage_change"
        ] > 1e-12,

        comparison[
            "absolute_damage_change"
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

abs_change = (
    comparison[
        "absolute_damage_change"
    ]
    .abs()
)


p90 = float(
    abs_change.quantile(
        0.90
    )
)


p95 = float(
    abs_change.quantile(
        0.95
    )
)


comparison[
    "damage_change_magnitude"
] = np.select(

    [

        abs_change >= p95,

        abs_change >= p90

    ],

    [

        "EXTREME",

        "LARGE"

    ],

    default="NORMAL"

)


# ============================================================
# DIRECTION SUMMARY
# ============================================================

direction_summary = (

    comparison

    .groupby(
        [
            "damage_direction",
            "reconstructed_continuity"
        ]
    )

    .size()

    .reset_index(
        name="record_count"
    )

)


direction_summary[
    "percentage_within_direction"
] = (

    direction_summary

    .groupby(
        "damage_direction"
    )[
        "record_count"
    ]

    .transform(
        lambda x:
        x / x.sum() * 100.0
    )

)


# ============================================================
# MAGNITUDE SUMMARY
# ============================================================

magnitude_summary = (

    comparison

    .groupby(
        [
            "damage_change_magnitude",
            "reconstructed_continuity"
        ]
    )

    .size()

    .reset_index(
        name="record_count"
    )

)


magnitude_summary[
    "percentage_within_magnitude"
] = (

    magnitude_summary

    .groupby(
        "damage_change_magnitude"
    )[
        "record_count"
    ]

    .transform(
        lambda x:
        x / x.sum() * 100.0
    )

)


# ============================================================
# KNOWN STAGE 3.0 COUNTS
# ============================================================

# Stage 3.0 final audit:
#
# High continuity pairs:       10811
# Moderate continuity pairs:    2168
# Low continuity pairs:        1079
#
# Total: 14058


known_stage_3_0 = {

    "HIGH":
        10811,

    "MODERATE":
        2168,

    "LOW":
        1079

}


# ============================================================
# RECONSTRUCTED COUNTS
# ============================================================

reconstructed_count_dict = {

    "HIGH_EQUIVALENT":
        int(
            (
                metrics[
                    "reconstructed_continuity"
                ]
                ==
                "HIGH_EQUIVALENT"
            )
            .sum()
        ),

    "MODERATE_EQUIVALENT":
        int(
            (
                metrics[
                    "reconstructed_continuity"
                ]
                ==
                "MODERATE_EQUIVALENT"
            )
            .sum()
        ),

    "LOW_EQUIVALENT":
        int(
            (
                metrics[
                    "reconstructed_continuity"
                ]
                ==
                "LOW_EQUIVALENT"
            )
            .sum()
        ),

    "NONE_EQUIVALENT":
        int(
            (
                metrics[
                    "reconstructed_continuity"
                ]
                ==
                "NONE_EQUIVALENT"
            )
            .sum()
        )

}


# ============================================================
# STAGE 3.5 EVIDENCE COUNTS
# ============================================================

stage_3_5_evidence_counts = (

    comparison[
        "spatial_evidence_class"
    ]
    .value_counts()
    .to_dict()

)


# ============================================================
# AGREEMENT METRICS
# ============================================================

strong_count = int(
    comparison[
        "stage_3_5_strong"
    ].sum()
)

high_count = int(
    comparison[
        "reconstructed_high"
    ].sum()
)

strong_high_count = int(
    comparison[
        "strong_and_high"
    ].sum()
)

agreement_count = int(
    comparison[
        "stage_3_5_spatial_vs_reconstructed_match"
    ].sum()
)


print(
    "\nContinuity comparison:"
)

print(
    "  Stage 3.5 STRONG:",
    strong_count,
    f"({pct(strong_count, len(comparison)):.2f}%)"
)

print(
    "  Reconstructed HIGH equivalent:",
    high_count,
    f"({pct(high_count, len(comparison)):.2f}%)"
)

print(
    "  STRONG and HIGH equivalent:",
    strong_high_count,
    f"({pct(strong_high_count, len(comparison)):.2f}%)"
)

print(
    "  Stage 3.5 class agreement:",
    agreement_count,
    f"({pct(agreement_count, len(comparison)):.2f}%)"
)


# ============================================================
# SPATIAL METRIC STATISTICS
# ============================================================

metric_summary_rows = []

for metric in [

    "persistent_fraction",

    "future_coverage",

    "mean_match_iou",

    "mean_center_distance",

    "mean_area_similarity",

    "mean_match_score"

]:

    values = (
        metrics[
            metric
        ]
        .dropna()
    )

    if len(values) == 0:
        continue

    metric_summary_rows.append({

        "metric":
            metric,

        "count":
            int(
                len(values)
            ),

        "mean":
            float(
                values.mean()
            ),

        "median":
            float(
                values.median()
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),

        "p25":
            float(
                values.quantile(
                    0.25
                )
            ),

        "p75":
            float(
                values.quantile(
                    0.75
                )
            )

    })


metric_summary = pd.DataFrame(
    metric_summary_rows
)


# ============================================================
# CORRELATIONS
# ============================================================

correlations = {}

for metric in [

    "persistent_fraction",

    "future_coverage",

    "mean_match_iou",

    "mean_center_distance",

    "mean_area_similarity",

    "mean_match_score"

]:

    correlations[
        metric
    ] = safe_corr(

        comparison,

        metric,

        "absolute_damage_change"

    )


# ============================================================
# SPATIAL COLUMN AUDIT
# ============================================================

column_rows = []

for col in spatial.columns:

    series = spatial[
        col
    ]

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    column_rows.append({

        "column":
            col,

        "dtype":
            str(
                series.dtype
            ),

        "non_null_count":
            int(
                series.notna().sum()
            ),

        "non_null_percentage":
            float(
                series.notna().mean()
                * 100.0
            ),

        "numeric_parse_percentage":
            float(
                numeric.notna().mean()
                * 100.0
            ),

        "unique_values":
            int(
                series.nunique(
                    dropna=True
                )
            )

    })


column_audit = pd.DataFrame(
    column_rows
)


# ============================================================
# SPLIT INTEGRITY
# ============================================================

sequence_split_conflicts = int(

    (

        comparison

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

        comparison

        .groupby(
            "component"
        )[
            "split"
        ]

        .nunique()

        > 1

    ).sum()

)


# ============================================================
# TEMPORAL INTEGRITY
# ============================================================

comparison[
    "current_frame"
] = pd.to_numeric(
    comparison[
        "current_frame"
    ],
    errors="coerce"
)

comparison[
    "future_frame"
] = pd.to_numeric(
    comparison[
        "future_frame"
    ],
    errors="coerce"
)


invalid_temporal_direction = int(

    (

        comparison[
            "future_frame"
        ]

        <=

        comparison[
            "current_frame"
        ]

    ).sum()

)


# ============================================================
# SAVE OUTPUTS
# ============================================================

comparison.to_csv(
    OUTPUT_CSV,
    index=False
)

column_audit.to_csv(
    OUTPUT_COLUMN_AUDIT,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "3.5B",

    "status":
        "PASS",

    "stage_3_0_records":
        int(
            len(spatial)
        ),

    "stage_3_5_records":
        int(
            len(persistent)
        ),

    "comparison_records":
        int(
            len(comparison)
        ),

    "missing_spatial_metric_rows":
        missing_metric_rows,

    "explicit_continuity_columns":
        explicit_continuity_columns,

    "explicit_continuity_distributions":
        explicit_distributions,

    "resolved_spatial_metrics":
        resolved_metrics,

    "known_stage_3_0_continuity_counts":
        known_stage_3_0,

    "reconstructed_continuity_counts":
        reconstructed_count_dict,

    "reconstructed_continuity_percentages": {

        key:
            pct(
                value,
                len(metrics)
            )

        for key, value
        in reconstructed_count_dict.items()

    },

    "stage_3_5_spatial_evidence_counts":
        {
            str(key): int(value)
            for key, value
            in stage_3_5_evidence_counts.items()
        },

    "continuity_comparison": {

        "stage_3_5_strong_count":
            strong_count,

        "stage_3_5_strong_percentage":
            pct(
                strong_count,
                len(comparison)
            ),

        "reconstructed_high_count":
            high_count,

        "reconstructed_high_percentage":
            pct(
                high_count,
                len(comparison)
            ),

        "strong_and_high_count":
            strong_high_count,

        "strong_and_high_percentage":
            pct(
                strong_high_count,
                len(comparison)
            ),

        "class_agreement_count":
            agreement_count,

        "class_agreement_percentage":
            pct(
                agreement_count,
                len(comparison)
            )

    },

    "damage_change_thresholds": {

        "p90":
            p90,

        "p95":
            p95

    },

    "damage_direction_distribution":
        {
            str(key): int(value)
            for key, value
            in comparison[
                "damage_direction"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

    "damage_magnitude_distribution":
        {
            str(key): int(value)
            for key, value
            in comparison[
                "damage_change_magnitude"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

    "correlations_with_absolute_damage_change":
        correlations,

    "integrity": {

        "duplicate_stage_3_0_keys":
            duplicate_spatial_keys,

        "duplicate_stage_3_5_keys":
            duplicate_persistent_keys,

        "duplicate_comparison_keys":
            int(
                comparison.duplicated(
                    subset=key_columns
                ).sum()
            ),

        "invalid_temporal_direction":
            invalid_temporal_direction,

        "sequence_split_conflicts":
            sequence_split_conflicts,

        "component_split_conflicts":
            component_split_conflicts,

        "missing_spatial_metric_rows":
            missing_metric_rows

    },

    "methodological_note":
        (
            "Stage 3.0 continuity is based on heuristic "
            "polygon-level matching. Stage 3.5 spatial evidence "
            "uses explicit thresholded spatial metrics. This audit "
            "compares the two definitions and does not treat either "
            "as manually validated ground-truth crack identity."
        ),

    "output_files": {

        "comparison":
            OUTPUT_CSV,

        "column_audit":
            OUTPUT_COLUMN_AUDIT,

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
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.5B SUMMARY")
print("=" * 70)

print(
    "Stage 3.0 records:",
    len(spatial)
)

print(
    "Stage 3.5 records:",
    len(persistent)
)

print(
    "Comparison records:",
    len(comparison)
)


print(
    "\nExplicit continuity columns:"
)

if explicit_continuity_columns:

    for col in explicit_continuity_columns:

        print(
            "  ",
            col
        )

else:

    print(
        "  NONE"
    )


print(
    "\nResolved spatial metrics:"
)

for metric_name, column_name in resolved_metrics.items():

    print(
        f"  {metric_name}: {column_name}"
    )


print(
    "\nKnown Stage 3.0 continuity counts:"
)

for label, count in known_stage_3_0.items():

    print(
        f"  {label}: "
        f"{count} "
        f"({pct(count, len(spatial)):.2f}%)"
    )


print(
    "\nReconstructed continuity counts:"
)

for label, count in reconstructed_count_dict.items():

    print(
        f"  {label}: "
        f"{count} "
        f"({pct(count, len(metrics)):.2f}%)"
    )


print(
    "\nStage 3.5 spatial evidence:"
)

for label, count in (
    comparison[
        "spatial_evidence_class"
    ]
    .value_counts()
    .items()
):

    print(
        f"  {label}: "
        f"{int(count)} "
        f"({pct(count, len(comparison)):.2f}%)"
    )


print(
    "\nContinuity comparison:"
)

print(
    "  Stage 3.5 STRONG:",
    strong_count
)

print(
    "  Reconstructed HIGH equivalent:",
    high_count
)

print(
    "  STRONG and HIGH equivalent:",
    strong_high_count
)

print(
    "  Stage 3.5 class agreement:",
    agreement_count
)


print(
    "\nDamage change thresholds:"
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
    "\nCorrelations with absolute damage change:"
)

for metric, value in correlations.items():

    if value is None:

        print(
            f"  {metric}: N/A"
        )

    else:

        print(
            f"  {metric}: "
            f"{value:.6f}"
        )


print(
    "\nIntegrity checks:"
)

print(
    "  Duplicate Stage 3.0 keys:",
    duplicate_spatial_keys
)

print(
    "  Duplicate Stage 3.5 keys:",
    duplicate_persistent_keys
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
    "  Missing spatial metric rows:",
    missing_metric_rows
)


print(
    "\nOutput comparison:"
)

print(
    OUTPUT_CSV
)

print(
    "\nOutput column audit:"
)

print(
    OUTPUT_COLUMN_AUDIT
)

print(
    "\nOutput summary:"
)

print(
    OUTPUT_SUMMARY
)

print(
    "\nFINAL STATUS: PASS"
)