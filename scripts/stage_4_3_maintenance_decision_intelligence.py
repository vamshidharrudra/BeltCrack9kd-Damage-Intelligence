import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 4.3
# MAINTENANCE DECISION INTELLIGENCE
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_27_final_inference_predictions.csv"
)

SPATIAL_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_1_spatial_track_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_4_3_maintenance_intelligence"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "maintenance_decisions.csv"
)

SUMMARY_JSON = os.path.join(
    OUTPUT_DIR,
    "stage_4_3_summary.json"
)

SEQUENCE_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "sequence_maintenance_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Damage thresholds established from the project's
# physical-damage analysis.

VERY_LOW_THRESHOLD = 0.002
LOW_THRESHOLD = 0.005
MEDIUM_THRESHOLD = 0.008
HIGH_THRESHOLD = 0.012
VERY_HIGH_THRESHOLD = 0.016

# Rapid-change threshold from Stage 3.4 / 3.15.
RAPID_CHANGE_THRESHOLD = 0.003742

# Strong spatial continuity threshold.
# Stage 3.0 continuity is heuristic crack-polygon matching.
STRONG_SPATIAL_THRESHOLD = 0.75

# Maintenance decision scores.
NORMAL_MAX = 24
MONITOR_MAX = 49
INSPECT_MAX = 74


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    if pd.isna(value):
        return np.nan

    return float(value)


def damage_level(value):

    if value < VERY_LOW_THRESHOLD:
        return "VERY_LOW"

    if value < LOW_THRESHOLD:
        return "LOW"

    if value < MEDIUM_THRESHOLD:
        return "MEDIUM"

    if value < HIGH_THRESHOLD:
        return "HIGH"

    if value < VERY_HIGH_THRESHOLD:
        return "VERY_HIGH"

    return "EXTREME"


def severity_score(level):

    scores = {
        "VERY_LOW": 0,
        "LOW": 5,
        "MEDIUM": 15,
        "HIGH": 25,
        "VERY_HIGH": 35,
        "EXTREME": 45
    }

    return scores.get(
        level,
        0
    )


def change_score(change):

    magnitude = abs(
        float(change)
    )

    if magnitude >= 0.010079:
        return 30

    if magnitude >= 0.005489:
        return 25

    if magnitude >= 0.003742:
        return 18

    if magnitude >= 0.001924:
        return 10

    return 0


def spatial_score(value):

    if pd.isna(value):
        return 0

    value = float(value)

    if value >= STRONG_SPATIAL_THRESHOLD:
        return 20

    if value >= 0.50:
        return 12

    if value >= 0.25:
        return 6

    return 0


def direction_score(change):

    change = float(change)

    if change >= RAPID_CHANGE_THRESHOLD:
        return 20

    if change >= 0.001924:
        return 10

    if change <= -RAPID_CHANGE_THRESHOLD:
        return 8

    return 0


def decision_from_score(score):

    if score <= NORMAL_MAX:
        return "NORMAL"

    if score <= MONITOR_MAX:
        return "MONITOR"

    if score <= INSPECT_MAX:
        return "INSPECT"

    return "PRIORITY_INSPECTION"


def recommendation(
    decision,
    severity,
    predicted_change,
    spatial_quality
):

    if decision == "PRIORITY_INSPECTION":

        return (
            "Prioritize physical inspection of the "
            "belt region represented by this frame."
        )

    if decision == "INSPECT":

        return (
            "Schedule a closer belt inspection and "
            "review the recent damage progression."
        )

    if decision == "MONITOR":

        return (
            "Continue monitoring the damage progression "
            "in subsequent frames."
        )

    return (
        "No immediate inspection priority indicated "
        "by this research model."
    )


def build_reasons(
    severity,
    predicted_change,
    spatial_quality,
    decision
):

    reasons = []

    if severity in [
        "HIGH",
        "VERY_HIGH",
        "EXTREME"
    ]:

        reasons.append(
            "elevated predicted damage level"
        )

    if predicted_change >= RAPID_CHANGE_THRESHOLD:

        reasons.append(
            "rapid predicted damage increase"
        )

    elif predicted_change >= 0.001924:

        reasons.append(
            "positive predicted damage progression"
        )

    if (
        not pd.isna(spatial_quality)
        and
        spatial_quality >= STRONG_SPATIAL_THRESHOLD
    ):

        reasons.append(
            "strong spatial continuity evidence"
        )

    elif (
        not pd.isna(spatial_quality)
        and
        spatial_quality >= 0.50
    ):

        reasons.append(
            "moderate spatial continuity evidence"
        )

    if len(reasons) == 0:

        reasons.append(
            "no strong escalation indicator"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("STAGE 4.3 - MAINTENANCE DECISION INTELLIGENCE")
print("=" * 80)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)


if not os.path.exists(
    PREDICTION_PATH
):

    raise FileNotFoundError(
        "Missing prediction file:\n"
        +
        PREDICTION_PATH
    )


if not os.path.exists(
    SPATIAL_PATH
):

    raise FileNotFoundError(
        "Missing spatial feature file:\n"
        +
        SPATIAL_PATH
    )


print(
    "FOUND:",
    PREDICTION_PATH
)

print(
    "FOUND:",
    SPATIAL_PATH
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 80)
print("LOADING DATA")
print("=" * 80)


predictions = pd.read_csv(
    PREDICTION_PATH
)

spatial = pd.read_csv(
    SPATIAL_PATH
)


print(
    "Prediction records:",
    len(predictions)
)

print(
    "Spatial records:",
    len(spatial)
)


# ============================================================
# NORMALIZE KEYS
# ============================================================

predictions[
    "sequence_key"
] = predictions[
    "sequence_key"
].astype(str)


spatial[
    "sequence_key"
] = spatial[
    "sequence"
].astype(str)


# ------------------------------------------------------------
# FRAME KEYS
# ------------------------------------------------------------

if "frame_key" in predictions.columns:

    predictions[
        "frame_key"
    ] = predictions[
        "frame_key"
    ].apply(
        lambda x: int(float(x))
    )

else:

    raise ValueError(
        "Prediction file does not contain frame_key."
    )


if "current_frame" in spatial.columns:

    spatial[
        "frame_key"
    ] = spatial[
        "current_frame"
    ].apply(
        lambda x: int(float(x))
    )

elif "frame_number" in spatial.columns:

    spatial[
        "frame_key"
    ] = spatial[
        "frame_number"
    ].apply(
        lambda x: int(float(x))
    )

else:

    raise ValueError(
        "Could not identify current frame in spatial data."
    )


# ============================================================
# DUPLICATE CHECKS
# ============================================================

print()
print("=" * 80)
print("KEY INTEGRITY")
print("=" * 80)


prediction_duplicates = (
    predictions[
        [
            "sequence_key",
            "frame_key"
        ]
    ]
    .duplicated()
    .sum()
)


spatial_duplicates = (
    spatial[
        [
            "sequence_key",
            "frame_key"
        ]
    ]
    .duplicated()
    .sum()
)


print(
    "Prediction duplicate keys:",
    prediction_duplicates
)

print(
    "Spatial duplicate keys:",
    spatial_duplicates
)


if prediction_duplicates != 0:

    raise ValueError(
        "Prediction duplicate keys detected."
    )


if spatial_duplicates != 0:

    raise ValueError(
        "Spatial duplicate keys detected."
    )


# ============================================================
# IDENTIFY SPATIAL QUALITY FEATURE
# ============================================================

spatial_quality_candidates = [

    "track_continuity_quality",

    "spatial_match_quality",

    "persistent_fraction",

    "mean_match_iou"

]


spatial_quality_column = None


for column in spatial_quality_candidates:

    if column in spatial.columns:

        spatial_quality_column = column

        break


if spatial_quality_column is None:

    raise ValueError(
        "No suitable spatial continuity feature found."
    )


print(
    "Spatial quality feature:",
    spatial_quality_column
)


# ============================================================
# MERGE
# ============================================================

print()
print("=" * 80)
print("MERGING PREDICTION AND SPATIAL INFORMATION")
print("=" * 80)


spatial_subset = spatial[
    [
        "sequence_key",
        "frame_key",
        spatial_quality_column
    ]
].copy()


merged = predictions.merge(
    spatial_subset,
    on=[
        "sequence_key",
        "frame_key"
    ],
    how="left"
)


print(
    "Merged records:",
    len(merged)
)


if len(merged) != len(predictions):

    raise ValueError(
        "Merge changed prediction record count."
    )


# ============================================================
# BUILD DECISION FEATURES
# ============================================================

print()
print("=" * 80)
print("BUILDING MAINTENANCE DECISION FEATURES")
print("=" * 80)


merged[
    "current_damage"
] = pd.to_numeric(
    merged[
        "current_damage"
    ],
    errors="coerce"
)


merged[
    "predicted_damage"
] = pd.to_numeric(
    merged[
        "predicted_future_crack_area_ratio"
    ],
    errors="coerce"
)


merged[
    "actual_damage"
] = pd.to_numeric(
    merged[
        "future_crack_area_ratio_t1"
    ],
    errors="coerce"
)


merged[
    "predicted_change"
] = (
    merged[
        "predicted_damage"
    ]
    -
    merged[
        "current_damage"
    ]
)


merged[
    "prediction_error"
] = (
    merged[
        "absolute_error"
    ]
)


merged[
    "predicted_level"
] = merged[
    "predicted_damage"
].apply(
    damage_level
)


# ============================================================
# SCORE COMPONENTS
# ============================================================

merged[
    "severity_score"
] = merged[
    "predicted_level"
].apply(
    severity_score
)


merged[
    "change_score"
] = merged[
    "predicted_change"
].apply(
    change_score
)


merged[
    "spatial_score"
] = merged[
    spatial_quality_column
].apply(
    spatial_score
)


merged[
    "direction_score"
] = merged[
    "predicted_change"
].apply(
    direction_score
)


# ============================================================
# TOTAL RISK SCORE
# ============================================================

merged[
    "maintenance_risk_score"
] = (
    merged[
        "severity_score"
    ]
    +
    merged[
        "change_score"
    ]
    +
    merged[
        "spatial_score"
    ]
    +
    merged[
        "direction_score"
    ]
)


# Cap score for easier interpretation.

merged[
    "maintenance_risk_score"
] = merged[
    "maintenance_risk_score"
].clip(
    upper=100
)


# ============================================================
# MAINTENANCE DECISION
# ============================================================

merged[
    "maintenance_decision"
] = merged[
    "maintenance_risk_score"
].apply(
    decision_from_score
)


# ============================================================
# REASONS
# ============================================================

merged[
    "decision_reason"
] = merged.apply(
    lambda row:
    build_reasons(
        row[
            "predicted_level"
        ],
        row[
            "predicted_change"
        ],
        row[
            spatial_quality_column
        ],
        row[
            "maintenance_decision"
        ]
    ),
    axis=1
)


# ============================================================
# RECOMMENDATION
# ============================================================

merged[
    "maintenance_recommendation"
] = merged.apply(
    lambda row:
    recommendation(
        row[
            "maintenance_decision"
        ],
        row[
            "predicted_level"
        ],
        row[
            "predicted_change"
        ],
        row[
            spatial_quality_column
        ]
    ),
    axis=1
)


# ============================================================
# UNCERTAINTY INDICATOR
# ============================================================

# This is deliberately called an uncertainty indicator,
# NOT calibrated probability/confidence.

error_p95 = float(
    merged[
        "prediction_error"
    ].quantile(
        0.95
    )
)


error_p99 = float(
    merged[
        "prediction_error"
    ].quantile(
        0.99
    )
)


def uncertainty_indicator(error):

    if error <= error_p95:

        return "LOW"

    if error <= error_p99:

        return "MODERATE"

    return "HIGH"


merged[
    "prediction_uncertainty_indicator"
] = merged[
    "prediction_error"
].apply(
    uncertainty_indicator
)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [

    "sequence_key",

    "frame_key",

    "current_damage",

    "predicted_damage",

    "actual_damage",

    "predicted_change",

    "prediction_error",

    "predicted_level",

    spatial_quality_column,

    "severity_score",

    "change_score",

    "spatial_score",

    "direction_score",

    "maintenance_risk_score",

    "maintenance_decision",

    "prediction_uncertainty_indicator",

    "decision_reason",

    "maintenance_recommendation"
]


output_columns = [

    column
    for column in output_columns
    if column in merged.columns

]


decision_df = merged[
    output_columns
].copy()


# ============================================================
# SAVE ALL DECISIONS
# ============================================================

decision_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

print()
print("=" * 80)
print("SEQUENCE-LEVEL MAINTENANCE SUMMARY")
print("=" * 80)


sequence_rows = []


for sequence, group in decision_df.groupby(
    "sequence_key"
):

    highest = group.sort_values(
        "maintenance_risk_score",
        ascending=False
    ).iloc[0]


    sequence_rows.append(
        {
            "sequence":
                sequence,

            "records":
                int(len(group)),

            "mean_risk_score":
                float(
                    group[
                        "maintenance_risk_score"
                    ].mean()
                ),

            "max_risk_score":
                float(
                    group[
                        "maintenance_risk_score"
                    ].max()
                ),

            "priority_inspection_count":
                int(
                    (
                        group[
                            "maintenance_decision"
                        ]
                        ==
                        "PRIORITY_INSPECTION"
                    ).sum()
                ),

            "inspection_count":
                int(
                    (
                        group[
                            "maintenance_decision"
                        ]
                        ==
                        "INSPECT"
                    ).sum()
                ),

            "monitor_count":
                int(
                    (
                        group[
                            "maintenance_decision"
                        ]
                        ==
                        "MONITOR"
                    ).sum()
                ),

            "normal_count":
                int(
                    (
                        group[
                            "maintenance_decision"
                        ]
                        ==
                        "NORMAL"
                    ).sum()
                ),

            "highest_risk_frame":
                int(
                    highest[
                        "frame_key"
                    ]
                ),

            "highest_risk_level":
                str(
                    highest[
                        "predicted_level"
                    ]
                ),

            "highest_risk_decision":
                str(
                    highest[
                        "maintenance_decision"
                    ]
                )
        }
    )


sequence_summary = pd.DataFrame(
    sequence_rows
)


sequence_summary.to_csv(
    SEQUENCE_SUMMARY_CSV,
    index=False
)


print(
    sequence_summary.to_string(
        index=False
    )
)


# ============================================================
# GLOBAL DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("MAINTENANCE DECISION DISTRIBUTION")
print("=" * 80)


decision_counts = (
    decision_df[
        "maintenance_decision"
    ]
    .value_counts()
)


print(
    decision_counts
)


print()
print(
    "Predicted severity distribution:"
)

print(
    decision_df[
        "predicted_level"
    ].value_counts()
)


print()
print(
    "Uncertainty indicator distribution:"
)

print(
    decision_df[
        "prediction_uncertainty_indicator"
    ].value_counts()
)


# ============================================================
# HIGHEST RISK RECORDS
# ============================================================

print()
print("=" * 80)
print("TOP 20 HIGHEST-RISK FRAMES")
print("=" * 80)


top_risk = decision_df.sort_values(
    "maintenance_risk_score",
    ascending=False
).head(20)


print(
    top_risk[
        [
            "sequence_key",
            "frame_key",
            "predicted_damage",
            "predicted_change",
            "maintenance_risk_score",
            "maintenance_decision",
            "predicted_level",
            "prediction_uncertainty_indicator"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

total_records = len(
    decision_df
)


priority_count = int(
    (
        decision_df[
            "maintenance_decision"
        ]
        ==
        "PRIORITY_INSPECTION"
    ).sum()
)


inspect_count = int(
    (
        decision_df[
            "maintenance_decision"
        ]
        ==
        "INSPECT"
    ).sum()
)


monitor_count = int(
    (
        decision_df[
            "maintenance_decision"
        ]
        ==
        "MONITOR"
    ).sum()
)


normal_count = int(
    (
        decision_df[
            "maintenance_decision"
        ]
        ==
        "NORMAL"
    ).sum()
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

print()
print("=" * 80)
print("INTEGRITY CHECKS")
print("=" * 80)


numeric_columns = [
    "current_damage",
    "predicted_damage",
    "predicted_change",
    "prediction_error",
    "maintenance_risk_score"
]


missing_numeric = int(
    decision_df[
        numeric_columns
    ].isna().sum().sum()
)


nonfinite_numeric = int(
    np.isfinite(
        decision_df[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).sum()
)


total_numeric_values = (
    decision_df[
        numeric_columns
    ].size
)


nonfinite_numeric = (
    total_numeric_values
    -
    nonfinite_numeric
)


print(
    "Total records:",
    total_records
)

print(
    "Missing numeric values:",
    missing_numeric
)

print(
    "Non-finite numeric values:",
    nonfinite_numeric
)

print(
    "Decision values:",
    decision_df[
        "maintenance_decision"
    ].nunique()
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "4.3",

    "status":
        "PASS"
        if (
            total_records > 0
            and
            missing_numeric == 0
            and
            nonfinite_numeric == 0
        )
        else "REVIEW",

    "input_prediction_records":
        int(len(predictions)),

    "input_spatial_records":
        int(len(spatial)),

    "decision_records":
        int(total_records),

    "decision_distribution":
        {
            str(k):
                int(v)
            for k, v
            in decision_counts.items()
        },

    "severity_distribution":
        {
            str(k):
                int(v)
            for k, v
            in decision_df[
                "predicted_level"
            ].value_counts().items()
        },

    "uncertainty_distribution":
        {
            str(k):
                int(v)
            for k, v
            in decision_df[
                "prediction_uncertainty_indicator"
            ].value_counts().items()
        },

    "counts":
        {
            "priority_inspection":
                priority_count,

            "inspect":
                inspect_count,

            "monitor":
                monitor_count,

            "normal":
                normal_count
        },

    "error_reference":
        {
            "p95":
                error_p95,

            "p99":
                error_p99
        },

    "thresholds":
        {
            "very_low":
                VERY_LOW_THRESHOLD,

            "low":
                LOW_THRESHOLD,

            "medium":
                MEDIUM_THRESHOLD,

            "high":
                HIGH_THRESHOLD,

            "very_high":
                VERY_HIGH_THRESHOLD,

            "rapid_change":
                RAPID_CHANGE_THRESHOLD,

            "strong_spatial":
                STRONG_SPATIAL_THRESHOLD
        },

    "integrity":
        {
            "missing_numeric_zero":
                bool(
                    missing_numeric == 0
                ),

            "nonfinite_numeric_zero":
                bool(
                    nonfinite_numeric == 0
                ),

            "records_preserved":
                bool(
                    total_records
                    ==
                    len(predictions)
                )
        },

    "outputs":
        {
            "maintenance_decisions":
                OUTPUT_CSV,

            "sequence_summary":
                SEQUENCE_SUMMARY_CSV,

            "summary":
                SUMMARY_JSON
        },

    "interpretation":
        (
            "Research-oriented maintenance prioritization "
            "layer built on predicted crack-area damage, "
            "damage progression, and heuristic spatial "
            "continuity. It is not a certified industrial "
            "maintenance or safety decision system."
        )
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STAGE 4.3 FINAL SUMMARY")
print("=" * 80)

print(
    "Decision records:",
    total_records
)

print(
    "NORMAL:",
    normal_count
)

print(
    "MONITOR:",
    monitor_count
)

print(
    "INSPECT:",
    inspect_count
)

print(
    "PRIORITY_INSPECTION:",
    priority_count
)

print()
print(
    "Output:",
    OUTPUT_CSV
)

print(
    "Sequence summary:",
    SEQUENCE_SUMMARY_CSV
)

print(
    "Summary:",
    SUMMARY_JSON
)

print()
print("=" * 80)

if (
    total_records > 0
    and
    missing_numeric == 0
    and
    nonfinite_numeric == 0
):

    print(
        "STAGE 4.3 STATUS: PASS"
    )

else:

    print(
        "STAGE 4.3 STATUS: REVIEW"
    )

print("=" * 80)