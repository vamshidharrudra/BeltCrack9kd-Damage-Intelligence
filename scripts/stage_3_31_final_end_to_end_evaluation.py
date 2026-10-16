import os
import json
import numpy as np
import pandas as pd


# ============================================================
# STAGE 3.31
# FINAL END-TO-END DAMAGE INTELLIGENCE EVALUATION
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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

FINAL_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_final_end_to_end_summary.json"
)

SEQUENCE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_sequence_evaluation.csv"
)

DAMAGE_BIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_damage_bin_evaluation.csv"
)

ERROR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_error_analysis.csv"
)

MODEL_COMPARISON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_model_comparison.csv"
)

WORST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_worst_predictions.csv"
)

BEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_31_best_predictions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def mae(y, p):
    return float(
        np.mean(
            np.abs(
                np.asarray(y) - np.asarray(p)
            )
        )
    )


def rmse(y, p):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(y)
                    -
                    np.asarray(p)
                ) ** 2
            )
        )
    )


def r2_score(y, p):

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    denominator = np.sum(
        (y - np.mean(y)) ** 2
    )

    if denominator == 0:
        return np.nan

    return float(
        1
        -
        np.sum(
            (y - p) ** 2
        )
        /
        denominator
    )


# ------------------------------------------------------------
# IMPORTANT:
# Convert every NumPy/Pandas value into native Python.
# ------------------------------------------------------------

def make_json_safe(obj):

    if isinstance(
        obj,
        dict
    ):

        return {
            str(key):
                make_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            make_json_safe(value)
            for value in obj
        ]

    if isinstance(
        obj,
        np.integer
    ):

        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):

        if np.isnan(obj):
            return None

        if np.isinf(obj):
            return None

        return float(obj)

    if isinstance(
        obj,
        np.bool_
    ):

        return bool(obj)

    if isinstance(
        obj,
        bool
    ):

        return bool(obj)

    if isinstance(
        obj,
        float
    ):

        if np.isnan(obj):
            return None

        if np.isinf(obj):
            return None

        return float(obj)

    if isinstance(
        obj,
        np.ndarray
    ):

        return [
            make_json_safe(x)
            for x in obj.tolist()
        ]

    if pd.isna(obj):

        return None

    return obj


# ============================================================
# START
# ============================================================

header(
    "STAGE 3.31 - FINAL END-TO-END DAMAGE INTELLIGENCE EVALUATION"
)

print(
    "Project:",
    PROJECT_ROOT
)


# ============================================================
# INPUT CHECK
# ============================================================

header(
    "CHECKING INPUTS"
)

if not os.path.exists(
    PREDICTION_PATH
):

    raise FileNotFoundError(
        PREDICTION_PATH
    )

print(
    "FOUND:",
    os.path.basename(
        PREDICTION_PATH
    )
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

header(
    "LOADING FINAL INFERENCE PREDICTIONS"
)

df = pd.read_csv(
    PREDICTION_PATH
)

print(
    "Prediction records:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "sequence_key",

    "frame_key",

    "assigned_split",

    "current_damage",

    "predicted_future_crack_area_ratio",

    "future_crack_area_ratio_t1",

    "absolute_error",

    "predicted_damage_status",

    "predicted_damage_level"
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


# ============================================================
# OFFICIAL TEST SET
# ============================================================

header(
    "ISOLATING OFFICIAL HELD-OUT TEST SET"
)

test = df[
    df[
        "assigned_split"
    ].astype(str).str.upper()
    ==
    "TEST"
].copy()

test = test.sort_values(
    [
        "sequence_key",
        "frame_key"
    ]
).reset_index(
    drop=True
)

print(
    "Official TEST records:",
    len(test)
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_keys = int(
    test.duplicated(
        subset=[
            "sequence_key",
            "frame_key"
        ]
    ).sum()
)

print(
    "Duplicate sequence/frame keys:",
    duplicate_keys
)


# ============================================================
# NUMERIC VALIDATION
# ============================================================

header(
    "TARGET AND PREDICTION VALIDATION"
)

numeric_columns = [

    "current_damage",

    "predicted_future_crack_area_ratio",

    "future_crack_area_ratio_t1",

    "absolute_error"
]

for column in numeric_columns:

    test[column] = pd.to_numeric(
        test[column],
        errors="coerce"
    )


missing_numeric = int(
    test[
        numeric_columns
    ].isna().sum().sum()
)

nonfinite_numeric = int(
    (
        ~np.isfinite(
            test[
                numeric_columns
            ].to_numpy()
        )
    ).sum()
)

print(
    "Missing numeric values:",
    missing_numeric
)

print(
    "Non-finite numeric values:",
    nonfinite_numeric
)


# ============================================================
# ERROR VALIDATION
# ============================================================

test[
    "recomputed_error"
] = (

    test[
        "future_crack_area_ratio_t1"
    ]

    -

    test[
        "predicted_future_crack_area_ratio"
    ]

).abs()


error_difference = float(
    np.max(
        np.abs(
            test[
                "recomputed_error"
            ]
            -
            test[
                "absolute_error"
            ]
        )
    )
)

print(
    "Maximum stored/recomputed error difference:",
    error_difference
)


# ============================================================
# FINAL MODEL PERFORMANCE
# ============================================================

header(
    "FINAL MODEL PERFORMANCE"
)

y_true = test[
    "future_crack_area_ratio_t1"
].to_numpy(
    dtype=float
)

y_pred = test[
    "predicted_future_crack_area_ratio"
].to_numpy(
    dtype=float
)

y_current = test[
    "current_damage"
].to_numpy(
    dtype=float
)


final_mae = mae(
    y_true,
    y_pred
)

final_rmse = rmse(
    y_true,
    y_pred
)

final_r2 = r2_score(
    y_true,
    y_pred
)


# Persistence
persistence_mae = mae(
    y_true,
    y_current
)

persistence_rmse = rmse(
    y_true,
    y_current
)

persistence_r2 = r2_score(
    y_true,
    y_current
)


print(
    "FINAL MODEL MAE :",
    f"{final_mae:.8f}"
)

print(
    "FINAL MODEL RMSE:",
    f"{final_rmse:.8f}"
)

print(
    "FINAL MODEL R2  :",
    f"{final_r2:.8f}"
)

print(
    "PERSISTENCE MAE :",
    f"{persistence_mae:.8f}"
)

print(
    "PERSISTENCE RMSE:",
    f"{persistence_rmse:.8f}"
)

print(
    "PERSISTENCE R2  :",
    f"{persistence_r2:.8f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

improvement_vs_persistence = float(
    100
    *
    (
        persistence_mae
        -
        final_mae
    )
    /
    persistence_mae
)

temporal_only_mae = 0.00146310
temporal_only_rmse = 0.00212732
temporal_only_r2 = 0.65300701

improvement_vs_temporal = float(
    100
    *
    (
        temporal_only_mae
        -
        final_mae
    )
    /
    temporal_only_mae
)

print(
    "Improvement vs persistence:",
    f"{improvement_vs_persistence:.4f}%"
)

print(
    "Temporal-only MAE:",
    f"{temporal_only_mae:.8f}"
)

print(
    "Temporal-only R2:",
    f"{temporal_only_r2:.8f}"
)

print(
    "Improvement vs temporal-only:",
    f"{improvement_vs_temporal:.4f}%"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

header(
    "ERROR DISTRIBUTION"
)

absolute_error = np.abs(
    y_true - y_pred
)

signed_error = (
    y_true - y_pred
)

error_distribution = {

    "count":
        int(len(absolute_error)),

    "mean":
        float(np.mean(absolute_error)),

    "median":
        float(np.median(absolute_error)),

    "p75":
        float(np.percentile(
            absolute_error,
            75
        )),

    "p90":
        float(np.percentile(
            absolute_error,
            90
        )),

    "p95":
        float(np.percentile(
            absolute_error,
            95
        )),

    "p99":
        float(np.percentile(
            absolute_error,
            99
        )),

    "max":
        float(np.max(absolute_error)),

    "mean_signed_error":
        float(np.mean(signed_error)),

    "median_signed_error":
        float(np.median(signed_error)),

    "underprediction_count":
        int(np.sum(
            y_pred < y_true
        )),

    "overprediction_count":
        int(np.sum(
            y_pred > y_true
        )),

    "exact_prediction_count":
        int(np.sum(
            y_pred == y_true
        ))
}


for key, value in error_distribution.items():

    print(
        key + ":",
        value
    )


# ============================================================
# SEQUENCE GENERALIZATION
# ============================================================

header(
    "SEQUENCE-LEVEL GENERALIZATION"
)

sequence_rows = []

for sequence, group in test.groupby(
    "sequence_key"
):

    yt = group[
        "future_crack_area_ratio_t1"
    ].to_numpy(
        dtype=float
    )

    yp = group[
        "predicted_future_crack_area_ratio"
    ].to_numpy(
        dtype=float
    )

    yc = group[
        "current_damage"
    ].to_numpy(
        dtype=float
    )

    model_mae = mae(
        yt,
        yp
    )

    persistence = mae(
        yt,
        yc
    )

    sequence_improvement = float(
        100
        *
        (
            persistence
            -
            model_mae
        )
        /
        persistence
    ) if persistence != 0 else np.nan

    sequence_rows.append(
        {
            "sequence":
                str(sequence),

            "records":
                int(len(group)),

            "mae":
                model_mae,

            "rmse":
                rmse(yt, yp),

            "r2":
                r2_score(yt, yp),

            "persistence_mae":
                persistence,

            "improvement_vs_persistence":
                sequence_improvement
        }
    )


sequence_df = pd.DataFrame(
    sequence_rows
)

sequence_df = sequence_df.sort_values(
    "mae"
)

sequence_df.to_csv(
    SEQUENCE_OUTPUT,
    index=False
)

print(
    sequence_df.to_string(
        index=False
    )
)

sequences_beating_persistence = int(
    np.sum(
        sequence_df[
            "mae"
        ]
        <
        sequence_df[
            "persistence_mae"
        ]
    )
)

print(
    "\nSequences beating persistence:",
    sequences_beating_persistence,
    "/",
    len(sequence_df)
)


# ============================================================
# DAMAGE BIN ANALYSIS
# ============================================================

header(
    "DAMAGE-LEVEL GENERALIZATION"
)

bins = [
    -np.inf,
    0.002,
    0.005,
    0.008,
    0.012,
    0.016,
    np.inf
]

labels = [
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
    "EXTREME"
]

test[
    "actual_damage_bin"
] = pd.cut(
    test[
        "future_crack_area_ratio_t1"
    ],
    bins=bins,
    labels=labels,
    include_lowest=True
)

damage_rows = []

for level, group in test.groupby(
    "actual_damage_bin",
    observed=False
):

    if len(group) == 0:
        continue

    yt = group[
        "future_crack_area_ratio_t1"
    ].to_numpy(
        dtype=float
    )

    yp = group[
        "predicted_future_crack_area_ratio"
    ].to_numpy(
        dtype=float
    )

    yc = group[
        "current_damage"
    ].to_numpy(
        dtype=float
    )

    model_error = mae(
        yt,
        yp
    )

    persistence_error = mae(
        yt,
        yc
    )

    improvement = float(
        100
        *
        (
            persistence_error
            -
            model_error
        )
        /
        persistence_error
    ) if persistence_error != 0 else np.nan

    damage_rows.append(
        {
            "damage_bin":
                str(level),

            "records":
                int(len(group)),

            "mean_actual_damage":
                float(np.mean(yt)),

            "model_mae":
                model_error,

            "persistence_mae":
                persistence_error,

            "improvement_vs_persistence":
                improvement
        }
    )


damage_df = pd.DataFrame(
    damage_rows
)

damage_df.to_csv(
    DAMAGE_BIN_OUTPUT,
    index=False
)

print(
    damage_df.to_string(
        index=False
    )
)


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

header(
    "DIRECTION ANALYSIS"
)

test[
    "prediction_direction"
] = np.where(
    y_pred > y_current,
    "INCREASING",
    np.where(
        y_pred < y_current,
        "DECREASING",
        "STABLE"
    )
)

test[
    "actual_direction"
] = np.where(
    y_true > y_current,
    "INCREASING",
    np.where(
        y_true < y_current,
        "DECREASING",
        "STABLE"
    )
)

test[
    "direction_correct"
] = (
    test[
        "prediction_direction"
    ]
    ==
    test[
        "actual_direction"
    ]
)

direction_accuracy = float(
    test[
        "direction_correct"
    ].mean()
)

print(
    "Predicted direction accuracy:",
    f"{direction_accuracy:.6f}"
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

header(
    "WORST PREDICTIONS"
)

test[
    "absolute_error_recomputed"
] = absolute_error

test[
    "signed_error"
] = signed_error

display_columns = [

    "sequence_key",

    "frame_key",

    "current_damage",

    "predicted_future_crack_area_ratio",

    "future_crack_area_ratio_t1",

    "absolute_error_recomputed",

    "predicted_damage_status",

    "predicted_damage_level"
]

worst = test.sort_values(
    "absolute_error_recomputed",
    ascending=False
).head(
    25
).copy()

worst[
    display_columns
].to_csv(
    WORST_OUTPUT,
    index=False
)

print(
    worst[
        display_columns
    ].head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# BEST PREDICTIONS
# ============================================================

header(
    "BEST PREDICTIONS"
)

best = test.sort_values(
    "absolute_error_recomputed",
    ascending=True
).head(
    25
).copy()

best[
    display_columns
].to_csv(
    BEST_OUTPUT,
    index=False
)

print(
    best[
        display_columns
    ].head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

header(
    "MODEL COMPARISON"
)

model_comparison = pd.DataFrame(
    [
        {
            "model":
                "Persistence",

            "features":
                1,

            "test_mae":
                persistence_mae,

            "test_rmse":
                persistence_rmse,

            "test_r2":
                persistence_r2
        },

        {
            "model":
                "Temporal-only",

            "features":
                18,

            "test_mae":
                temporal_only_mae,

            "test_rmse":
                temporal_only_rmse,

            "test_r2":
                temporal_only_r2
        },

        {
            "model":
                "Final 33-feature",

            "features":
                33,

            "test_mae":
                final_mae,

            "test_rmse":
                final_rmse,

            "test_r2":
                final_r2
        }
    ]
)

model_comparison[
    "improvement_vs_persistence_percent"
] = (

    100
    *
    (
        persistence_mae
        -
        model_comparison[
            "test_mae"
        ]
    )
    /
    persistence_mae

)

model_comparison.to_csv(
    MODEL_COMPARISON_OUTPUT,
    index=False
)

print(
    model_comparison.to_string(
        index=False
    )
)


# ============================================================
# FINAL INTEGRITY CHECKS
# ============================================================

header(
    "FINAL END-TO-END INTEGRITY CHECKS"
)

checks = {

    "test_records_present":
        bool(len(test) > 0),

    "duplicate_keys_zero":
        bool(duplicate_keys == 0),

    "missing_numeric_zero":
        bool(missing_numeric == 0),

    "nonfinite_numeric_zero":
        bool(nonfinite_numeric == 0),

    "stored_error_matches":
        bool(error_difference < 1e-10),

    "final_mae_better_than_persistence":
        bool(final_mae < persistence_mae),

    "final_r2_positive":
        bool(final_r2 > 0),

    "all_test_sequences_evaluated":
        bool(len(sequence_df) > 0),

    "all_sequences_beat_persistence":
        bool(
            np.all(
                sequence_df[
                    "mae"
                ].to_numpy()
                <
                sequence_df[
                    "persistence_mae"
                ].to_numpy()
            )
        ),

    "damage_bins_evaluated":
        bool(len(damage_df) > 0)
}


for name, value in checks.items():

    print(
        name + ":",
        bool(value)
    )


overall_pass = bool(
    all(
        checks.values()
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

header(
    "FINAL STAGE 3.31 SUMMARY"
)

summary = {

    "stage":
        "3.31",

    "status":
        "PASS"
        if overall_pass
        else "REVIEW",

    "purpose":
        (
            "Final end-to-end evaluation of the "
            "validated BeltCrack9kd damage "
            "intelligence pipeline."
        ),

    "test_records":
        int(len(test)),

    "test_sequences":
        int(
            test[
                "sequence_key"
            ].nunique()
        ),

    "final_model":
        {
            "name":
                "GradientBoostingRegressor - 33 features",

            "test_mae":
                float(final_mae),

            "test_rmse":
                float(final_rmse),

            "test_r2":
                float(final_r2)
        },

    "persistence_baseline":
        {
            "test_mae":
                float(persistence_mae),

            "test_rmse":
                float(persistence_rmse),

            "test_r2":
                float(persistence_r2)
        },

    "temporal_only_reference":
        {
            "test_mae":
                float(temporal_only_mae),

            "test_rmse":
                float(temporal_only_rmse),

            "test_r2":
                float(temporal_only_r2)
        },

    "improvement":
        {
            "vs_persistence_percent":
                float(
                    improvement_vs_persistence
                ),

            "vs_temporal_only_percent":
                float(
                    improvement_vs_temporal
                )
        },

    "error_distribution":
        error_distribution,

    "direction_accuracy":
        float(
            direction_accuracy
        ),

    "sequence_generalization":
        {
            "sequence_count":
                int(len(sequence_df)),

            "sequences_beating_persistence":
                int(
                    sequences_beating_persistence
                )
        },

    "damage_bin_count":
        int(len(damage_df)),

    "integrity_checks":
        checks,

    "outputs":
        {
            "sequence_evaluation":
                SEQUENCE_OUTPUT,

            "damage_bin_evaluation":
                DAMAGE_BIN_OUTPUT,

            "error_analysis":
                ERROR_OUTPUT,

            "model_comparison":
                MODEL_COMPARISON_OUTPUT,

            "worst_predictions":
                WORST_OUTPUT,

            "best_predictions":
                BEST_OUTPUT
        }
}


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

summary = make_json_safe(
    summary
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    FINAL_SUMMARY,
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
# SAVE ERROR TABLE
# ============================================================

test[
    [
        "sequence_key",
        "frame_key",
        "current_damage",
        "predicted_future_crack_area_ratio",
        "future_crack_area_ratio_t1",
        "absolute_error_recomputed",
        "signed_error",
        "prediction_direction",
        "actual_direction",
        "direction_correct",
        "predicted_damage_status",
        "predicted_damage_level"
    ]
].to_csv(
    ERROR_OUTPUT,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\nFinal summary:",
    FINAL_SUMMARY
)

print(
    "Sequence evaluation:",
    SEQUENCE_OUTPUT
)

print(
    "Damage-bin evaluation:",
    DAMAGE_BIN_OUTPUT
)

print(
    "Error analysis:",
    ERROR_OUTPUT
)

print(
    "Model comparison:",
    MODEL_COMPARISON_OUTPUT
)

print(
    "Worst predictions:",
    WORST_OUTPUT
)

print(
    "Best predictions:",
    BEST_OUTPUT
)

print(
    "\n" + "=" * 80
)

print(
    "STAGE 3.31 STATUS:",
    "PASS" if overall_pass else "REVIEW"
)

print(
    "=" * 80
)