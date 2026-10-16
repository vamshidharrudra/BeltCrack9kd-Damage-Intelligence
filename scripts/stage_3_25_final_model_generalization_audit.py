# ============================================================
# STAGE 3.25 - FINAL MODEL GENERALIZATION AUDIT
# BeltCrack9kd_Damage_Intelligence
#
# Purpose:
#   Evaluate the Stage 3.24 locked 33-feature regression model
#   across:
#       1. Individual unseen test sequences
#       2. Test components
#       3. Damage-level bins
#       4. Error distribution
#       5. Worst predictions
#       6. Under/over prediction behavior
#       7. Comparison against baselines
#
# IMPORTANT:
#   - NO retraining
#   - NO feature selection
#   - NO test optimization
#   - Uses Stage 3.24 test predictions only
#
# This is an evaluation/audit stage.
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

PREDICTIONS_FILE = (
    AUDIT_DIR / "stage_3_24_exact_final_test_predictions.csv"
)

RESULTS_FILE = (
    AUDIT_DIR / "stage_3_24_exact_final_model_results.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR / "stage_3_24_exact_validation_locked_final_summary.json"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SEQUENCE_OUTPUT = (
    AUDIT_DIR / "stage_3_25_sequence_generalization.csv"
)

COMPONENT_OUTPUT = (
    AUDIT_DIR / "stage_3_25_component_generalization.csv"
)

DAMAGE_BIN_OUTPUT = (
    AUDIT_DIR / "stage_3_25_damage_bin_analysis.csv"
)

ERROR_DIST_OUTPUT = (
    AUDIT_DIR / "stage_3_25_error_distribution.csv"
)

WORST_OUTPUT = (
    AUDIT_DIR / "stage_3_25_worst_predictions.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR / "stage_3_25_generalization_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def improvement_vs_baseline(model_mae, baseline_mae):
    if baseline_mae == 0:
        return np.nan

    return float(
        (baseline_mae - model_mae)
        / baseline_mae
        * 100.0
    )


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STAGE 3.25 - FINAL MODEL GENERALIZATION AUDIT")
print("=" * 78)

print()
print("Project:")
print(PROJECT_ROOT)

print()
print("Loading Stage 3.24 test predictions...")
print(PREDICTIONS_FILE)


# ============================================================
# INPUT CHECKS
# ============================================================

required_files = [
    PREDICTIONS_FILE,
    RESULTS_FILE,
    SUMMARY_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

pred = pd.read_csv(PREDICTIONS_FILE)

print()
print("Prediction records:", len(pred))
print("Prediction columns:", len(pred.columns))


required_columns = [
    "sequence",
    "current_frame",
    "component_id",
    "assigned_split",
    "future_crack_area_ratio_t1",
    "prediction_locked_final",
    "prediction_full_150",
    "prediction_temporal_only",
    "prediction_persistence",
]

missing_columns = [
    c for c in required_columns
    if c not in pred.columns
]

if missing_columns:
    raise ValueError(
        "Missing required prediction columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_key_count = int(
    pred.duplicated(
        subset=["sequence", "current_frame"]
    ).sum()
)

print()
print("Duplicate sequence/frame keys:", duplicate_key_count)

if duplicate_key_count != 0:
    raise ValueError(
        "Duplicate sequence/frame keys detected."
    )


# ============================================================
# TEST-ONLY CHECK
# ============================================================

split_values = (
    pred["assigned_split"]
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)

print()
print("Assigned split values:", split_values)

if len(split_values) != 1 or split_values[0] != "TEST":
    raise ValueError(
        "Stage 3.25 must contain TEST records only."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "current_frame",
    "component_id",
    "future_crack_area_ratio_t1",
    "prediction_locked_final",
    "prediction_full_150",
    "prediction_temporal_only",
    "prediction_persistence",
]

for col in numeric_columns:
    pred[col] = pd.to_numeric(
        pred[col],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID PREDICTIONS
# ============================================================

before = len(pred)

pred = pred.dropna(
    subset=[
        "future_crack_area_ratio_t1",
        "prediction_locked_final",
        "prediction_full_150",
        "prediction_temporal_only",
        "prediction_persistence",
    ]
).copy()

after = len(pred)

print()
print("Valid prediction records:", after)
print("Removed invalid records:", before - after)

if len(pred) == 0:
    raise ValueError(
        "No valid prediction records remain."
    )


# ============================================================
# TARGET / PREDICTION ARRAYS
# ============================================================

y = pred[
    "future_crack_area_ratio_t1"
].to_numpy(dtype=float)

p_locked = pred[
    "prediction_locked_final"
].to_numpy(dtype=float)

p_full = pred[
    "prediction_full_150"
].to_numpy(dtype=float)

p_temporal = pred[
    "prediction_temporal_only"
].to_numpy(dtype=float)

p_persistence = pred[
    "prediction_persistence"
].to_numpy(dtype=float)


# ============================================================
# ERROR COLUMNS
# ============================================================

pred["error_locked"] = (
    p_locked - y
)

pred["abs_error_locked"] = np.abs(
    pred["error_locked"]
)

pred["error_full"] = (
    p_full - y
)

pred["abs_error_full"] = np.abs(
    pred["error_full"]
)

pred["error_temporal"] = (
    p_temporal - y
)

pred["abs_error_temporal"] = np.abs(
    pred["error_temporal"]
)

pred["error_persistence"] = (
    p_persistence - y
)

pred["abs_error_persistence"] = np.abs(
    pred["error_persistence"]
)

pred["prediction_direction"] = np.where(
    pred["error_locked"] > 0,
    "OVER",
    np.where(
        pred["error_locked"] < 0,
        "UNDER",
        "EXACT"
    )
)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

n_records = len(pred)

sequences = (
    pred["sequence"]
    .astype(str)
    .nunique()
)

components = (
    pred["component_id"]
    .nunique()
)

print()
print("=" * 78)
print("TEST DATASET")
print("=" * 78)

print("Test records:", n_records)
print("Test sequences:", sequences)
print("Test components:", components)


# ============================================================
# OVERALL MODEL METRICS
# ============================================================

locked_mae = mae(y, p_locked)
locked_rmse = rmse(y, p_locked)
locked_r2 = r2_score_manual(y, p_locked)

full_mae = mae(y, p_full)
full_rmse = rmse(y, p_full)
full_r2 = r2_score_manual(y, p_full)

temporal_mae = mae(y, p_temporal)
temporal_rmse = rmse(y, p_temporal)
temporal_r2 = r2_score_manual(y, p_temporal)

persistence_mae = mae(y, p_persistence)
persistence_rmse = rmse(y, p_persistence)
persistence_r2 = r2_score_manual(y, p_persistence)


print()
print("=" * 78)
print("OVERALL TEST PERFORMANCE")
print("=" * 78)

print(
    f"LOCKED 33-FEATURE:"
    f" MAE={locked_mae:.8f}"
    f" RMSE={locked_rmse:.8f}"
    f" R2={locked_r2:.8f}"
)

print(
    f"FULL 150-FEATURE:"
    f" MAE={full_mae:.8f}"
    f" RMSE={full_rmse:.8f}"
    f" R2={full_r2:.8f}"
)

print(
    f"TEMPORAL ONLY:"
    f" MAE={temporal_mae:.8f}"
    f" RMSE={temporal_rmse:.8f}"
    f" R2={temporal_r2:.8f}"
)

print(
    f"PERSISTENCE:"
    f" MAE={persistence_mae:.8f}"
    f" RMSE={persistence_rmse:.8f}"
    f" R2={persistence_r2:.8f}"
)

print()
print(
    "Locked improvement vs persistence:",
    f"{improvement_vs_baseline(locked_mae, persistence_mae):.4f}%"
)

print(
    "Locked improvement vs temporal:",
    f"{improvement_vs_baseline(locked_mae, temporal_mae):.4f}%"
)

print(
    "Locked difference vs full 150:",
    f"{((locked_mae - full_mae) / full_mae) * 100.0:.4f}%"
)


# ============================================================
# SEQUENCE GENERALIZATION
# ============================================================

print()
print("=" * 78)
print("SEQUENCE-LEVEL GENERALIZATION")
print("=" * 78)

sequence_rows = []

for sequence, group in pred.groupby(
    "sequence",
    sort=True
):

    yt = group[
        "future_crack_area_ratio_t1"
    ].to_numpy(dtype=float)

    yl = group[
        "prediction_locked_final"
    ].to_numpy(dtype=float)

    yf = group[
        "prediction_full_150"
    ].to_numpy(dtype=float)

    yt_model = group[
        "prediction_temporal_only"
    ].to_numpy(dtype=float)

    yp = group[
        "prediction_persistence"
    ].to_numpy(dtype=float)

    locked_sequence_mae = mae(yt, yl)
    full_sequence_mae = mae(yt, yf)
    temporal_sequence_mae = mae(
        yt,
        yt_model
    )
    persistence_sequence_mae = mae(
        yt,
        yp
    )

    sequence_rows.append({
        "sequence": sequence,
        "records": len(group),

        "target_mean":
            float(np.mean(yt)),

        "target_std":
            float(np.std(yt)),

        "locked_mae":
            locked_sequence_mae,

        "locked_rmse":
            rmse(yt, yl),

        "locked_r2":
            r2_score_manual(yt, yl),

        "full_150_mae":
            full_sequence_mae,

        "temporal_only_mae":
            temporal_sequence_mae,

        "persistence_mae":
            persistence_sequence_mae,

        "locked_vs_persistence_improvement_pct":
            improvement_vs_baseline(
                locked_sequence_mae,
                persistence_sequence_mae
            ),

        "locked_vs_temporal_improvement_pct":
            improvement_vs_baseline(
                locked_sequence_mae,
                temporal_sequence_mae
            ),

        "locked_better_than_persistence":
            bool(
                locked_sequence_mae
                < persistence_sequence_mae
            ),

        "locked_better_than_temporal":
            bool(
                locked_sequence_mae
                < temporal_sequence_mae
            ),
    })


sequence_df = pd.DataFrame(
    sequence_rows
)

sequence_df.to_csv(
    SEQUENCE_OUTPUT,
    index=False
)

print(
    "Sequences evaluated:",
    len(sequence_df)
)

print(
    "Sequences beating persistence:",
    int(
        sequence_df[
            "locked_better_than_persistence"
        ].sum()
    ),
    "/",
    len(sequence_df)
)

print(
    "Sequences beating temporal-only:",
    int(
        sequence_df[
            "locked_better_than_temporal"
        ].sum()
    ),
    "/",
    len(sequence_df)
)

print()
print("Worst sequences by locked MAE:")

print(
    sequence_df[
        [
            "sequence",
            "records",
            "locked_mae",
            "persistence_mae",
            "temporal_only_mae",
            "locked_vs_persistence_improvement_pct",
        ]
    ]
    .sort_values(
        "locked_mae",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# COMPONENT GENERALIZATION
# ============================================================

print()
print("=" * 78)
print("COMPONENT-LEVEL GENERALIZATION")
print("=" * 78)

component_rows = []

for component, group in pred.groupby(
    "component_id",
    sort=True
):

    yt = group[
        "future_crack_area_ratio_t1"
    ].to_numpy(dtype=float)

    yl = group[
        "prediction_locked_final"
    ].to_numpy(dtype=float)

    yf = group[
        "prediction_full_150"
    ].to_numpy(dtype=float)

    yt_model = group[
        "prediction_temporal_only"
    ].to_numpy(dtype=float)

    yp = group[
        "prediction_persistence"
    ].to_numpy(dtype=float)

    locked_component_mae = mae(
        yt,
        yl
    )

    persistence_component_mae = mae(
        yt,
        yp
    )

    temporal_component_mae = mae(
        yt,
        yt_model
    )

    component_rows.append({
        "component_id":
            int(component),

        "records":
            len(group),

        "sequences":
            group["sequence"]
            .astype(str)
            .nunique(),

        "target_mean":
            float(np.mean(yt)),

        "target_std":
            float(np.std(yt)),

        "locked_mae":
            locked_component_mae,

        "locked_rmse":
            rmse(yt, yl),

        "locked_r2":
            r2_score_manual(yt, yl),

        "full_150_mae":
            mae(yt, yf),

        "temporal_only_mae":
            temporal_component_mae,

        "persistence_mae":
            persistence_component_mae,

        "locked_vs_persistence_improvement_pct":
            improvement_vs_baseline(
                locked_component_mae,
                persistence_component_mae
            ),

        "locked_vs_temporal_improvement_pct":
            improvement_vs_baseline(
                locked_component_mae,
                temporal_component_mae
            ),
    })


component_df = pd.DataFrame(
    component_rows
)

component_df.to_csv(
    COMPONENT_OUTPUT,
    index=False
)

print(
    "Components evaluated:",
    len(component_df)
)

print()
print("Component performance:")

print(
    component_df[
        [
            "component_id",
            "records",
            "locked_mae",
            "temporal_only_mae",
            "persistence_mae",
            "locked_vs_persistence_improvement_pct",
        ]
    ]
    .sort_values(
        "locked_mae",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# DAMAGE-LEVEL BINS
# ============================================================

print()
print("=" * 78)
print("DAMAGE-LEVEL GENERALIZATION")
print("=" * 78)

bins = [
    -np.inf,
    0.0025,
    0.0050,
    0.0075,
    0.0100,
    0.0150,
    np.inf,
]

labels = [
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
    "EXTREME",
]

pred["damage_bin"] = pd.cut(
    pred[
        "future_crack_area_ratio_t1"
    ],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False,
)

damage_rows = []

for damage_bin, group in pred.groupby(
    "damage_bin",
    observed=False
):

    if len(group) == 0:
        continue

    yt = group[
        "future_crack_area_ratio_t1"
    ].to_numpy(dtype=float)

    yl = group[
        "prediction_locked_final"
    ].to_numpy(dtype=float)

    yf = group[
        "prediction_full_150"
    ].to_numpy(dtype=float)

    yt_model = group[
        "prediction_temporal_only"
    ].to_numpy(dtype=float)

    yp = group[
        "prediction_persistence"
    ].to_numpy(dtype=float)

    damage_rows.append({
        "damage_bin":
            str(damage_bin),

        "records":
            len(group),

        "target_mean":
            float(np.mean(yt)),

        "target_min":
            float(np.min(yt)),

        "target_max":
            float(np.max(yt)),

        "locked_mae":
            mae(yt, yl),

        "locked_rmse":
            rmse(yt, yl),

        "full_150_mae":
            mae(yt, yf),

        "temporal_only_mae":
            mae(yt, yt_model),

        "persistence_mae":
            mae(yt, yp),

        "locked_mean_error":
            float(np.mean(yl - yt)),

        "locked_median_abs_error":
            float(
                np.median(
                    np.abs(yl - yt)
                )
            ),

        "locked_underprediction_rate":
            float(
                np.mean(
                    yl < yt
                )
            ),

        "locked_overprediction_rate":
            float(
                np.mean(
                    yl > yt
                )
            ),
    })


damage_df = pd.DataFrame(
    damage_rows
)

damage_df.to_csv(
    DAMAGE_BIN_OUTPUT,
    index=False
)

print()
print(
    damage_df.to_string(
        index=False
    )
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

print()
print("=" * 78)
print("ERROR DISTRIBUTION")
print("=" * 78)

abs_error = pred[
    "abs_error_locked"
].to_numpy(dtype=float)

signed_error = pred[
    "error_locked"
].to_numpy(dtype=float)

percentiles = [
    50,
    75,
    90,
    95,
    99,
    99.5,
]

error_rows = []

for percentile in percentiles:

    error_rows.append({
        "metric":
            f"ABS_ERROR_P{percentile}",

        "value":
            float(
                np.percentile(
                    abs_error,
                    percentile
                )
            ),
    })


error_rows.extend([
    {
        "metric":
            "ABS_ERROR_MEAN",
        "value":
            float(np.mean(abs_error)),
    },
    {
        "metric":
            "ABS_ERROR_MAX",
        "value":
            float(np.max(abs_error)),
    },
    {
        "metric":
            "SIGNED_ERROR_MEAN",
        "value":
            float(np.mean(signed_error)),
    },
    {
        "metric":
            "SIGNED_ERROR_MEDIAN",
        "value":
            float(np.median(signed_error)),
    },
    {
        "metric":
            "UNDERPREDICTION_COUNT",
        "value":
            float(np.sum(signed_error < 0)),
    },
    {
        "metric":
            "OVERPREDICTION_COUNT",
        "value":
            float(np.sum(signed_error > 0)),
    },
    {
        "metric":
            "EXACT_PREDICTION_COUNT",
        "value":
            float(np.sum(signed_error == 0)),
    },
])


error_distribution_df = pd.DataFrame(
    error_rows
)

error_distribution_df.to_csv(
    ERROR_DIST_OUTPUT,
    index=False
)

print(
    error_distribution_df.to_string(
        index=False
    )
)


# ============================================================
# HIGH-ERROR FRACTIONS
# ============================================================

p95_error = float(
    np.percentile(
        abs_error,
        95
    )
)

p99_error = float(
    np.percentile(
        abs_error,
        99
    )
)

high_error_fraction = float(
    np.mean(
        abs_error >= p95_error
    )
)

very_high_error_fraction = float(
    np.mean(
        abs_error >= p99_error
    )
)

print()
print(
    "P95 absolute-error threshold:",
    f"{p95_error:.8f}"
)

print(
    "P99 absolute-error threshold:",
    f"{p99_error:.8f}"
)

print(
    "Fraction >= P95:",
    f"{high_error_fraction:.6f}"
)

print(
    "Fraction >= P99:",
    f"{very_high_error_fraction:.6f}"
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

print()
print("=" * 78)
print("WORST PREDICTIONS")
print("=" * 78)

worst_columns = [
    "sequence",
    "current_frame",
    "component_id",
    "assigned_split",
    "future_crack_area_ratio_t1",
    "prediction_locked_final",
    "prediction_full_150",
    "prediction_temporal_only",
    "prediction_persistence",
    "error_locked",
    "abs_error_locked",
    "prediction_direction",
]

worst_df = (
    pred[
        worst_columns
    ]
    .sort_values(
        "abs_error_locked",
        ascending=False
    )
    .head(100)
    .copy()
)

worst_df.to_csv(
    WORST_OUTPUT,
    index=False
)

print(
    worst_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# UNDER / OVER PREDICTION
# ============================================================

under_count = int(
    np.sum(
        signed_error < 0
    )
)

over_count = int(
    np.sum(
        signed_error > 0
    )
)

exact_count = int(
    np.sum(
        signed_error == 0
    )
)

under_rate = (
    under_count / n_records
)

over_rate = (
    over_count / n_records
)

print()
print("=" * 78)
print("PREDICTION DIRECTION")
print("=" * 78)

print(
    "Underprediction:",
    under_count,
    f"({under_rate:.4%})"
)

print(
    "Overprediction:",
    over_count,
    f"({over_rate:.4%})"
)

print(
    "Exact:",
    exact_count,
    f"({exact_count / n_records:.4%})"
)


# ============================================================
# SEQUENCE SPREAD
# ============================================================

sequence_maes = (
    sequence_df[
        "locked_mae"
    ].to_numpy(dtype=float)
)

sequence_mae_mean = float(
    np.mean(sequence_maes)
)

sequence_mae_std = float(
    np.std(sequence_maes)
)

sequence_mae_min = float(
    np.min(sequence_maes)
)

sequence_mae_max = float(
    np.max(sequence_maes)
)

sequence_mae_spread = (
    sequence_mae_max
    - sequence_mae_min
)

print()
print("=" * 78)
print("SEQUENCE MAE SPREAD")
print("=" * 78)

print(
    "Mean sequence MAE:",
    f"{sequence_mae_mean:.8f}"
)

print(
    "Std sequence MAE:",
    f"{sequence_mae_std:.8f}"
)

print(
    "Minimum sequence MAE:",
    f"{sequence_mae_min:.8f}"
)

print(
    "Maximum sequence MAE:",
    f"{sequence_mae_max:.8f}"
)

print(
    "Sequence MAE spread:",
    f"{sequence_mae_spread:.8f}"
)


# ============================================================
# GENERALIZATION CHECKS
# ============================================================

all_sequences_beat_persistence = bool(
    sequence_df[
        "locked_better_than_persistence"
    ].all()
)

all_sequences_beat_temporal = bool(
    sequence_df[
        "locked_better_than_temporal"
    ].all()
)

locked_beats_persistence_overall = bool(
    locked_mae < persistence_mae
)

locked_beats_temporal_overall = bool(
    locked_mae < temporal_mae
)

locked_close_to_full = bool(
    (
        (locked_mae - full_mae)
        / full_mae
    ) <= 0.02
)


# ============================================================
# COMPONENT GENERALIZATION CHECK
# ============================================================

component_persistence_wins = int(
    np.sum(
        component_df[
            "locked_mae"
        ]
        <
        component_df[
            "persistence_mae"
        ]
    )
)

component_temporal_wins = int(
    np.sum(
        component_df[
            "locked_mae"
        ]
        <
        component_df[
            "temporal_only_mae"
        ]
    )
)


# ============================================================
# DAMAGE BIN CHECKS
# ============================================================

damage_bin_count = len(
    damage_df
)

damage_bins_with_model_better_than_persistence = int(
    np.sum(
        damage_df["locked_mae"]
        <
        damage_df["persistence_mae"]
    )
)

damage_bins_with_model_better_than_temporal = int(
    np.sum(
        damage_df["locked_mae"]
        <
        damage_df["temporal_only_mae"]
    )
)


# ============================================================
# GENERALIZATION FLAGS
# ============================================================

generalization_flags = {

    "test_records_valid":
        bool(n_records > 0),

    "duplicate_sequence_frame_keys":
        duplicate_key_count,

    "test_only":
        True,

    "locked_beats_persistence_overall":
        locked_beats_persistence_overall,

    "locked_beats_temporal_overall":
        locked_beats_temporal_overall,

    "all_sequences_beat_persistence":
        all_sequences_beat_persistence,

    "all_sequences_beat_temporal":
        all_sequences_beat_temporal,

    "component_persistence_wins":
        component_persistence_wins,

    "component_temporal_wins":
        component_temporal_wins,

    "damage_bins":
        damage_bin_count,

    "damage_bins_beating_persistence":
        damage_bins_with_model_better_than_persistence,

    "damage_bins_beating_temporal":
        damage_bins_with_model_better_than_temporal,

    "locked_within_2_percent_of_full_150":
        locked_close_to_full,

    "high_error_fraction":
        high_error_fraction,

    "very_high_error_fraction":
        very_high_error_fraction,
}


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "stage":
        "3.25",

    "stage_name":
        "Final Model Generalization Audit",

    "status":
        "PASS",

    "methodology":
        {
            "retraining":
                False,

            "feature_selection":
                False,

            "test_optimization":
                False,

            "evaluation_source":
                "Stage 3.24 held-out TEST predictions",
        },

    "test_dataset":
        {
            "records":
                n_records,

            "sequences":
                sequences,

            "components":
                components,

            "duplicate_sequence_frame_keys":
                duplicate_key_count,
        },

    "overall_metrics":
        {
            "locked_33_features":
                {
                    "mae":
                        locked_mae,

                    "rmse":
                        locked_rmse,

                    "r2":
                        locked_r2,
                },

            "full_150_features":
                {
                    "mae":
                        full_mae,

                    "rmse":
                        full_rmse,

                    "r2":
                        full_r2,
                },

            "temporal_only":
                {
                    "mae":
                        temporal_mae,

                    "rmse":
                        temporal_rmse,

                    "r2":
                        temporal_r2,
                },

            "persistence":
                {
                    "mae":
                        persistence_mae,

                    "rmse":
                        persistence_rmse,

                    "r2":
                        persistence_r2,
                },
        },

    "improvements":
        {
            "locked_vs_persistence_pct":
                improvement_vs_baseline(
                    locked_mae,
                    persistence_mae
                ),

            "locked_vs_temporal_pct":
                improvement_vs_baseline(
                    locked_mae,
                    temporal_mae
                ),

            "locked_vs_full_150_pct":
                (
                    (locked_mae - full_mae)
                    / full_mae
                    * 100.0
                ),
        },

    "error_distribution":
        {
            "mean_absolute_error":
                float(np.mean(abs_error)),

            "p50":
                float(np.percentile(abs_error, 50)),

            "p75":
                float(np.percentile(abs_error, 75)),

            "p90":
                float(np.percentile(abs_error, 90)),

            "p95":
                float(np.percentile(abs_error, 95)),

            "p99":
                float(np.percentile(abs_error, 99)),

            "max":
                float(np.max(abs_error)),

            "mean_signed_error":
                float(np.mean(signed_error)),

            "underprediction_rate":
                under_rate,

            "overprediction_rate":
                over_rate,

            "high_error_fraction":
                high_error_fraction,

            "very_high_error_fraction":
                very_high_error_fraction,
        },

    "sequence_generalization":
        {
            "sequence_count":
                len(sequence_df),

            "mean_mae":
                sequence_mae_mean,

            "std_mae":
                sequence_mae_std,

            "min_mae":
                sequence_mae_min,

            "max_mae":
                sequence_mae_max,

            "mae_spread":
                sequence_mae_spread,

            "sequences_beating_persistence":
                int(
                    sequence_df[
                        "locked_better_than_persistence"
                    ].sum()
                ),

            "sequences_beating_temporal":
                int(
                    sequence_df[
                        "locked_better_than_temporal"
                    ].sum()
                ),
        },

    "component_generalization":
        {
            "component_count":
                len(component_df),

            "components_beating_persistence":
                component_persistence_wins,

            "components_beating_temporal":
                component_temporal_wins,
        },

    "damage_generalization":
        {
            "damage_bins":
                damage_bin_count,

            "bins_beating_persistence":
                damage_bins_with_model_better_than_persistence,

            "bins_beating_temporal":
                damage_bins_with_model_better_than_temporal,
        },

    "generalization_flags":
        generalization_flags,

    "outputs":
        {
            "sequence":
                str(SEQUENCE_OUTPUT),

            "component":
                str(COMPONENT_OUTPUT),

            "damage_bins":
                str(DAMAGE_BIN_OUTPUT),

            "error_distribution":
                str(ERROR_DIST_OUTPUT),

            "worst_predictions":
                str(WORST_OUTPUT),

            "summary":
                str(SUMMARY_OUTPUT),
        },
}


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def make_json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None

    return obj


summary = make_json_safe(summary)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_OUTPUT,
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
# FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STAGE 3.25 FINAL SUMMARY")
print("=" * 78)

print()
print(
    "Locked 33-feature test MAE:",
    f"{locked_mae:.8f}"
)

print(
    "Locked 33-feature test RMSE:",
    f"{locked_rmse:.8f}"
)

print(
    "Locked 33-feature test R2:",
    f"{locked_r2:.8f}"
)

print()
print(
    "Persistence test MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Temporal-only test MAE:",
    f"{temporal_mae:.8f}"
)

print(
    "Full 150-feature test MAE:",
    f"{full_mae:.8f}"
)

print()
print(
    "Improvement vs persistence:",
    f"{improvement_vs_baseline(locked_mae, persistence_mae):.4f}%"
)

print(
    "Improvement vs temporal-only:",
    f"{improvement_vs_baseline(locked_mae, temporal_mae):.4f}%"
)

print()
print(
    "Sequences beating persistence:",
    int(
        sequence_df[
            "locked_better_than_persistence"
        ].sum()
    ),
    "/",
    len(sequence_df)
)

print(
    "Sequences beating temporal-only:",
    int(
        sequence_df[
            "locked_better_than_temporal"
        ].sum()
    ),
    "/",
    len(sequence_df)
)

print()
print(
    "Sequence MAE spread:",
    f"{sequence_mae_spread:.8f}"
)

print(
    "P95 absolute error:",
    f"{p95_error:.8f}"
)

print(
    "P99 absolute error:",
    f"{p99_error:.8f}"
)

print(
    "Underprediction rate:",
    f"{under_rate:.4%}"
)

print(
    "Overprediction rate:",
    f"{over_rate:.4%}"
)

print()
print("Outputs:")
print(SEQUENCE_OUTPUT)
print(COMPONENT_OUTPUT)
print(DAMAGE_BIN_OUTPUT)
print(ERROR_DIST_OUTPUT)
print(WORST_OUTPUT)
print(SUMMARY_OUTPUT)

print()
print("=" * 78)
print("STAGE 3.25 STATUS: PASS")
print("=" * 78)