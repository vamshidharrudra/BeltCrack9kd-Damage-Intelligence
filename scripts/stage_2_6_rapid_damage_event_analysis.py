import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

TEMPORAL_FEATURES = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

TARGETS = os.path.join(
    AUDIT_DIR,
    "stage_1_3_prediction_targets.csv"
)

OUTPUT_CASES = os.path.join(
    AUDIT_DIR,
    "stage_2_6_rapid_damage_cases.csv"
)

OUTPUT_SEQUENCE = os.path.join(
    AUDIT_DIR,
    "stage_2_6_sequence_rapid_damage_analysis.csv"
)

OUTPUT_CHANGE = os.path.join(
    AUDIT_DIR,
    "stage_2_6_change_feature_analysis.csv"
)

OUTPUT_ERROR = os.path.join(
    AUDIT_DIR,
    "stage_2_6_error_change_analysis.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_2_6_rapid_damage_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def safe_rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def classify_change(value):

    value = abs(float(value))

    if value < 0.001:
        return "LOW_CHANGE"

    elif value < 0.005:
        return "MEDIUM_CHANGE"

    else:
        return "HIGH_CHANGE"


def classify_relative_change(value):

    value = abs(float(value))

    if value < 0.10:
        return "LOW_RELATIVE_CHANGE"

    elif value < 0.50:
        return "MEDIUM_RELATIVE_CHANGE"

    else:
        return "HIGH_RELATIVE_CHANGE"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 2.6 - RAPID DAMAGE EVENT ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD STAGE 2.2
# ============================================================

print()
print("Loading Stage 2.2 temporal features...")

temporal = pd.read_csv(
    TEMPORAL_FEATURES
)

print(
    f"Temporal records: {len(temporal)}"
)


# ============================================================
# LOAD STAGE 1.3
# ============================================================

print()
print("Loading Stage 1.3 targets...")

targets = pd.read_csv(
    TARGETS
)

print(
    f"Target records: {len(targets)}"
)


# ============================================================
# SELECT PRIMARY TARGET
# ============================================================

primary_target = targets[
    targets["horizon"] == 1
].copy()

print(
    f"Primary t+1 target records: "
    f"{len(primary_target)}"
)


# ============================================================
# STANDARDIZE IDENTIFIERS
# ============================================================

temporal["sequence"] = (
    temporal["sequence"].astype(str)
)

primary_target["sequence"] = (
    primary_target["sequence"].astype(str)
)

temporal["component"] = (
    temporal["component"].astype(int)
)

primary_target["component"] = (
    primary_target["component"].astype(int)
)

temporal["frame_number"] = (
    pd.to_numeric(
        temporal["frame_number"],
        errors="raise"
    ).astype(int)
)

primary_target["current_frame"] = (
    pd.to_numeric(
        primary_target["current_frame"],
        errors="raise"
    ).astype(int)
)

primary_target["future_frame"] = (
    pd.to_numeric(
        primary_target["future_frame"],
        errors="raise"
    ).astype(int)
)


# ============================================================
# VERIFY STAGE 2.2 TARGET
# ============================================================

if "target_t_plus_1" not in temporal.columns:

    raise ValueError(
        "Stage 2.2 temporal features do not contain "
        "'target_t_plus_1'."
    )


# ============================================================
# VERIFY STAGE 2.2 TARGET AGAINST STAGE 1.3
# ============================================================

print()
print("=" * 70)
print("VERIFYING STAGE 2.2 TARGET AGAINST STAGE 1.3")
print("=" * 70)

target_reference = primary_target[
    [
        "sequence",
        "component",
        "current_frame",
        "future_frame",
        "target_crack_area_ratio"
    ]
].copy()

target_reference = target_reference.rename(
    columns={
        "current_frame": "frame_number",
        "target_crack_area_ratio":
            "verified_future_damage"
    }
)

verification = temporal.merge(
    target_reference,
    on=[
        "sequence",
        "component",
        "frame_number"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    f"Matched target records: "
    f"{len(verification)}"
)

if len(verification) != len(temporal):

    raise ValueError(
        "Stage 2.2 and Stage 1.3 target records "
        "could not be matched one-to-one."
    )

verification["target_difference"] = (
    verification["target_t_plus_1"]
    -
    verification["verified_future_damage"]
).abs()

max_target_difference = float(
    verification[
        "target_difference"
    ].max()
)

print(
    f"Maximum target difference: "
    f"{max_target_difference:.12f}"
)

if max_target_difference > 1e-12:

    raise ValueError(
        "Stage 2.2 target_t_plus_1 does not match "
        "Stage 1.3 target_crack_area_ratio."
    )

print(
    "Target verification: PASS"
)


# ============================================================
# MODEL DATA
# ============================================================

data = temporal.copy()

data["future_damage"] = (
    data["target_t_plus_1"]
)


# ============================================================
# VERIFY TARGET FRAME
# ============================================================

if "target_current_frame" in data.columns:

    frame_difference = (
        data["target_current_frame"]
        -
        data["frame_number"]
    ).abs()

    max_frame_difference = int(
        frame_difference.max()
    )

    print(
        f"Maximum current-frame mapping difference: "
        f"{max_frame_difference}"
    )

    if max_frame_difference != 0:

        raise ValueError(
            "target_current_frame does not match "
            "frame_number."
        )


if "target_future_frame" in data.columns:

    future_difference = (
        data["target_future_frame"]
        -
        data["frame_number"]
    )

    invalid_future = (
        future_difference != 1
    )

    invalid_count = int(
        invalid_future.sum()
    )

    print(
        f"Invalid t+1 frame mappings: "
        f"{invalid_count}"
    )

    if invalid_count != 0:

        raise ValueError(
            "Invalid t+1 frame mapping detected."
        )


# ============================================================
# IDENTIFY MODEL FEATURES
# ============================================================

exclude_columns = {
    "sequence",
    "component",
    "frame_number",
    "record_id",
    "split",
    "future_damage",
    "target_t_plus_1",
    "target_current_frame",
    "target_future_frame"
}


numeric_columns = data.select_dtypes(
    include=[np.number]
).columns.tolist()


feature_columns = [
    column
    for column in numeric_columns
    if column not in exclude_columns
]


# Remove future/target information.
future_keywords = [
    "future",
    "target",
    "t_plus",
    "tplus"
]

feature_columns = [
    column
    for column in feature_columns
    if not any(
        keyword in column.lower()
        for keyword in future_keywords
    )
]


# Remove constant columns.
feature_columns = [
    column
    for column in feature_columns
    if data[column].nunique(
        dropna=False
    ) > 1
]


print()
print(
    f"Candidate model features: "
    f"{len(feature_columns)}"
)


# ============================================================
# COMPLETE MODEL DATA
# ============================================================

model_data = data.dropna(
    subset=
    feature_columns
    +
    ["future_damage"]
).copy()

print(
    f"Complete model records: "
    f"{len(model_data)}"
)


# ============================================================
# TRAIN / TEST
# ============================================================

train = model_data[
    model_data["split"] == "train"
].copy()

test = model_data[
    model_data["split"] == "test"
].copy()

print(
    f"TRAIN: {len(train)}"
)

print(
    f"TEST: {len(test)}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING GRADIENT BOOSTING MODEL")
print("=" * 70)

X_train = train[
    feature_columns
]

y_train = train[
    "future_damage"
]

X_test = test[
    feature_columns
]

y_test = test[
    "future_damage"
]

model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=5,
    loss="squared_error"
)

model.fit(
    X_train,
    y_train
)

test["model_prediction"] = (
    model.predict(X_test)
)

test["model_prediction"] = (
    test["model_prediction"]
    .clip(lower=0)
)


# ============================================================
# PERSISTENCE
# ============================================================

test["persistence_prediction"] = (
    test["crack_area_ratio"]
)


# ============================================================
# BASIC ERROR
# ============================================================

test["model_error"] = (
    test["model_prediction"]
    -
    test["future_damage"]
)

test["absolute_error"] = (
    test["model_error"].abs()
)

test["persistence_error"] = (
    test["persistence_prediction"]
    -
    test["future_damage"]
)

test["persistence_absolute_error"] = (
    test["persistence_error"].abs()
)


# ============================================================
# ACTUAL DAMAGE CHANGE
# ============================================================

test["actual_damage_change"] = (
    test["future_damage"]
    -
    test["crack_area_ratio"]
)

test["absolute_damage_change"] = (
    test["actual_damage_change"].abs()
)


# ============================================================
# PREDICTED DAMAGE CHANGE
# ============================================================

test["predicted_damage_change"] = (
    test["model_prediction"]
    -
    test["crack_area_ratio"]
)

test["change_prediction_error"] = (
    test["predicted_damage_change"]
    -
    test["actual_damage_change"]
).abs()


# ============================================================
# ERROR DIRECTION
# ============================================================

test["error_direction"] = np.where(
    test["model_error"] > 0,
    "OVERPREDICT",
    "UNDERPREDICT"
)

test.loc[
    test["model_error"] == 0,
    "error_direction"
] = "EXACT"


# ============================================================
# CHANGE LEVEL
# ============================================================

test["change_level"] = (
    test["absolute_damage_change"]
    .apply(
        classify_change
    )
)


# ============================================================
# RELATIVE DAMAGE CHANGE
# ============================================================

epsilon = 1e-8

test["relative_damage_change"] = (
    test["absolute_damage_change"]
    /
    (
        test["crack_area_ratio"].abs()
        +
        epsilon
    )
)

test["relative_change_level"] = (
    test["relative_damage_change"]
    .apply(
        classify_relative_change
    )
)


# ============================================================
# CRACK COUNT CHANGE
# ============================================================

if "delta1_crack_count" in test.columns:

    test["crack_count_change"] = (
        test["delta1_crack_count"]
    )

else:

    test["crack_count_change"] = np.nan


# ============================================================
# LARGEST CRACK CHANGE
# ============================================================

if (
    "delta1_largest_crack_area_ratio"
    in test.columns
):

    test["largest_crack_change"] = (
        test[
            "delta1_largest_crack_area_ratio"
        ]
    )

else:

    test["largest_crack_change"] = np.nan


# ============================================================
# CENTROID MOVEMENT
# ============================================================

centroid_x = (
    "mean_crack_centroid_x_norm"
)

centroid_y = (
    "mean_crack_centroid_y_norm"
)


if (
    centroid_x in test.columns
    and
    centroid_y in test.columns
):

    dx = test[
        centroid_x
    ].diff()

    dy = test[
        centroid_y
    ].diff()

    test["centroid_movement"] = (
        np.sqrt(
            dx ** 2
            +
            dy ** 2
        )
    )

else:

    test["centroid_movement"] = np.nan


# ============================================================
# LARGE ERROR THRESHOLD
# ============================================================

error_threshold = (
    test[
        "absolute_error"
    ].quantile(0.95)
)

test["large_error"] = (
    test["absolute_error"]
    >=
    error_threshold
)


print()
print(
    f"95th percentile absolute error: "
    f"{error_threshold:.8f}"
)

print(
    f"Large-error records: "
    f"{int(test['large_error'].sum())}"
)


# ============================================================
# RAPID DAMAGE THRESHOLD
# ============================================================

rapid_threshold = (
    test[
        "absolute_damage_change"
    ].quantile(0.95)
)

test["rapid_damage_event"] = (
    test[
        "absolute_damage_change"
    ]
    >=
    rapid_threshold
)


print()
print(
    f"95th percentile damage-change threshold: "
    f"{rapid_threshold:.8f}"
)

print(
    f"Rapid damage events: "
    f"{int(test['rapid_damage_event'].sum())}"
)


# ============================================================
# INTERSECTION
# ============================================================

test[
    "large_error_and_rapid_change"
] = (
    test["large_error"]
    &
    test["rapid_damage_event"]
)

intersection_count = int(
    test[
        "large_error_and_rapid_change"
    ].sum()
)

print(
    f"Large-error + rapid-change cases: "
    f"{intersection_count}"
)


# ============================================================
# SAVE PREDICTION CASES
# ============================================================

case_columns = [
    "sequence",
    "component",
    "frame_number",
    "crack_area_ratio",
    "future_damage",
    "model_prediction",
    "absolute_error",
    "persistence_absolute_error",
    "actual_damage_change",
    "predicted_damage_change",
    "absolute_damage_change",
    "change_prediction_error",
    "change_level",
    "relative_damage_change",
    "relative_change_level",
    "crack_count_change",
    "largest_crack_change",
    "centroid_movement",
    "error_direction",
    "large_error",
    "rapid_damage_event",
    "large_error_and_rapid_change"
]

case_columns = [
    column
    for column in case_columns
    if column in test.columns
]

cases = test[
    case_columns
].sort_values(
    "absolute_error",
    ascending=False
)

cases.to_csv(
    OUTPUT_CASES,
    index=False
)


# ============================================================
# SEQUENCE ANALYSIS
# ============================================================

sequence_rows = []

for sequence, group in test.groupby(
    "sequence",
    sort=True
):

    model_mae = mean_absolute_error(
        group["future_damage"],
        group["model_prediction"]
    )

    persistence_mae = mean_absolute_error(
        group["future_damage"],
        group["persistence_prediction"]
    )

    sequence_rows.append({

        "sequence": sequence,

        "component": int(
            group["component"].iloc[0]
        ),

        "records": int(
            len(group)
        ),

        "MAE": float(
            model_mae
        ),

        "RMSE": safe_rmse(
            group["future_damage"],
            group["model_prediction"]
        ),

        "R2": float(
            r2_score(
                group["future_damage"],
                group["model_prediction"]
            )
        ),

        "persistence_MAE": float(
            persistence_mae
        ),

        "improvement_vs_persistence_percent":
            float(
                (
                    persistence_mae
                    -
                    model_mae
                )
                /
                persistence_mae
                *
                100
            ),

        "mean_absolute_damage_change":
            float(
                group[
                    "absolute_damage_change"
                ].mean()
            ),

        "median_absolute_damage_change":
            float(
                group[
                    "absolute_damage_change"
                ].median()
            ),

        "rapid_event_fraction":
            float(
                group[
                    "rapid_damage_event"
                ].mean()
            ),

        "large_error_fraction":
            float(
                group[
                    "large_error"
                ].mean()
            ),

        "large_error_rapid_fraction":
            float(
                group[
                    "large_error_and_rapid_change"
                ].mean()
            )
    })


sequence_df = pd.DataFrame(
    sequence_rows
)

sequence_df.to_csv(
    OUTPUT_SEQUENCE,
    index=False
)


# ============================================================
# CHANGE LEVEL ANALYSIS
# ============================================================

change_rows = []

for level, group in test.groupby(
    "change_level",
    sort=True
):

    model_mae = mean_absolute_error(
        group["future_damage"],
        group["model_prediction"]
    )

    persistence_mae = mean_absolute_error(
        group["future_damage"],
        group["persistence_prediction"]
    )

    change_rows.append({

        "change_level": level,

        "records": int(
            len(group)
        ),

        "fraction": float(
            len(group)
            /
            len(test)
        ),

        "model_MAE": float(
            model_mae
        ),

        "persistence_MAE": float(
            persistence_mae
        ),

        "improvement_vs_persistence_percent":
            float(
                (
                    persistence_mae
                    -
                    model_mae
                )
                /
                persistence_mae
                *
                100
            ),

        "mean_absolute_change":
            float(
                group[
                    "absolute_damage_change"
                ].mean()
            ),

        "large_error_fraction":
            float(
                group[
                    "large_error"
                ].mean()
            )
    })


change_df = pd.DataFrame(
    change_rows
)

change_df.to_csv(
    OUTPUT_CHANGE,
    index=False
)


# ============================================================
# RELATIVE CHANGE ERROR ANALYSIS
# ============================================================

error_rows = []

for level, group in test.groupby(
    "relative_change_level",
    sort=True
):

    error_rows.append({

        "relative_change_level":
            level,

        "records":
            int(len(group)),

        "mean_absolute_error":
            float(
                group[
                    "absolute_error"
                ].mean()
            ),

        "median_absolute_error":
            float(
                group[
                    "absolute_error"
                ].median()
            ),

        "mean_absolute_damage_change":
            float(
                group[
                    "absolute_damage_change"
                ].mean()
            ),

        "large_error_fraction":
            float(
                group[
                    "large_error"
                ].mean()
            ),

        "underprediction_fraction":
            float(
                (
                    group[
                        "error_direction"
                    ]
                    ==
                    "UNDERPREDICT"
                ).mean()
            ),

        "overprediction_fraction":
            float(
                (
                    group[
                        "error_direction"
                    ]
                    ==
                    "OVERPREDICT"
                ).mean()
            )
    })


error_change_df = pd.DataFrame(
    error_rows
)

error_change_df.to_csv(
    OUTPUT_ERROR,
    index=False
)


# ============================================================
# TOP RAPID DAMAGE EVENTS
# ============================================================

top_rapid = test.sort_values(
    "absolute_damage_change",
    ascending=False
).head(20)


print()
print("=" * 70)
print("TOP RAPID DAMAGE EVENTS")
print("=" * 70)


display_columns = [
    "sequence",
    "frame_number",
    "crack_area_ratio",
    "future_damage",
    "absolute_damage_change",
    "model_prediction",
    "absolute_error",
    "error_direction"
]


print(
    top_rapid[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# TOP LARGE ERRORS
# ============================================================

top_errors = test.sort_values(
    "absolute_error",
    ascending=False
).head(20)


print()
print("=" * 70)
print("TOP LARGE ERRORS")
print("=" * 70)


print(
    top_errors[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# CORRELATIONS
# ============================================================

correlations = {}

candidate_change_columns = [
    "absolute_damage_change",
    "crack_count_change",
    "largest_crack_change",
    "centroid_movement",
    "relative_damage_change"
]


for column in candidate_change_columns:

    if column not in test.columns:
        continue

    valid = test[
        [
            column,
            "absolute_error"
        ]
    ].dropna()

    if len(valid) < 3:

        correlations[column] = None

        continue

    correlations[column] = float(
        valid[
            column
        ].corr(
            valid[
                "absolute_error"
            ]
        )
    )


# ============================================================
# OVERALL METRICS
# ============================================================

model_mae = mean_absolute_error(
    y_test,
    test["model_prediction"]
)

model_rmse = safe_rmse(
    y_test,
    test["model_prediction"]
)

model_r2 = r2_score(
    y_test,
    test["model_prediction"]
)

persistence_mae = mean_absolute_error(
    y_test,
    test["persistence_prediction"]
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage": "2.6",

    "status": "PASS",

    "target_verification": {

        "stage_2_2_records":
            int(len(temporal)),

        "stage_1_3_primary_records":
            int(len(primary_target)),

        "matched_records":
            int(len(verification)),

        "maximum_target_difference":
            max_target_difference
    },

    "test_records":
        int(len(test)),

    "model": {

        "name":
            "GradientBoosting",

        "MAE":
            float(model_mae),

        "RMSE":
            float(model_rmse),

        "R2":
            float(model_r2)
    },

    "persistence": {

        "MAE":
            float(persistence_mae)
    },

    "model_vs_persistence_improvement_percent":
        float(
            (
                persistence_mae
                -
                model_mae
            )
            /
            persistence_mae
            *
            100
        ),

    "large_error": {

        "threshold_95_percent":
            float(error_threshold),

        "records":
            int(
                test[
                    "large_error"
                ].sum()
            ),

        "fraction":
            float(
                test[
                    "large_error"
                ].mean()
            )
    },

    "rapid_damage": {

        "threshold_95_percent":
            float(rapid_threshold),

        "records":
            int(
                test[
                    "rapid_damage_event"
                ].sum()
            ),

        "fraction":
            float(
                test[
                    "rapid_damage_event"
                ].mean()
            )
    },

    "large_error_and_rapid_damage": {

        "records":
            intersection_count,

        "fraction":
            float(
                test[
                    "large_error_and_rapid_change"
                ].mean()
            )
    },

    "error_direction": {

        "underprediction_fraction":
            float(
                (
                    test[
                        "error_direction"
                    ]
                    ==
                    "UNDERPREDICT"
                ).mean()
            ),

        "overprediction_fraction":
            float(
                (
                    test[
                        "error_direction"
                    ]
                    ==
                    "OVERPREDICT"
                ).mean()
            )
    },

    "change_correlations_with_absolute_error":
        correlations,

    "best_sequence":
        sequence_df.loc[
            sequence_df["MAE"].idxmin(),
            "sequence"
        ],

    "worst_sequence":
        sequence_df.loc[
            sequence_df["MAE"].idxmax(),
            "sequence"
        ],

    "output_files": {

        "cases":
            OUTPUT_CASES,

        "sequence_analysis":
            OUTPUT_SEQUENCE,

        "change_analysis":
            OUTPUT_CHANGE,

        "error_change_analysis":
            OUTPUT_ERROR,

        "summary":
            OUTPUT_SUMMARY
    }
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STAGE 2.6 SUMMARY")
print("=" * 70)

print(
    f"GradientBoosting MAE: "
    f"{model_mae:.8f}"
)

print(
    f"GradientBoosting RMSE: "
    f"{model_rmse:.8f}"
)

print(
    f"GradientBoosting R2: "
    f"{model_r2:.8f}"
)

print(
    f"Persistence MAE: "
    f"{persistence_mae:.8f}"
)

print(
    "Model vs persistence improvement: "
    f"{((persistence_mae - model_mae) / persistence_mae) * 100:.2f}%"
)

print(
    f"95th percentile error threshold: "
    f"{error_threshold:.8f}"
)

print(
    f"Large-error records: "
    f"{int(test['large_error'].sum())}"
)

print(
    f"95th percentile damage-change threshold: "
    f"{rapid_threshold:.8f}"
)

print(
    f"Rapid damage events: "
    f"{int(test['rapid_damage_event'].sum())}"
)

print(
    f"Large-error + rapid-change cases: "
    f"{intersection_count}"
)

print()
print("Correlation with absolute prediction error:")

for key, value in correlations.items():

    if value is None or pd.isna(value):

        print(
            f"  {key}: NaN"
        )

    else:

        print(
            f"  {key}: {value:.6f}"
        )


print()
print(
    f"Best sequence: "
    f"{summary['best_sequence']}"
)

print(
    f"Worst sequence: "
    f"{summary['worst_sequence']}"
)

print()
print("Output files:")

print(OUTPUT_CASES)
print(OUTPUT_SEQUENCE)
print(OUTPUT_CHANGE)
print(OUTPUT_ERROR)
print(OUTPUT_SUMMARY)

print()
print("FINAL STATUS: PASS")