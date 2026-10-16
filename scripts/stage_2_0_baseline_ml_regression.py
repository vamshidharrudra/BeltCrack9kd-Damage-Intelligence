import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# STAGE 2.0
# BASELINE ML REGRESSION
#
# Prediction:
#
#   frame t physical damage features
#              |
#              v
#        regression model
#              |
#              v
#   crack_area_ratio at t+1
#
# Models:
#
#   1. Persistence baseline
#   2. Linear Regression
#   3. Ridge Regression
#   4. Random Forest
#   5. Gradient Boosting
#
# No future-frame features are used.
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

LEAKAGE_REPORT = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_4_feature_target_leakage_audit.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

OUTPUT_RESULTS = os.path.join(
    OUTPUT_DIR,
    "stage_2_0_baseline_results.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    OUTPUT_DIR,
    "stage_2_0_baseline_predictions.csv"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "stage_2_0_baseline_ml_summary.json"
)


# ============================================================
# START
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.0 - BASELINE ML REGRESSION")
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
        LEAKAGE_REPORT,
        "Stage 1.4 leakage audit"
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
# LOAD LEAKAGE REPORT
# ============================================================

print(
    "Checking Stage 1.4 leakage audit..."
)

with open(
    LEAKAGE_REPORT,
    "r",
    encoding="utf-8"
) as file:

    leakage_report = json.load(
        file
    )


leakage_status = (
    leakage_report.get(
        "status"
    )
)


print(
    "Stage 1.4 status:",
    leakage_status
)


if leakage_status != "PASS":

    print("")
    print(
        "ERROR: Stage 1.4 did not PASS."
    )

    print(
        "ML training is stopped."
    )

    raise SystemExit(1)


# ============================================================
# LOAD FEATURES
# ============================================================

print("")
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


# ============================================================
# LOAD TARGETS
# ============================================================

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


# ============================================================
# PRIMARY TARGET ONLY
# ============================================================

primary = targets[
    targets[
        "horizon"
    ]
    ==
    1
].copy()


print("")
print(
    "Primary t+1 records:",
    len(primary)
)


# ============================================================
# EXTRACT MODEL FEATURES FROM STAGE 1.4
# ============================================================

model_features = leakage_report.get(
    "model_features",
    []
)


if not model_features:

    print("")
    print(
        "ERROR: No model features found in Stage 1.4 report."
    )

    raise SystemExit(1)


print("")
print(
    "Model features:",
    len(model_features)
)

for column in model_features:

    print(
        "  -",
        column
    )


# ============================================================
# VERIFY FEATURES EXIST
# ============================================================

missing_model_features = [
    column
    for column in model_features
    if column not in features.columns
]


if missing_model_features:

    print("")
    print(
        "ERROR: Model features missing from Stage 1.0:"
    )

    for column in missing_model_features:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# MERGE CURRENT FRAME FEATURES
# ============================================================

print("")
print(
    "Merging current-frame features with primary targets..."
)


feature_columns_for_merge = [
    "record_id"
] + model_features


current_features = features[
    feature_columns_for_merge
].copy()


current_features = current_features.rename(
    columns={
        column:
            "feature_" + column
        for column in model_features
    }
)


merged = primary.merge(
    current_features,
    left_on="current_record_id",
    right_on="record_id",
    how="left"
)


print(
    "Merged records:",
    len(merged)
)


# ============================================================
# CHECK MERGE
# ============================================================

missing_feature_rows = int(
    merged[
        "record_id"
    ].isna().sum()
)


print(
    "Missing feature rows:",
    missing_feature_rows
)


if missing_feature_rows != 0:

    print("")
    print(
        "ERROR: Some target records could not be matched."
    )

    raise SystemExit(1)


# ============================================================
# CREATE X AND y
# ============================================================

feature_columns_merged = [
    "feature_" + column
    for column in model_features
]


X = merged[
    feature_columns_merged
].copy()


y = pd.to_numeric(
    merged[
        "target_crack_area_ratio"
    ],
    errors="coerce"
)


# Rename X columns back to original names.

X.columns = model_features


# ============================================================
# CHECK NUMERIC DATA
# ============================================================

print("")
print(
    "Checking feature data..."
)


for column in X.columns:

    X[
        column
    ] = pd.to_numeric(
        X[
            column
        ],
        errors="coerce"
    )


missing_feature_values = int(
    X.isna().sum().sum()
)


missing_target_values = int(
    y.isna().sum()
)


print(
    "Missing feature values:",
    missing_feature_values
)

print(
    "Missing target values:",
    missing_target_values
)


if (
    missing_feature_values > 0
    or
    missing_target_values > 0
):

    print("")
    print(
        "ERROR: Missing numeric values detected."
    )

    raise SystemExit(1)


# ============================================================
# SPLIT DATA
#
# The split was already constructed in Stage 0.9D.1 and
# verified in Stage 0.9D.2.
#
# We use the split stored with the prediction target.
# ============================================================

train_mask = (
    merged[
        "split"
    ]
    ==
    "train"
)

val_mask = (
    merged[
        "split"
    ]
    ==
    "val"
)

test_mask = (
    merged[
        "split"
    ]
    ==
    "test"
)


X_train = X.loc[
    train_mask
].copy()

y_train = y.loc[
    train_mask
].copy()


X_val = X.loc[
    val_mask
].copy()

y_val = y.loc[
    val_mask
].copy()


X_test = X.loc[
    test_mask
].copy()

y_test = y.loc[
    test_mask
].copy()


print("")
print(
    "ML split sizes:"
)

print(
    "TRAIN:",
    len(X_train)
)

print(
    "VAL:",
    len(X_val)
)

print(
    "TEST:",
    len(X_test)
)


# ============================================================
# CHECK SEQUENCE SPLIT INTEGRITY
# ============================================================

sequence_split_counts = (
    merged[
        [
            "sequence",
            "split"
        ]
    ]
    .drop_duplicates()
    .groupby(
        "sequence"
    )[
        "split"
    ]
    .nunique()
)


sequence_conflicts = int(
    (
        sequence_split_counts
        >
        1
    ).sum()
)


print("")
print(
    "Sequence split conflicts:",
    sequence_conflicts
)


if sequence_conflicts != 0:

    print(
        "ERROR: Sequence leakage detected."
    )

    raise SystemExit(1)


# ============================================================
# COMPONENT SPLIT INTEGRITY
# ============================================================

component_split_counts = (
    merged[
        [
            "component",
            "split"
        ]
    ]
    .drop_duplicates()
    .groupby(
        "component"
    )[
        "split"
    ]
    .nunique()
)


component_conflicts = int(
    (
        component_split_counts
        >
        1
    ).sum()
)


print(
    "Component split conflicts:",
    component_conflicts
)


if component_conflicts != 0:

    print(
        "ERROR: Component leakage detected."
    )

    raise SystemExit(1)


# ============================================================
# PERSISTENCE BASELINE
#
# Predict:
#
#   future damage = current damage
#
# This is a very important baseline for temporal forecasting.
# ============================================================

print("")
print("=" * 70)
print("BASELINE 1 - PERSISTENCE")
print("=" * 70)


persistence_train_pred = pd.to_numeric(
    merged.loc[
        train_mask,
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


persistence_val_pred = pd.to_numeric(
    merged.loc[
        val_mask,
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


persistence_test_pred = pd.to_numeric(
    merged.loc[
        test_mask,
        "current_crack_area_ratio"
    ],
    errors="coerce"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# STORAGE
# ============================================================

results = []

prediction_records = []


# ============================================================
# ADD PERSISTENCE RESULTS
# ============================================================

persistence_predictions = [
    (
        "train",
        merged.loc[
            train_mask
        ],
        persistence_train_pred
    ),
    (
        "val",
        merged.loc[
            val_mask
        ],
        persistence_val_pred
    ),
    (
        "test",
        merged.loc[
            test_mask
        ],
        persistence_test_pred
    )
]


for split_name, split_rows, predictions in (
    persistence_predictions
):

    true_values = (
        split_rows[
            "target_crack_area_ratio"
        ]
        .astype(float)
    )


    metrics = calculate_metrics(
        true_values,
        predictions
    )


    results.append({

        "model":
            "Persistence",

        "split":
            split_name,

        "records":
            len(
                true_values
            ),

        "MAE":
            metrics[
                "MAE"
            ],

        "RMSE":
            metrics[
                "RMSE"
            ],

        "R2":
            metrics[
                "R2"
            ]
    })


# ============================================================
# DEFINE ML MODELS
# ============================================================

models = {

    "LinearRegression":
        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        ),

    "Ridge":
        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0
                    )
                )
            ]
        ),

    "RandomForest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "GradientBoosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

for model_name, model in models.items():

    print("")
    print("=" * 70)

    print(
        "TRAINING:",
        model_name
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    split_predictions = [
        (
            "train",
            X_train,
            y_train,
            train_pred
        ),
        (
            "val",
            X_val,
            y_val,
            val_pred
        ),
        (
            "test",
            X_test,
            y_test,
            test_pred
        )
    ]


    for split_name, X_split, y_split, predictions in (
        split_predictions
    ):

        metrics = calculate_metrics(
            y_split,
            predictions
        )


        results.append({

            "model":
                model_name,

            "split":
                split_name,

            "records":
                len(
                    y_split
                ),

            "MAE":
                metrics[
                    "MAE"
                ],

            "RMSE":
                metrics[
                    "RMSE"
                ],

            "R2":
                metrics[
                    "R2"
                ]
        })


    # --------------------------------------------------------
    # STORE TEST PREDICTIONS
    # --------------------------------------------------------

    test_rows = merged.loc[
        test_mask
    ].copy()


    for i in range(
        len(
            test_rows
        )
    ):

        row = test_rows.iloc[
            i
        ]


        prediction_records.append({

            "model":
                model_name,

            "record_id":
                int(
                    row[
                        "current_record_id"
                    ]
                ),

            "sequence":
                row[
                    "sequence"
                ],

            "component":
                int(
                    row[
                        "component"
                    ]
                ),

            "current_frame":
                int(
                    row[
                        "current_frame"
                    ]
                ),

            "future_frame":
                int(
                    row[
                        "future_frame"
                    ]
                ),

            "current_crack_area_ratio":
                float(
                    row[
                        "current_crack_area_ratio"
                    ]
                ),

            "actual_target_crack_area_ratio":
                float(
                    row[
                        "target_crack_area_ratio"
                    ]
                ),

            "predicted_target_crack_area_ratio":
                float(
                    predictions[
                        i
                    ]
                )
        })


# ============================================================
# ADD PERSISTENCE TEST PREDICTIONS
# ============================================================

test_rows = merged.loc[
    test_mask
].copy()


for i in range(
    len(
        test_rows
    )
):

    row = test_rows.iloc[
        i
    ]


    prediction_records.append({

        "model":
            "Persistence",

        "record_id":
            int(
                row[
                    "current_record_id"
                ]
            ),

        "sequence":
            row[
                "sequence"
            ],

        "component":
            int(
                row[
                    "component"
                ]
            ),

        "current_frame":
            int(
                row[
                    "current_frame"
                ]
            ),

        "future_frame":
            int(
                row[
                    "future_frame"
                ]
            ),

        "current_crack_area_ratio":
            float(
                row[
                    "current_crack_area_ratio"
                ]
            ),

        "actual_target_crack_area_ratio":
            float(
                row[
                    "target_crack_area_ratio"
                ]
            ),

        "predicted_target_crack_area_ratio":
            float(
                persistence_test_pred.iloc[
                    i
                ]
            )
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


predictions_df = pd.DataFrame(
    prediction_records
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# FIND BEST MODEL USING VALIDATION MAE
#
# Validation is used for model selection.
# Test is NOT used for selecting the model.
# ============================================================

validation_results = results_df[
    results_df[
        "split"
    ]
    ==
    "val"
].copy()


validation_results = (
    validation_results
    .sort_values(
        by="MAE",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)


best_model_name = (
    validation_results.iloc[
        0
    ][
        "model"
    ]
)


best_validation_mae = float(
    validation_results.iloc[
        0
    ][
        "MAE"
    ]
)


# ============================================================
# BEST TEST RESULT
# ============================================================

best_test_row = results_df[
    (
        results_df[
            "model"
        ]
        ==
        best_model_name
    )
    &
    (
        results_df[
            "split"
        ]
        ==
        "test"
    )
].iloc[
    0
]


# Persistence test row

persistence_test_row = results_df[
    (
        results_df[
            "model"
        ]
        ==
        "Persistence"
    )
    &
    (
        results_df[
            "split"
        ]
        ==
        "test"
    )
].iloc[
    0
]


best_test_mae = float(
    best_test_row[
        "MAE"
    ]
)


persistence_test_mae = float(
    persistence_test_row[
        "MAE"
    ]
)


if persistence_test_mae > 0:

    mae_improvement_percent = (
        (
            persistence_test_mae
            -
            best_test_mae
        )
        /
        persistence_test_mae
    ) * 100.0

else:

    mae_improvement_percent = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.0 RESULTS")
print("=" * 70)

print("")
print(
    "MODEL PERFORMANCE"
)

print("")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print("")
print("=" * 70)
print("VALIDATION MODEL RANKING")
print("=" * 70)

print("")

print(
    validation_results[
        [
            "model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print("")
print(
    "Selected model:",
    best_model_name
)

print(
    "Validation MAE:",
    f"{best_validation_mae:.8f}"
)


print("")
print("=" * 70)
print("FINAL TEST COMPARISON")
print("=" * 70)

print("")

print(
    "Persistence test MAE:",
    f"{persistence_test_mae:.8f}"
)

print(
    best_model_name,
    "test MAE:",
    f"{best_test_mae:.8f}"
)

print(
    "MAE improvement over persistence:",
    f"{mae_improvement_percent:.2f}%"
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary = {

    "stage":
        "2.0",

    "title":
        "Baseline ML Regression",

    "prediction_target":
        "crack_area_ratio at t+1",

    "feature_count":
        len(
            model_features
        ),

    "model_features":
        model_features,

    "dataset_sizes": {

        "train":
            int(
                len(
                    X_train
                )
            ),

        "val":
            int(
                len(
                    X_val
                )
            ),

        "test":
            int(
                len(
                    X_test
                )
            )
    },

    "sequence_split_conflicts":
        sequence_conflicts,

    "component_split_conflicts":
        component_conflicts,

    "results":
        results_df.to_dict(
            orient="records"
        ),

    "validation_ranking":
        validation_results[
            [
                "model",
                "MAE",
                "RMSE",
                "R2"
            ]
        ].to_dict(
            orient="records"
        ),

    "selected_model":
        best_model_name,

    "selected_model_validation_mae":
        best_validation_mae,

    "selected_model_test_mae":
        best_test_mae,

    "persistence_test_mae":
        persistence_test_mae,

    "test_mae_improvement_over_persistence_percent":
        float(
            mae_improvement_percent
        ),

    "outputs": {

        "results":
            OUTPUT_RESULTS,

        "predictions":
            OUTPUT_PREDICTIONS,

        "summary":
            OUTPUT_JSON
    }
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.0 COMPLETE")
print("=" * 70)

print("")
print(
    "Results:",
    OUTPUT_RESULTS
)

print(
    "Predictions:",
    OUTPUT_PREDICTIONS
)

print(
    "Summary:",
    OUTPUT_JSON
)

print("")

print(
    "Selected model:",
    best_model_name
)

print(
    "Test MAE:",
    f"{best_test_mae:.8f}"
)

print(
    "Persistence test MAE:",
    f"{persistence_test_mae:.8f}"
)

print(
    "Improvement over persistence:",
    f"{mae_improvement_percent:.2f}%"
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")