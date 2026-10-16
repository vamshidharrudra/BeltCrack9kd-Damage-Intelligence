import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)


TEMPORAL_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_2_temporal_features.csv"
)

TARGET_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_3_prediction_targets.csv"
)

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_0_9d1_optimized_split_plan.json"
)


# ============================================================
# OUTPUTS
# ============================================================

IMPORTANCE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_4_temporal_feature_importance.csv"
)

ABLATION_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_4_temporal_ablation_results.csv"
)

SEQUENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_4_sequence_performance.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_4_feature_importance_summary.json"
)


# ============================================================
# HEADER
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.4 - TEMPORAL FEATURE IMPORTANCE AND ABLATION")
print("=" * 70)
print("")


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    TEMPORAL_FILE,
    TARGET_FILE,
    SPLIT_FILE
]:

    if not os.path.exists(path):

        print(
            "ERROR: Required file not found:"
        )

        print(path)

        raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading Stage 2.2 temporal features..."
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
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
    "Loading optimized split plan..."
)

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as file:

    split_plan = json.load(
        file
    )


# ============================================================
# SPLIT PLAN
# ============================================================

train_components = set(
    int(x)
    for x in split_plan[
        "train_components"
    ]
)

val_components = set(
    int(x)
    for x in split_plan[
        "val_components"
    ]
)

test_components = set(
    int(x)
    for x in split_plan[
        "test_components"
    ]
)


# ============================================================
# TARGET VERIFICATION
# ============================================================

primary_targets = targets[
    targets[
        "horizon"
    ]
    ==
    1
].copy()


primary_targets[
    "component"
] = pd.to_numeric(
    primary_targets[
        "component"
    ],
    errors="coerce"
)


primary_targets[
    "current_frame"
] = pd.to_numeric(
    primary_targets[
        "current_frame"
    ],
    errors="coerce"
)


primary_targets[
    "target_crack_area_ratio"
] = pd.to_numeric(
    primary_targets[
        "target_crack_area_ratio"
    ],
    errors="coerce"
)


target_reference = primary_targets[
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
        "current_frame":
            "reference_current_frame",

        "future_frame":
            "reference_future_frame",

        "target_crack_area_ratio":
            "reference_target"
    }
)


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

if "target_t_plus_1" in temporal.columns:

    target_column = "target_t_plus_1"

elif "target_t_plus_1_x" in temporal.columns:

    target_column = "target_t_plus_1_x"

elif "target_t_plus_1_y" in temporal.columns:

    target_column = "target_t_plus_1_y"

else:

    print(
        "ERROR: t+1 target not found in Stage 2.2."
    )

    raise SystemExit(1)


if target_column != "target_t_plus_1":

    temporal[
        "target_t_plus_1"
    ] = temporal[
        target_column
    ]


# ============================================================
# ASSIGN SPLITS
# ============================================================

def assign_split(component):

    component = int(component)

    if component in train_components:
        return "TRAIN"

    if component in val_components:
        return "VAL"

    if component in test_components:
        return "TEST"

    return None


temporal[
    "component"
] = pd.to_numeric(
    temporal[
        "component"
    ],
    errors="coerce"
)


temporal[
    "frame_number"
] = pd.to_numeric(
    temporal[
        "frame_number"
    ],
    errors="coerce"
)


temporal[
    "ml_split"
] = temporal[
    "component"
].apply(
    assign_split
)


if temporal[
    "ml_split"
].isna().any():

    print(
        "ERROR: Some records have no split."
    )

    raise SystemExit(1)


# ============================================================
# TEMPORAL FEATURE GROUPS
# ============================================================

lag_features = [

    "lag1_crack_area_ratio",
    "lag2_crack_area_ratio",
    "lag3_crack_area_ratio",

    "lag1_largest_crack_area_ratio",
    "lag2_largest_crack_area_ratio",
    "lag3_largest_crack_area_ratio",

    "lag1_crack_count",
    "lag2_crack_count",

    "lag1_total_crack_perimeter_px",
    "lag2_total_crack_perimeter_px",

    "lag1_total_crack_area_px",
    "lag2_total_crack_area_px"
]


change_features = [

    "delta1_crack_area_ratio",
    "delta2_crack_area_ratio",

    "delta1_crack_count",

    "accel_crack_area_ratio"
]


rolling_features = [

    "rolling3_mean_crack_area_ratio",
    "rolling3_std_crack_area_ratio"
]


trend_features = [

    "recent_crack_area_slope",
    "recent_crack_area_growth"
]


temporal_features = (
    lag_features
    +
    change_features
    +
    rolling_features
    +
    trend_features
)


# ============================================================
# CURRENT FEATURES
# ============================================================

current_features = [

    "image_width_px",
    "image_height_px",
    "image_area_px",

    "crack_count",
    "total_crack_area_px",
    "crack_area_ratio",

    "largest_crack_area_px",
    "largest_crack_area_ratio",

    "smallest_crack_area_px",

    "mean_crack_area_px",
    "median_crack_area_px",

    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "max_crack_perimeter_px",

    "mean_crack_bbox_width_px",
    "mean_crack_bbox_height_px",
    "mean_crack_bbox_area_px",

    "max_crack_bbox_width_px",
    "max_crack_bbox_height_px",

    "mean_crack_centroid_x_px",
    "mean_crack_centroid_y_px",

    "mean_crack_centroid_x_norm",
    "mean_crack_centroid_y_norm",

    "crack_centroid_x_std_px",
    "crack_centroid_y_std_px",

    "crack_centroid_x_std_norm",
    "crack_centroid_y_std_norm",

    "crack_bbox_union_width_px",
    "crack_bbox_union_height_px",

    "json_image_width_px",
    "json_image_height_px",

    "annotation_shape_count",
    "valid_crack_polygon_count",
    "invalid_shape_count"
]


# ============================================================
# CHECK FEATURES
# ============================================================

all_required_features = (
    current_features
    +
    temporal_features
)


missing_features = [
    feature
    for feature in all_required_features
    if feature not in temporal.columns
]


if missing_features:

    print(
        "ERROR: Missing features:"
    )

    for feature in missing_features:

        print(
            "  -",
            feature
        )

    raise SystemExit(1)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for feature in all_required_features:

    temporal[
        feature
    ] = pd.to_numeric(
        temporal[
            feature
        ],
        errors="coerce"
    )


temporal[
    "target_t_plus_1"
] = pd.to_numeric(
    temporal[
        "target_t_plus_1"
    ],
    errors="coerce"
)


# ============================================================
# BUILD COMPLETE DATASET
# ============================================================

model_features = (
    current_features
    +
    temporal_features
)


complete = temporal[
    [
        "sequence",
        "component",
        "frame_number",
        "ml_split",
        "target_t_plus_1"
    ]
    +
    model_features
].dropna(
    subset=model_features + [
        "target_t_plus_1"
    ]
).copy()


print("")
print(
    "Complete records:",
    len(complete)
)


# ============================================================
# SPLITS
# ============================================================

train = complete[
    complete["ml_split"] == "TRAIN"
].copy()


val = complete[
    complete["ml_split"] == "VAL"
].copy()


test = complete[
    complete["ml_split"] == "TEST"
].copy()


print(
    "TRAIN:",
    len(train)
)

print(
    "VAL:",
    len(val)
)

print(
    "TEST:",
    len(test)
)


# ============================================================
# BASE MODEL
# ============================================================

print("")
print("=" * 70)
print("BASE TEMPORAL GRADIENT BOOSTING MODEL")
print("=" * 70)


base_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


X_train = train[
    model_features
]

y_train = train[
    "target_t_plus_1"
]


X_val = val[
    model_features
]

y_val = val[
    "target_t_plus_1"
]


X_test = test[
    model_features
]

y_test = test[
    "target_t_plus_1"
]


base_model.fit(
    X_train,
    y_train
)


base_test_prediction = base_model.predict(
    X_test
)


base_test_mae = mean_absolute_error(
    y_test,
    base_test_prediction
)


base_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        base_test_prediction
    )
)


base_test_r2 = r2_score(
    y_test,
    base_test_prediction
)


print(
    "Base TEST MAE:",
    f"{base_test_mae:.8f}"
)

print(
    "Base TEST RMSE:",
    f"{base_test_rmse:.8f}"
)

print(
    "Base TEST R2:",
    f"{base_test_r2:.8f}"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("")
print("=" * 70)
print("TEMPORAL FEATURE IMPORTANCE")
print("=" * 70)


print(
    "Calculating permutation importance on TEST..."
)


permutation = permutation_importance(
    base_model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


importance_df = pd.DataFrame(
    {
        "feature":
            model_features,

        "importance_mean":
            permutation.importances_mean,

        "importance_std":
            permutation.importances_std
    }
)


importance_df[
    "absolute_importance"
] = importance_df[
    "importance_mean"
].abs()


importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False
).reset_index(
    drop=True
)


print("")
print(
    "Top feature importances:"
)

print(
    importance_df.head(
        20
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# TEMPORAL-ONLY IMPORTANCE
# ============================================================

temporal_importance = importance_df[
    importance_df[
        "feature"
    ].isin(
        temporal_features
    )
].copy()


print("")
print(
    "Top temporal features:"
)

print(
    temporal_importance.head(
        20
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# ABLATION FUNCTION
# ============================================================

def run_ablation(
    name,
    features
):

    print("")
    print(
        "Ablation:",
        name
    )

    train_local = train[
        features
    ]

    test_local = test[
        features
    ]


    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )


    model.fit(
        train_local,
        y_train
    )


    prediction = model.predict(
        test_local
    )


    mae = mean_absolute_error(
        y_test,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    r2 = r2_score(
        y_test,
        prediction
    )


    improvement = (
        (
            base_test_mae
            -
            mae
        )
        /
        base_test_mae
    ) * 100.0


    return {
        "experiment":
            name,

        "feature_count":
            len(features),

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "improvement_vs_full_temporal_percent":
            float(improvement)
    }


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

ablation_results = []


# ------------------------------------------------------------
# Full model
# ------------------------------------------------------------

ablation_results.append(
    run_ablation(
        "FULL_TEMPORAL",
        current_features + temporal_features
    )
)


# ------------------------------------------------------------
# Remove lag features
# ------------------------------------------------------------

features_without_lag = (
    current_features
    +
    change_features
    +
    rolling_features
    +
    trend_features
)


ablation_results.append(
    run_ablation(
        "WITHOUT_LAG_FEATURES",
        features_without_lag
    )
)


# ------------------------------------------------------------
# Remove change features
# ------------------------------------------------------------

features_without_change = (
    current_features
    +
    lag_features
    +
    rolling_features
    +
    trend_features
)


ablation_results.append(
    run_ablation(
        "WITHOUT_CHANGE_FEATURES",
        features_without_change
    )
)


# ------------------------------------------------------------
# Remove rolling features
# ------------------------------------------------------------

features_without_rolling = (
    current_features
    +
    lag_features
    +
    change_features
    +
    trend_features
)


ablation_results.append(
    run_ablation(
        "WITHOUT_ROLLING_FEATURES",
        features_without_rolling
    )
)


# ------------------------------------------------------------
# Remove trend features
# ------------------------------------------------------------

features_without_trend = (
    current_features
    +
    lag_features
    +
    change_features
    +
    rolling_features
)


ablation_results.append(
    run_ablation(
        "WITHOUT_TREND_FEATURES",
        features_without_trend
    )
)


# ------------------------------------------------------------
# Current-only
# ------------------------------------------------------------

ablation_results.append(
    run_ablation(
        "CURRENT_ONLY",
        current_features
    )
)


# ============================================================
# SINGLE FEATURE GROUP MODELS
# ============================================================

ablation_results.append(
    run_ablation(
        "CURRENT_PLUS_LAGS_ONLY",
        current_features + lag_features
    )
)


ablation_results.append(
    run_ablation(
        "CURRENT_PLUS_CHANGE_ONLY",
        current_features + change_features
    )
)


ablation_results.append(
    run_ablation(
        "CURRENT_PLUS_ROLLING_ONLY",
        current_features + rolling_features
    )
)


ablation_results.append(
    run_ablation(
        "CURRENT_PLUS_TREND_ONLY",
        current_features + trend_features
    )
)


ablation_df = pd.DataFrame(
    ablation_results
)


ablation_df = ablation_df.sort_values(
    "MAE"
).reset_index(
    drop=True
)


# ============================================================
# PRINT ABLATION
# ============================================================

print("")
print("=" * 70)
print("ABLATION RESULTS")
print("=" * 70)

print(
    ablation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# SEQUENCE-LEVEL PERFORMANCE
# ============================================================

print("")
print("=" * 70)
print("SEQUENCE-LEVEL PERFORMANCE")
print("=" * 70)


test_predictions = test[
    [
        "sequence",
        "component",
        "frame_number",
        "target_t_plus_1"
    ]
].copy()


test_predictions[
    "prediction"
] = base_test_prediction


test_predictions[
    "absolute_error"
] = (
    test_predictions[
        "prediction"
    ]
    -
    test_predictions[
        "target_t_plus_1"
    ]
).abs()


sequence_rows = []


for sequence, group in test_predictions.groupby(
    "sequence"
):

    mae = mean_absolute_error(
        group[
            "target_t_plus_1"
        ],
        group[
            "prediction"
        ]
    )


    rmse = np.sqrt(
        mean_squared_error(
            group[
                "target_t_plus_1"
            ],
            group[
                "prediction"
            ]
        )
    )


    r2 = r2_score(
        group[
            "target_t_plus_1"
        ],
        group[
            "prediction"
        ]
    )


    sequence_rows.append(
        {
            "sequence":
                sequence,

            "component":
                int(
                    group[
                        "component"
                    ].iloc[0]
                ),

            "records":
                len(group),

            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "R2":
                float(r2)
        }
    )


sequence_df = pd.DataFrame(
    sequence_rows
).sort_values(
    "MAE"
)


print(
    sequence_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# COMPONENT-LEVEL PERFORMANCE
# ============================================================

component_rows = []


for component, group in test_predictions.groupby(
    "component"
):

    mae = mean_absolute_error(
        group[
            "target_t_plus_1"
        ],
        group[
            "prediction"
        ]
    )


    rmse = np.sqrt(
        mean_squared_error(
            group[
                "target_t_plus_1"
            ],
            group[
                "prediction"
            ]
        )
    )


    r2 = r2_score(
        group[
            "target_t_plus_1"
        ],
        group[
            "prediction"
        ]
    )


    component_rows.append(
        {
            "component":
                int(component),

            "records":
                len(group),

            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "R2":
                float(r2)
        }
    )


component_df = pd.DataFrame(
    component_rows
).sort_values(
    "MAE"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)


ablation_df.to_csv(
    ABLATION_FILE,
    index=False
)


sequence_df.to_csv(
    SEQUENCE_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

best_ablation = ablation_df.iloc[
    0
]


summary = {

    "stage":
        "2.4",

    "status":
        "PASS",

    "model":
        "GradientBoostingRegressor",

    "base_model_features":
        len(model_features),

    "base_test_records":
        len(test),

    "base_test_MAE":
        float(base_test_mae),

    "base_test_RMSE":
        float(base_test_rmse),

    "base_test_R2":
        float(base_test_r2),

    "temporal_feature_count":
        len(temporal_features),

    "lag_feature_count":
        len(lag_features),

    "change_feature_count":
        len(change_features),

    "rolling_feature_count":
        len(rolling_features),

    "trend_feature_count":
        len(trend_features),

    "top_temporal_features":
        temporal_importance.head(
            20
        ).to_dict(
            orient="records"
        ),

    "ablation_results":
        ablation_df.to_dict(
            orient="records"
        ),

    "best_ablation":
        best_ablation.to_dict(),

    "sequence_performance":
        sequence_df.to_dict(
            orient="records"
        ),

    "component_performance":
        component_df.to_dict(
            orient="records"
        ),

    "outputs":
        {
            "importance":
                IMPORTANCE_FILE,

            "ablation":
                ABLATION_FILE,

            "sequence":
                SEQUENCE_FILE,

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
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.4 SUMMARY")
print("=" * 70)

print("")

print(
    "Base temporal TEST MAE:",
    f"{base_test_mae:.8f}"
)

print(
    "Base temporal TEST RMSE:",
    f"{base_test_rmse:.8f}"
)

print(
    "Base temporal TEST R2:",
    f"{base_test_r2:.8f}"
)

print("")

print(
    "Best ablation:",
    best_ablation[
        "experiment"
    ]
)

print(
    "Best ablation MAE:",
    f"{best_ablation['MAE']:.8f}"
)

print("")

print(
    "Top temporal features:"
)

for _, row in temporal_importance.head(
    10
).iterrows():

    print(
        "  ",
        row["feature"],
        ":",
        f"{row['importance_mean']:.8f}"
    )


print("")

print(
    "Importance file:"
)

print(
    IMPORTANCE_FILE
)

print("")

print(
    "Ablation file:"
)

print(
    ABLATION_FILE
)

print("")

print(
    "Sequence file:"
)

print(
    SEQUENCE_FILE
)

print("")

print(
    "Summary file:"
)

print(
    SUMMARY_FILE
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")