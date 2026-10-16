import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)


# ============================================================
# INPUT FILES
# ============================================================

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
# OUTPUT FILES
# ============================================================

PREDICTION_CASE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_temporal_prediction_cases.csv"
)

SEQUENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_sequence_error_analysis.csv"
)

COMPONENT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_component_error_analysis.csv"
)

DAMAGE_LEVEL_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_damage_level_error_analysis.csv"
)

DAMAGE_CHANGE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_damage_change_error_analysis.csv"
)

MODEL_COMPARISON_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_model_comparison.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_5_temporal_error_summary.json"
)


# ============================================================
# HEADER
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.5 - TEMPORAL PREDICTION ERROR AND FAILURE ANALYSIS")
print("=" * 70)
print("")


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TEMPORAL_FILE,
    TARGET_FILE,
    SPLIT_FILE
]

for path in required_files:

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
# SPLIT COMPONENTS
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
# TARGET
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
    "future_frame"
] = pd.to_numeric(
    primary_targets[
        "future_frame"
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
# RESOLVE TARGET COLUMN
# ============================================================

if "target_t_plus_1" in temporal.columns:

    target_column = "target_t_plus_1"

elif "target_t_plus_1_x" in temporal.columns:

    target_column = "target_t_plus_1_x"

elif "target_t_plus_1_y" in temporal.columns:

    target_column = "target_t_plus_1_y"

else:

    print(
        "ERROR: t+1 target not found."
    )

    raise SystemExit(1)


if target_column != "target_t_plus_1":

    temporal[
        "target_t_plus_1"
    ] = temporal[
        target_column
    ]


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "component",
    "frame_number",
    "target_t_plus_1",
    "crack_area_ratio",
    "crack_count",
    "largest_crack_area_ratio",
    "largest_crack_area_px",
    "total_crack_area_px",
    "mean_crack_area_px",
    "total_crack_perimeter_px",
    "lag1_crack_area_ratio",
    "lag2_crack_area_ratio",
    "lag3_crack_area_ratio",
    "delta1_crack_area_ratio",
    "delta2_crack_area_ratio",
    "accel_crack_area_ratio",
    "rolling3_mean_crack_area_ratio",
    "rolling3_std_crack_area_ratio",
    "recent_crack_area_slope",
    "recent_crack_area_growth"
]


for column in numeric_columns:

    if column in temporal.columns:

        temporal[
            column
        ] = pd.to_numeric(
            temporal[
                column
            ],
            errors="coerce"
        )


# ============================================================
# ASSIGN SPLIT
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
    "ml_split"
] = temporal[
    "component"
].apply(
    assign_split
)


# ============================================================
# TEMPORAL FEATURES
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


temporal_features = [

    "lag1_crack_area_ratio",
    "lag2_crack_area_ratio",
    "lag3_crack_area_ratio",

    "lag1_largest_crack_area_ratio",
    "lag2_largest_crack_area_ratio",
    "lag3_largest_crack_area_ratio",

    "delta1_crack_area_ratio",
    "delta2_crack_area_ratio",

    "accel_crack_area_ratio",

    "rolling3_mean_crack_area_ratio",
    "rolling3_std_crack_area_ratio",

    "recent_crack_area_slope",
    "recent_crack_area_growth",

    "lag1_crack_count",
    "lag2_crack_count",
    "delta1_crack_count",

    "lag1_total_crack_perimeter_px",
    "lag2_total_crack_perimeter_px",

    "lag1_total_crack_area_px",
    "lag2_total_crack_area_px"
]


all_features = (
    current_features
    +
    temporal_features
)


# ============================================================
# COMPLETE DATA
# ============================================================

complete_columns = [
    "sequence",
    "component",
    "frame_number",
    "ml_split",
    "target_t_plus_1",
    "crack_area_ratio"
] + all_features


complete_columns = list(
    dict.fromkeys(
        complete_columns
    )
)


missing_columns = [
    column
    for column in complete_columns
    if column not in temporal.columns
]


if missing_columns:

    print(
        "ERROR: Missing required columns:"
    )

    for column in missing_columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


data = temporal[
    complete_columns
].copy()


data = data.dropna(
    subset=all_features + [
        "target_t_plus_1"
    ]
).copy()


print("")
print(
    "Complete model records:",
    len(data)
)


# ============================================================
# TEST DATA
# ============================================================

test = data[
    data[
        "ml_split"
    ]
    ==
    "TEST"
].copy()


train = data[
    data[
        "ml_split"
    ]
    ==
    "TRAIN"
].copy()


print(
    "TRAIN:",
    len(train)
)

print(
    "TEST:",
    len(test)
)


# ============================================================
# TRAIN BEST MODEL
# ============================================================

print("")
print("=" * 70)
print("TRAINING BEST STAGE 2.4 MODEL")
print("=" * 70)


model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


model.fit(
    train[
        all_features
    ],
    train[
        "target_t_plus_1"
    ]
)


test_prediction = model.predict(
    test[
        all_features
    ]
)


# ============================================================
# PERSISTENCE
# ============================================================

persistence_prediction = (
    test[
        "crack_area_ratio"
    ].to_numpy(
        dtype=float
    )
)


actual = (
    test[
        "target_t_plus_1"
    ].to_numpy(
        dtype=float
    )
)


# ============================================================
# BUILD PREDICTION CASE TABLE
# ============================================================

cases = test[
    [
        "sequence",
        "component",
        "frame_number",
        "crack_area_ratio",
        "target_t_plus_1",
        "delta1_crack_area_ratio",
        "delta2_crack_area_ratio",
        "accel_crack_area_ratio",
        "rolling3_mean_crack_area_ratio",
        "rolling3_std_crack_area_ratio",
        "recent_crack_area_slope",
        "recent_crack_area_growth"
    ]
].copy()


cases = cases.rename(
    columns={
        "crack_area_ratio":
            "current_damage",

        "target_t_plus_1":
            "future_damage"
    }
)


cases[
    "model_prediction"
] = test_prediction


cases[
    "persistence_prediction"
] = persistence_prediction


cases[
    "model_residual"
] = (
    cases[
        "model_prediction"
    ]
    -
    cases[
        "future_damage"
    ]
)


cases[
    "persistence_residual"
] = (
    cases[
        "persistence_prediction"
    ]
    -
    cases[
        "future_damage"
    ]
)


cases[
    "model_absolute_error"
] = cases[
    "model_residual"
].abs()


cases[
    "persistence_absolute_error"
] = cases[
    "persistence_residual"
].abs()


cases[
    "actual_damage_change"
] = (
    cases[
        "future_damage"
    ]
    -
    cases[
        "current_damage"
    ]
)


cases[
    "predicted_damage_change"
] = (
    cases[
        "model_prediction"
    ]
    -
    cases[
        "current_damage"
    ]
)


cases[
    "change_error"
] = (
    cases[
        "predicted_damage_change"
    ]
    -
    cases[
        "actual_damage_change"
    ]
)


# ============================================================
# ERROR DIRECTION
# ============================================================

cases[
    "error_direction"
] = np.where(
    cases[
        "model_residual"
    ]
    >
    0,
    "OVERPREDICT",
    np.where(
        cases[
            "model_residual"
        ]
        <
        0,
        "UNDERPREDICT",
        "EXACT"
    )
)


cases[
    "actual_change_direction"
] = np.where(
    cases[
        "actual_damage_change"
    ]
    >
    1e-12,
    "INCREASE",
    np.where(
        cases[
            "actual_damage_change"
        ]
        <
        -1e-12,
        "DECREASE",
        "STABLE"
    )
)


# ============================================================
# DAMAGE LEVEL
# ============================================================

damage_quantiles = test[
    "crack_area_ratio"
].quantile(
    [
        0.33,
        0.66
    ]
)


low_threshold = float(
    damage_quantiles.iloc[0]
)


high_threshold = float(
    damage_quantiles.iloc[1]
)


def damage_level(value):

    if value <= low_threshold:
        return "LOW"

    if value <= high_threshold:
        return "MEDIUM"

    return "HIGH"


cases[
    "damage_level"
] = cases[
    "current_damage"
].apply(
    damage_level
)


# ============================================================
# CHANGE MAGNITUDE
# ============================================================

change_abs = cases[
    "actual_damage_change"
].abs()


change_q50 = float(
    change_abs.quantile(
        0.50
    )
)

change_q90 = float(
    change_abs.quantile(
        0.90
    )
)


def change_level(value):

    value = abs(
        float(value)
    )

    if value <= change_q50:
        return "LOW_CHANGE"

    if value <= change_q90:
        return "MEDIUM_CHANGE"

    return "HIGH_CHANGE"


cases[
    "change_level"
] = cases[
    "actual_damage_change"
].apply(
    change_level
)


# ============================================================
# LARGE ERROR FLAG
# ============================================================

error_q95 = float(
    cases[
        "model_absolute_error"
    ].quantile(
        0.95
    )
)


cases[
    "large_error"
] = (
    cases[
        "model_absolute_error"
    ]
    >=
    error_q95
)


# ============================================================
# PRINT OVERALL ERROR
# ============================================================

model_mae = mean_absolute_error(
    actual,
    test_prediction
)

model_rmse = np.sqrt(
    mean_squared_error(
        actual,
        test_prediction
    )
)

model_r2 = r2_score(
    actual,
    test_prediction
)


persistence_mae = mean_absolute_error(
    actual,
    persistence_prediction
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        actual,
        persistence_prediction
    )
)

persistence_r2 = r2_score(
    actual,
    persistence_prediction
)


print("")
print(
    "GradientBoosting TEST MAE:",
    f"{model_mae:.8f}"
)

print(
    "GradientBoosting TEST RMSE:",
    f"{model_rmse:.8f}"
)

print(
    "GradientBoosting TEST R2:",
    f"{model_r2:.8f}"
)

print("")

print(
    "Persistence TEST MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Persistence TEST RMSE:",
    f"{persistence_rmse:.8f}"
)

print(
    "Persistence TEST R2:",
    f"{persistence_r2:.8f}"
)


# ============================================================
# MODEL VS PERSISTENCE
# ============================================================

model_better = (
    cases[
        "model_absolute_error"
    ]
    <
    cases[
        "persistence_absolute_error"
    ]
)


persistence_better = (
    cases[
        "persistence_absolute_error"
    ]
    <
    cases[
        "model_absolute_error"
    ]
)


equal_error = (
    cases[
        "model_absolute_error"
    ]
    ==
    cases[
        "persistence_absolute_error"
    ]
)


print("")
print(
    "Model beats persistence:",
    int(model_better.sum())
)

print(
    "Persistence beats model:",
    int(persistence_better.sum())
)

print(
    "Equal:",
    int(equal_error.sum())
)


# ============================================================
# SEQUENCE ANALYSIS
# ============================================================

sequence_rows = []


for sequence, group in cases.groupby(
    "sequence"
):

    sequence_actual = group[
        "future_damage"
    ]

    sequence_prediction = group[
        "model_prediction"
    ]

    sequence_persistence = group[
        "persistence_prediction"
    ]


    mae = mean_absolute_error(
        sequence_actual,
        sequence_prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            sequence_actual,
            sequence_prediction
        )
    )


    r2 = r2_score(
        sequence_actual,
        sequence_prediction
    )


    persistence_sequence_mae = mean_absolute_error(
        sequence_actual,
        sequence_persistence
    )


    improvement = (
        (
            persistence_sequence_mae
            -
            mae
        )
        /
        persistence_sequence_mae
    ) * 100.0


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
                float(r2),

            "persistence_MAE":
                float(
                    persistence_sequence_mae
                ),

            "improvement_vs_persistence_percent":
                float(improvement),

            "mean_actual_change":
                float(
                    group[
                        "actual_damage_change"
                    ].mean()
                ),

            "positive_change_fraction":
                float(
                    (
                        group[
                            "actual_damage_change"
                        ]
                        >
                        0
                    ).mean()
                ),

            "mean_absolute_change":
                float(
                    group[
                        "actual_damage_change"
                    ].abs().mean()
                )
        }
    )


sequence_df = pd.DataFrame(
    sequence_rows
).sort_values(
    "MAE"
)


print("")
print("=" * 70)
print("SEQUENCE ERROR ANALYSIS")
print("=" * 70)

print(
    sequence_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

component_rows = []


for component, group in cases.groupby(
    "component"
):

    mae = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "model_prediction"
        ]
    )


    persistence_component_mae = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "persistence_prediction"
        ]
    )


    improvement = (
        (
            persistence_component_mae
            -
            mae
        )
        /
        persistence_component_mae
    ) * 100.0


    component_rows.append(
        {
            "component":
                int(component),

            "records":
                len(group),

            "MAE":
                float(mae),

            "persistence_MAE":
                float(
                    persistence_component_mae
                ),

            "improvement_vs_persistence_percent":
                float(improvement)
        }
    )


component_df = pd.DataFrame(
    component_rows
).sort_values(
    "MAE"
)


# ============================================================
# DAMAGE LEVEL ANALYSIS
# ============================================================

damage_rows = []


for level, group in cases.groupby(
    "damage_level"
):

    mae = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "model_prediction"
        ]
    )


    rmse = np.sqrt(
        mean_squared_error(
            group[
                "future_damage"
            ],
            group[
                "model_prediction"
            ]
        )
    )


    persistence_mae_level = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "persistence_prediction"
        ]
    )


    damage_rows.append(
        {
            "damage_level":
                level,

            "records":
                len(group),

            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "persistence_MAE":
                float(
                    persistence_mae_level
                ),

            "mean_current_damage":
                float(
                    group[
                        "current_damage"
                    ].mean()
                ),

            "mean_future_damage":
                float(
                    group[
                        "future_damage"
                    ].mean()
                )
        }
    )


damage_df = pd.DataFrame(
    damage_rows
)


# ============================================================
# DAMAGE CHANGE ANALYSIS
# ============================================================

change_rows = []


for level, group in cases.groupby(
    "change_level"
):

    mae = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "model_prediction"
        ]
    )


    persistence_mae_change = mean_absolute_error(
        group[
            "future_damage"
        ],
        group[
            "persistence_prediction"
        ]
    )


    change_rows.append(
        {
            "change_level":
                level,

            "records":
                len(group),

            "MAE":
                float(mae),

            "persistence_MAE":
                float(
                    persistence_mae_change
                ),

            "mean_absolute_change":
                float(
                    group[
                        "actual_damage_change"
                    ].abs().mean()
                ),

            "mean_signed_change":
                float(
                    group[
                        "actual_damage_change"
                    ].mean()
                ),

            "positive_change_fraction":
                float(
                    (
                        group[
                            "actual_damage_change"
                        ]
                        >
                        0
                    ).mean()
                )
        }
    )


change_df = pd.DataFrame(
    change_rows
)


# ============================================================
# ERROR DIRECTION ANALYSIS
# ============================================================

direction_rows = []


for direction, group in cases.groupby(
    "actual_change_direction"
):

    direction_rows.append(
        {
            "actual_change_direction":
                direction,

            "records":
                len(group),

            "MAE":
                float(
                    group[
                        "model_absolute_error"
                    ].mean()
                ),

            "mean_residual":
                float(
                    group[
                        "model_residual"
                    ].mean()
                ),

            "underprediction_fraction":
                float(
                    (
                        group[
                            "model_residual"
                        ]
                        <
                        0
                    ).mean()
                ),

            "overprediction_fraction":
                float(
                    (
                        group[
                            "model_residual"
                        ]
                        >
                        0
                    ).mean()
                )
        }
    )


direction_df = pd.DataFrame(
    direction_rows
)


# ============================================================
# LARGE ERROR CASES
# ============================================================

large_error_cases = cases[
    cases[
        "large_error"
    ]
].copy()


large_error_cases = large_error_cases.sort_values(
    "model_absolute_error",
    ascending=False
)


print("")
print("=" * 70)
print("LARGE ERROR CASES")
print("=" * 70)

print(
    "95th percentile absolute error:",
    f"{error_q95:.8f}"
)

print(
    "Large-error records:",
    len(large_error_cases)
)

print("")

print(
    large_error_cases[
        [
            "sequence",
            "component",
            "frame_number",
            "current_damage",
            "future_damage",
            "model_prediction",
            "model_absolute_error",
            "actual_damage_change",
            "predicted_damage_change",
            "change_level",
            "error_direction"
        ]
    ].head(
        20
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# OVERALL RESIDUAL ANALYSIS
# ============================================================

mean_residual = float(
    cases[
        "model_residual"
    ].mean()
)


median_residual = float(
    cases[
        "model_residual"
    ].median()
)


underprediction_fraction = float(
    (
        cases[
            "model_residual"
        ]
        <
        0
    ).mean()
)


overprediction_fraction = float(
    (
        cases[
            "model_residual"
        ]
        >
        0
    ).mean()
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

cases.to_csv(
    PREDICTION_CASE_FILE,
    index=False
)


sequence_df.to_csv(
    SEQUENCE_FILE,
    index=False
)


component_df.to_csv(
    COMPONENT_FILE,
    index=False
)


damage_df.to_csv(
    DAMAGE_LEVEL_FILE,
    index=False
)


change_df.to_csv(
    DAMAGE_CHANGE_FILE,
    index=False
)


comparison_df = pd.DataFrame(
    [
        {
            "model":
                "Temporal GradientBoosting",

            "MAE":
                model_mae,

            "RMSE":
                model_rmse,

            "R2":
                model_r2,

            "records":
                len(test)
        },
        {
            "model":
                "Persistence",

            "MAE":
                persistence_mae,

            "RMSE":
                persistence_rmse,

            "R2":
                persistence_r2,

            "records":
                len(test)
        }
    ]
)


comparison_df.to_csv(
    MODEL_COMPARISON_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

best_sequence = sequence_df.iloc[
    sequence_df[
        "MAE"
    ].idxmin()
]


worst_sequence = sequence_df.iloc[
    sequence_df[
        "MAE"
    ].idxmax()
]


summary = {

    "stage":
        "2.5",

    "status":
        "PASS",

    "model":
        "GradientBoostingRegressor",

    "test_records":
        int(len(test)),

    "test_MAE":
        float(model_mae),

    "test_RMSE":
        float(model_rmse),

    "test_R2":
        float(model_r2),

    "persistence_MAE":
        float(persistence_mae),

    "persistence_RMSE":
        float(persistence_rmse),

    "persistence_R2":
        float(persistence_r2),

    "model_vs_persistence_improvement_percent":
        float(
            (
                (
                    persistence_mae
                    -
                    model_mae
                )
                /
                persistence_mae
            )
            * 100.0
        ),

    "model_beats_persistence_fraction":
        float(
            model_better.mean()
        ),

    "persistence_beats_model_fraction":
        float(
            persistence_better.mean()
        ),

    "mean_residual":
        mean_residual,

    "median_residual":
        median_residual,

    "underprediction_fraction":
        underprediction_fraction,

    "overprediction_fraction":
        overprediction_fraction,

    "error_95th_percentile":
        error_q95,

    "low_damage_threshold":
        low_threshold,

    "high_damage_threshold":
        high_threshold,

    "change_median_threshold":
        change_q50,

    "change_90th_percentile_threshold":
        change_q90,

    "best_sequence":
        best_sequence.to_dict(),

    "worst_sequence":
        worst_sequence.to_dict(),

    "sequence_results":
        sequence_df.to_dict(
            orient="records"
        ),

    "component_results":
        component_df.to_dict(
            orient="records"
        ),

    "damage_level_results":
        damage_df.to_dict(
            orient="records"
        ),

    "damage_change_results":
        change_df.to_dict(
            orient="records"
        ),

    "direction_results":
        direction_df.to_dict(
            orient="records"
        ),

    "outputs":
        {
            "prediction_cases":
                PREDICTION_CASE_FILE,

            "sequence":
                SEQUENCE_FILE,

            "component":
                COMPONENT_FILE,

            "damage_level":
                DAMAGE_LEVEL_FILE,

            "damage_change":
                DAMAGE_CHANGE_FILE,

            "model_comparison":
                MODEL_COMPARISON_FILE,

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
print("STAGE 2.5 SUMMARY")
print("=" * 70)

print("")

print(
    "Temporal GradientBoosting MAE:",
    f"{model_mae:.8f}"
)

print(
    "Temporal GradientBoosting RMSE:",
    f"{model_rmse:.8f}"
)

print(
    "Temporal GradientBoosting R2:",
    f"{model_r2:.8f}"
)

print("")

print(
    "Persistence MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Model vs persistence improvement:",
    f"{((persistence_mae - model_mae) / persistence_mae) * 100.0:.2f}%"
)

print("")

print(
    "Model beats persistence:",
    f"{model_better.mean() * 100.0:.2f}%"
)

print(
    "Persistence beats model:",
    f"{persistence_better.mean() * 100.0:.2f}%"
)

print("")

print(
    "Mean residual:",
    f"{mean_residual:.8f}"
)

print(
    "Median residual:",
    f"{median_residual:.8f}"
)

print(
    "Underprediction fraction:",
    f"{underprediction_fraction * 100.0:.2f}%"
)

print(
    "Overprediction fraction:",
    f"{overprediction_fraction * 100.0:.2f}%"
)

print("")

print(
    "Best sequence:",
    best_sequence["sequence"]
)

print(
    "Best sequence MAE:",
    f"{best_sequence['MAE']:.8f}"
)

print(
    "Worst sequence:",
    worst_sequence["sequence"]
)

print(
    "Worst sequence MAE:",
    f"{worst_sequence['MAE']:.8f}"
)

print("")

print(
    "Prediction cases:"
)

print(
    PREDICTION_CASE_FILE
)

print("")

print(
    "Sequence analysis:"
)

print(
    SEQUENCE_FILE
)

print("")

print(
    "Component analysis:"
)

print(
    COMPONENT_FILE
)

print("")

print(
    "Damage-level analysis:"
)

print(
    DAMAGE_LEVEL_FILE
)

print("")

print(
    "Damage-change analysis:"
)

print(
    DAMAGE_CHANGE_FILE
)

print("")

print(
    "Model comparison:"
)

print(
    MODEL_COMPARISON_FILE
)

print("")

print(
    "Summary:"
)

print(
    SUMMARY_FILE
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")