import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

TEMPORAL_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_2_temporal_features.csv"
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

SPLIT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_0_9d1_optimized_split_plan.json"
)

RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_3_temporal_ml_results.csv"
)

PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_3_temporal_ml_predictions.csv"
)

IMPROVEMENT_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_3_current_vs_temporal.csv"
)

VERIFICATION_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_3_target_verification.csv"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_2_3_temporal_ml_summary.json"
)


# ============================================================
# HEADER
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.3 - TEMPORAL ML EXPERIMENT")
print("=" * 70)
print("")


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    TEMPORAL_FILE,
    FEATURE_FILE,
    TARGET_FILE,
    SPLIT_FILE
]

for path in required_files:

    if not os.path.exists(path):

        print("ERROR: Required file not found:")
        print(path)
        raise SystemExit(1)


# ============================================================
# LOAD FILES
# ============================================================

print("Loading Stage 2.2 temporal features...")

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
)


print("")
print("Loading Stage 1.0 features...")

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Original feature records:",
    len(features)
)


print("")
print("Loading Stage 1.3 targets...")

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


print("")
print("Loading optimized split plan...")

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_plan = json.load(f)


# ============================================================
# SPLIT PLAN
# ============================================================

if "train_components" not in split_plan:
    print("ERROR: train_components not found.")
    raise SystemExit(1)

if "val_components" not in split_plan:
    print("ERROR: val_components not found.")
    raise SystemExit(1)

if "test_components" not in split_plan:
    print("ERROR: test_components not found.")
    raise SystemExit(1)


train_components = [
    int(x)
    for x in split_plan["train_components"]
]

val_components = [
    int(x)
    for x in split_plan["val_components"]
]

test_components = [
    int(x)
    for x in split_plan["test_components"]
]


print("")
print("Optimized component split:")

print(
    "TRAIN:",
    train_components
)

print(
    "VAL:",
    val_components
)

print(
    "TEST:",
    test_components
)


# ============================================================
# SPLIT OVERLAP CHECK
# ============================================================

train_set = set(train_components)
val_set = set(val_components)
test_set = set(test_components)


if train_set.intersection(val_set):
    print("ERROR: TRAIN/VAL component overlap.")
    raise SystemExit(1)

if train_set.intersection(test_set):
    print("ERROR: TRAIN/TEST component overlap.")
    raise SystemExit(1)

if val_set.intersection(test_set):
    print("ERROR: VAL/TEST component overlap.")
    raise SystemExit(1)


# ============================================================
# PRIMARY TARGET FROM STAGE 1.3
# ============================================================

required_target_columns = [
    "sequence",
    "component",
    "horizon",
    "current_frame",
    "future_frame",
    "target_crack_area_ratio"
]

for column in required_target_columns:

    if column not in targets.columns:

        print(
            "ERROR: Missing Stage 1.3 column:",
            column
        )

        raise SystemExit(1)


primary_targets = targets[
    targets["horizon"] == 1
].copy()


print("")
print(
    "Primary t+1 target records:",
    len(primary_targets)
)


# ============================================================
# PRIMARY TARGET DUPLICATE CHECK
# ============================================================

target_key_columns = [
    "sequence",
    "component",
    "current_frame"
]

duplicate_targets = int(
    primary_targets.duplicated(
        subset=target_key_columns
    ).sum()
)


print(
    "Duplicate target keys:",
    duplicate_targets
)


if duplicate_targets != 0:

    print(
        "ERROR: Duplicate t+1 target keys."
    )

    raise SystemExit(1)


# ============================================================
# TARGET REFERENCE TABLE
# ============================================================

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
        "current_frame": "reference_current_frame",
        "future_frame": "reference_future_frame",
        "target_crack_area_ratio": "reference_target"
    }
)


target_reference["component"] = pd.to_numeric(
    target_reference["component"],
    errors="coerce"
)

target_reference["reference_current_frame"] = pd.to_numeric(
    target_reference["reference_current_frame"],
    errors="coerce"
)

target_reference["reference_future_frame"] = pd.to_numeric(
    target_reference["reference_future_frame"],
    errors="coerce"
)

target_reference["reference_target"] = pd.to_numeric(
    target_reference["reference_target"],
    errors="coerce"
)


# ============================================================
# IDENTIFY TARGET COLUMN IN STAGE 2.2
# ============================================================

print("")
print("=" * 70)
print("IDENTIFYING T+1 TARGET IN STAGE 2.2")
print("=" * 70)


target_candidates = []

for column in [
    "target_t_plus_1",
    "target_t_plus_1_x",
    "target_t_plus_1_y"
]:

    if column in temporal.columns:

        target_candidates.append(column)


print(
    "Target candidates found:"
)

for column in target_candidates:

    print(
        "  -",
        column
    )


if len(target_candidates) == 0:

    print("")
    print(
        "ERROR: No target_t_plus_1 column exists in Stage 2.2."
    )

    print("")
    print(
        "Columns containing target:"
    )

    for column in temporal.columns:

        if "target" in column.lower():

            print(
                "  -",
                column
            )

    raise SystemExit(1)


# ============================================================
# FIND WHICH TARGET CANDIDATE MATCHES STAGE 1.3
# ============================================================

matching_candidates = []


for candidate in target_candidates:

    check = temporal[
        [
            "sequence",
            "component",
            "frame_number",
            candidate
        ]
    ].copy()


    check["component"] = pd.to_numeric(
        check["component"],
        errors="coerce"
    )


    check["frame_number"] = pd.to_numeric(
        check["frame_number"],
        errors="coerce"
    )


    check = check.merge(
        target_reference,
        left_on=[
            "sequence",
            "component",
            "frame_number"
        ],
        right_on=[
            "sequence",
            "component",
            "reference_current_frame"
        ],
        how="inner"
    )


    if len(check) == 0:
        continue


    check["difference"] = (
        pd.to_numeric(
            check[candidate],
            errors="coerce"
        )
        -
        check["reference_target"]
    ).abs()


    max_difference = float(
        check["difference"].max()
    )


    mismatch_count = int(
        (
            check["difference"] > 1e-12
        ).sum()
    )


    matching_candidates.append(
        {
            "candidate": candidate,
            "records": len(check),
            "max_difference": max_difference,
            "mismatch_count": mismatch_count
        }
    )


if len(matching_candidates) == 0:

    print("")
    print(
        "ERROR: Could not match Stage 2.2 target to Stage 1.3."
    )

    raise SystemExit(1)


candidate_df = pd.DataFrame(
    matching_candidates
)


print("")
print(
    candidate_df.to_string(
        index=False
    )
)


valid_candidates = candidate_df[
    (
        candidate_df["records"]
        ==
        len(primary_targets)
    )
    &
    (
        candidate_df["mismatch_count"]
        ==
        0
    )
]


if len(valid_candidates) != 1:

    print("")
    print(
        "ERROR: Target column could not be uniquely verified."
    )

    raise SystemExit(1)


target_column = valid_candidates.iloc[0]["candidate"]


print("")
print(
    "Verified Stage 2.2 target column:",
    target_column
)


# ============================================================
# STANDARDIZE TARGET COLUMN
# ============================================================

if target_column != "target_t_plus_1":

    temporal["target_t_plus_1"] = temporal[
        target_column
    ]


# ============================================================
# FINAL TARGET VERIFICATION
# ============================================================

print("")
print("=" * 70)
print("TARGET VERIFICATION")
print("=" * 70)


target_check = temporal[
    [
        "sequence",
        "component",
        "frame_number",
        "target_t_plus_1"
    ]
].copy()


target_check["component"] = pd.to_numeric(
    target_check["component"],
    errors="coerce"
)

target_check["frame_number"] = pd.to_numeric(
    target_check["frame_number"],
    errors="coerce"
)


target_check = target_check.merge(
    target_reference,
    left_on=[
        "sequence",
        "component",
        "frame_number"
    ],
    right_on=[
        "sequence",
        "component",
        "reference_current_frame"
    ],
    how="inner",
    validate="one_to_one"
)


print(
    "Matched target records:",
    len(target_check)
)


if len(target_check) != len(primary_targets):

    print(
        "ERROR: Target record count mismatch."
    )

    raise SystemExit(1)


target_check["target_difference"] = (
    pd.to_numeric(
        target_check["target_t_plus_1"],
        errors="coerce"
    )
    -
    target_check["reference_target"]
).abs()


max_difference = float(
    target_check["target_difference"].max()
)


mismatch_count = int(
    (
        target_check["target_difference"]
        >
        1e-12
    ).sum()
)


print(
    "Maximum target difference:",
    max_difference
)

print(
    "Target mismatches:",
    mismatch_count
)


# ============================================================
# FUTURE FRAME CHECK
# ============================================================

target_check["frame_gap"] = (
    target_check["reference_future_frame"]
    -
    target_check["reference_current_frame"]
)


invalid_gap_count = int(
    (
        target_check["frame_gap"]
        !=
        1
    ).sum()
)


invalid_direction_count = int(
    (
        target_check["reference_future_frame"]
        <=
        target_check["reference_current_frame"]
    ).sum()
)


print(
    "Invalid target frame gaps:",
    invalid_gap_count
)

print(
    "Invalid target directions:",
    invalid_direction_count
)


if mismatch_count != 0:

    print(
        "ERROR: Target verification failed."
    )

    raise SystemExit(1)


if invalid_gap_count != 0:

    print(
        "ERROR: Target frame gap is not 1."
    )

    raise SystemExit(1)


if invalid_direction_count != 0:

    print(
        "ERROR: Invalid future direction."
    )

    raise SystemExit(1)


target_check["status"] = "MATCH"


target_check.to_csv(
    VERIFICATION_FILE,
    index=False
)


print(
    "Target verification: PASS"
)


# ============================================================
# ASSIGN ML SPLIT
# ============================================================

def get_ml_split(component):

    component = int(component)

    if component in train_set:
        return "TRAIN"

    if component in val_set:
        return "VAL"

    if component in test_set:
        return "TEST"

    return None


temporal["component"] = pd.to_numeric(
    temporal["component"],
    errors="coerce"
)

temporal["frame_number"] = pd.to_numeric(
    temporal["frame_number"],
    errors="coerce"
)


temporal["ml_split"] = temporal[
    "component"
].apply(
    get_ml_split
)


unknown_split_count = int(
    temporal["ml_split"].isna().sum()
)


print("")
print(
    "Records without split assignment:",
    unknown_split_count
)


if unknown_split_count != 0:

    print(
        "ERROR: Unknown components found."
    )

    raise SystemExit(1)


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
# TEMPORAL FEATURES
# ============================================================

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


# ============================================================
# FEATURE EXISTENCE CHECK
# ============================================================

missing_current = [
    x
    for x in current_features
    if x not in temporal.columns
]


missing_temporal = [
    x
    for x in temporal_features
    if x not in temporal.columns
]


if missing_current:

    print("")
    print("ERROR: Missing current features:")

    for x in missing_current:
        print("  -", x)

    raise SystemExit(1)


if missing_temporal:

    print("")
    print("ERROR: Missing temporal features:")

    for x in missing_temporal:
        print("  -", x)

    raise SystemExit(1)


all_features = (
    current_features
    +
    temporal_features
)


# ============================================================
# FUTURE FEATURE NAME CHECK
# ============================================================

future_features = []

for feature in all_features:

    name = feature.lower()

    if (
        "future" in name
        or
        "target" in name
        or
        "t_plus" in name
    ):

        future_features.append(feature)


print("")
print("=" * 70)
print("FUTURE LEAKAGE CHECK")
print("=" * 70)

print(
    "Current features:",
    len(current_features)
)

print(
    "Temporal features:",
    len(temporal_features)
)

print(
    "Total features:",
    len(all_features)
)

print(
    "Future-information features:",
    len(future_features)
)


if future_features:

    print(
        "ERROR: Future-information features detected."
    )

    for feature in future_features:
        print("  -", feature)

    raise SystemExit(1)


# ============================================================
# BUILD MODEL DATA
# ============================================================

model_columns = [
    "sequence",
    "component",
    "frame_number",
    "ml_split",
    "target_t_plus_1",
    "crack_area_ratio"
]

model_columns.extend(
    all_features
)


model_columns = list(
    dict.fromkeys(
        model_columns
    )
)


model_data = temporal[
    model_columns
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for feature in all_features:

    model_data[feature] = pd.to_numeric(
        model_data[feature],
        errors="coerce"
    )


model_data["target_t_plus_1"] = pd.to_numeric(
    model_data["target_t_plus_1"],
    errors="coerce"
)


# ============================================================
# CURRENT COMPLETE
# ============================================================

model_data["current_complete"] = (
    model_data[
        current_features
    ]
    .notna()
    .all(axis=1)
)


# ============================================================
# TEMPORAL COMPLETE
# ============================================================

model_data["temporal_complete"] = (
    model_data[
        temporal_features
    ]
    .notna()
    .all(axis=1)
)


current_data = model_data[
    model_data["current_complete"]
].copy()


temporal_data = model_data[
    model_data["temporal_complete"]
].copy()


print("")
print("=" * 70)
print("MODEL DATASET")
print("=" * 70)

print(
    "Total records:",
    len(model_data)
)

print(
    "Current-only complete:",
    len(current_data)
)

print(
    "Temporal complete:",
    len(temporal_data)
)


# ============================================================
# SPLIT COUNTS
# ============================================================

print("")
print("Current-only split counts:")

print(
    current_data[
        "ml_split"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print("")
print("Temporal split counts:")

print(
    temporal_data[
        "ml_split"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SPLIT INTEGRITY
# ============================================================

for name, dataset in [
    ("CURRENT", current_data),
    ("TEMPORAL", temporal_data)
]:

    component_conflicts = int(
        (
            dataset
            .groupby("component")["ml_split"]
            .nunique()
            > 1
        ).sum()
    )


    sequence_conflicts = int(
        (
            dataset
            .groupby("sequence")["ml_split"]
            .nunique()
            > 1
        ).sum()
    )


    print("")
    print(
        name,
        "component split conflicts:",
        component_conflicts
    )

    print(
        name,
        "sequence split conflicts:",
        sequence_conflicts
    )


    if component_conflicts != 0:
        print(
            "ERROR: Component split leakage."
        )
        raise SystemExit(1)


    if sequence_conflicts != 0:
        print(
            "ERROR: Sequence split leakage."
        )
        raise SystemExit(1)


# ============================================================
# MODEL FACTORY
# ============================================================

def make_models():

    return {

        "LinearRegression":
            LinearRegression(),

        "Ridge":
            Ridge(
                alpha=1.0
            ),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt"
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def train_and_evaluate(
    feature_set,
    model_name,
    model,
    dataset,
    feature_columns
):

    train = dataset[
        dataset["ml_split"] == "TRAIN"
    ].copy()

    val = dataset[
        dataset["ml_split"] == "VAL"
    ].copy()

    test = dataset[
        dataset["ml_split"] == "TEST"
    ].copy()


    X_train = train[
        feature_columns
    ]

    y_train = train[
        "target_t_plus_1"
    ]


    X_val = val[
        feature_columns
    ]

    y_val = val[
        "target_t_plus_1"
    ]


    X_test = test[
        feature_columns
    ]

    y_test = test[
        "target_t_plus_1"
    ]


    print("")
    print(
        "Training:",
        feature_set,
        "|",
        model_name
    )


    model.fit(
        X_train,
        y_train
    )


    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    rows = []


    for split_name, actual, prediction in [
        (
            "TRAIN",
            y_train,
            train_pred
        ),
        (
            "VAL",
            y_val,
            val_pred
        ),
        (
            "TEST",
            y_test,
            test_pred
        )
    ]:

        mae = mean_absolute_error(
            actual,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                prediction
            )
        )

        r2 = r2_score(
            actual,
            prediction
        )


        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "split": split_name,
                "records": len(actual),
                "feature_count": len(feature_columns),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }
        )


    prediction_table = pd.DataFrame(
        {
            "feature_set": feature_set,
            "model": model_name,
            "sequence": test["sequence"].values,
            "component": test["component"].values,
            "current_frame": test["frame_number"].values,
            "actual_target": y_test.values,
            "prediction": test_pred
        }
    )


    return rows, prediction_table


# ============================================================
# RUN EXPERIMENTS
# ============================================================

all_results = []
all_predictions = []


print("")
print("=" * 70)
print("CURRENT-ONLY ML")
print("=" * 70)


for model_name, model in make_models().items():

    rows, predictions = train_and_evaluate(
        "current_only",
        model_name,
        model,
        current_data,
        current_features
    )

    all_results.extend(rows)

    all_predictions.append(
        predictions
    )


print("")
print("=" * 70)
print("CURRENT + TEMPORAL ML")
print("=" * 70)


for model_name, model in make_models().items():

    rows, predictions = train_and_evaluate(
        "current_plus_temporal",
        model_name,
        model,
        temporal_data,
        all_features
    )

    all_results.extend(rows)

    all_predictions.append(
        predictions
    )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

temporal_test = temporal_data[
    temporal_data["ml_split"] == "TEST"
].copy()


persistence_actual = (
    temporal_test[
        "target_t_plus_1"
    ]
    .to_numpy()
)


persistence_prediction = (
    temporal_test[
        "crack_area_ratio"
    ]
    .to_numpy()
)


persistence_mae = mean_absolute_error(
    persistence_actual,
    persistence_prediction
)


persistence_rmse = np.sqrt(
    mean_squared_error(
        persistence_actual,
        persistence_prediction
    )
)


persistence_r2 = r2_score(
    persistence_actual,
    persistence_prediction
)


all_results.append(
    {
        "feature_set": "baseline",
        "model": "Persistence",
        "split": "TEST",
        "records": len(temporal_test),
        "feature_count": 0,
        "MAE": float(persistence_mae),
        "RMSE": float(persistence_rmse),
        "R2": float(persistence_r2)
    }
)


all_predictions.append(
    pd.DataFrame(
        {
            "feature_set": "baseline",
            "model": "Persistence",
            "sequence": temporal_test["sequence"].values,
            "component": temporal_test["component"].values,
            "current_frame": temporal_test["frame_number"].values,
            "actual_target": persistence_actual,
            "prediction": persistence_prediction
        }
    )
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


predictions_df["residual"] = (
    predictions_df["prediction"]
    -
    predictions_df["actual_target"]
)


predictions_df["absolute_error"] = (
    predictions_df["residual"]
    .abs()
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

validation_results = results_df[
    results_df["split"] == "VAL"
].copy()


validation_results = validation_results.sort_values(
    "MAE"
)


print("")
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    validation_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# TEST RESULTS
# ============================================================

test_results = results_df[
    results_df["split"] == "TEST"
].copy()


test_results = test_results.sort_values(
    "MAE"
)


print("")
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    test_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ============================================================
# CURRENT VS TEMPORAL
# ============================================================

improvement_rows = []


for model_name in [
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "GradientBoosting"
]:

    current_rows = test_results[
        (
            test_results["model"]
            ==
            model_name
        )
        &
        (
            test_results["feature_set"]
            ==
            "current_only"
        )
    ]


    temporal_rows = test_results[
        (
            test_results["model"]
            ==
            model_name
        )
        &
        (
            test_results["feature_set"]
            ==
            "current_plus_temporal"
        )
    ]


    if len(current_rows) != 1:
        continue

    if len(temporal_rows) != 1:
        continue


    current_mae = float(
        current_rows.iloc[0]["MAE"]
    )

    temporal_mae = float(
        temporal_rows.iloc[0]["MAE"]
    )

    current_rmse = float(
        current_rows.iloc[0]["RMSE"]
    )

    temporal_rmse = float(
        temporal_rows.iloc[0]["RMSE"]
    )


    mae_improvement = (
        (
            current_mae
            -
            temporal_mae
        )
        /
        current_mae
    ) * 100.0


    rmse_improvement = (
        (
            current_rmse
            -
            temporal_rmse
        )
        /
        current_rmse
    ) * 100.0


    improvement_rows.append(
        {
            "model": model_name,
            "current_only_MAE": current_mae,
            "current_plus_temporal_MAE": temporal_mae,
            "MAE_improvement_percent": mae_improvement,
            "current_only_RMSE": current_rmse,
            "current_plus_temporal_RMSE": temporal_rmse,
            "RMSE_improvement_percent": rmse_improvement
        }
    )


improvement_df = pd.DataFrame(
    improvement_rows
)


print("")
print("=" * 70)
print("CURRENT VS TEMPORAL IMPROVEMENT")
print("=" * 70)

if len(improvement_df) > 0:

    print(
        improvement_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}"
        )
    )

else:

    print(
        "No comparison rows generated."
    )


# ============================================================
# BEST TEMPORAL MODEL
# ============================================================

temporal_test_results = test_results[
    test_results["feature_set"]
    ==
    "current_plus_temporal"
].copy()


best_temporal_index = temporal_test_results[
    "MAE"
].idxmin()


best_temporal = temporal_test_results.loc[
    best_temporal_index
]


best_temporal_model = str(
    best_temporal["model"]
)


best_temporal_mae = float(
    best_temporal["MAE"]
)


best_temporal_rmse = float(
    best_temporal["RMSE"]
)


best_temporal_r2 = float(
    best_temporal["R2"]
)


# ============================================================
# TEMPORAL VS PERSISTENCE
# ============================================================

temporal_vs_persistence = (
    (
        persistence_mae
        -
        best_temporal_mae
    )
    /
    persistence_mae
) * 100.0


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


predictions_df.to_csv(
    PREDICTION_FILE,
    index=False
)


improvement_df.to_csv(
    IMPROVEMENT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "stage": "2.3",
    "status": "PASS",
    "target": "crack_area_ratio at t+1",

    "target_source": (
        "Stage 2.2 target independently verified "
        "against Stage 1.3"
    ),

    "target_column": target_column,

    "target_verification": "PASS",

    "matched_target_records": int(
        len(target_check)
    ),

    "target_mismatch_count": mismatch_count,

    "maximum_target_difference": max_difference,

    "invalid_target_frame_gaps": invalid_gap_count,

    "invalid_target_directions": invalid_direction_count,

    "train_components": train_components,
    "val_components": val_components,
    "test_components": test_components,

    "current_feature_count": len(
        current_features
    ),

    "temporal_feature_count": len(
        temporal_features
    ),

    "total_feature_count": len(
        all_features
    ),

    "total_records": int(
        len(model_data)
    ),

    "current_complete_records": int(
        len(current_data)
    ),

    "temporal_complete_records": int(
        len(temporal_data)
    ),

    "persistence_test_MAE": float(
        persistence_mae
    ),

    "persistence_test_RMSE": float(
        persistence_rmse
    ),

    "persistence_test_R2": float(
        persistence_r2
    ),

    "best_temporal_model": best_temporal_model,

    "best_temporal_test_MAE": best_temporal_mae,

    "best_temporal_test_RMSE": best_temporal_rmse,

    "best_temporal_test_R2": best_temporal_r2,

    "best_temporal_vs_persistence_MAE_improvement_percent": float(
        temporal_vs_persistence
    ),

    "validation_results": validation_results.to_dict(
        orient="records"
    ),

    "test_results": test_results.to_dict(
        orient="records"
    ),

    "current_vs_temporal": improvement_df.to_dict(
        orient="records"
    ),

    "outputs": {
        "results": RESULT_FILE,
        "predictions": PREDICTION_FILE,
        "improvement": IMPROVEMENT_FILE,
        "target_verification": VERIFICATION_FILE,
        "summary": SUMMARY_FILE
    }
}


with open(
    SUMMARY_FILE,
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
# FINAL SUMMARY
# ============================================================

print("")
print("=" * 70)
print("STAGE 2.3 SUMMARY")
print("=" * 70)

print("")

print(
    "Target verification: PASS"
)

print(
    "Target records:",
    len(target_check)
)

print(
    "Maximum target difference:",
    f"{max_difference:.12f}"
)

print(
    "Target mismatches:",
    mismatch_count
)

print("")

print(
    "Current feature count:",
    len(current_features)
)

print(
    "Temporal feature count:",
    len(temporal_features)
)

print(
    "Total feature count:",
    len(all_features)
)

print("")

print(
    "Current complete records:",
    len(current_data)
)

print(
    "Temporal complete records:",
    len(temporal_data)
)

print("")

print(
    "Persistence TEST MAE:",
    f"{persistence_mae:.8f}"
)

print(
    "Best temporal model:",
    best_temporal_model
)

print(
    "Best temporal TEST MAE:",
    f"{best_temporal_mae:.8f}"
)

print(
    "Best temporal TEST RMSE:",
    f"{best_temporal_rmse:.8f}"
)

print(
    "Best temporal TEST R2:",
    f"{best_temporal_r2:.8f}"
)

print(
    "Temporal vs persistence MAE improvement:",
    f"{temporal_vs_persistence:.2f}%"
)

print("")

print(
    "Results:",
    RESULT_FILE
)

print(
    "Predictions:",
    PREDICTION_FILE
)

print(
    "Improvement:",
    IMPROVEMENT_FILE
)

print(
    "Target verification:",
    VERIFICATION_FILE
)

print(
    "Summary:",
    SUMMARY_FILE
)

print("")

print(
    "FINAL STATUS: PASS"
)

print("")