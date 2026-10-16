import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"


# ============================================================================
# INPUT FILES
# ============================================================================

FEATURE_FILE = (
    AUDIT_DIR /
    "stage_1_0_damage_features.csv"
)

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_15_physical_regression_targets.csv"
)

ASSIGNMENT_FILE = (
    AUDIT_DIR /
    "stage_0_9d1_component_assignments.csv"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

RESULT_FILE = (
    AUDIT_DIR /
    "stage_3_16_physical_regression_results.csv"
)

PREDICTION_FILE = (
    AUDIT_DIR /
    "stage_3_16_physical_regression_predictions.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_16_physical_regression_summary.json"
)


# ============================================================================
# HELPERS
# ============================================================================

def safe_json(value):

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):

        return {
            str(k): safe_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            safe_json(v)
            for v in value
        ]

    return value


def find_column(
    df,
    candidates,
    required=True,
):

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise KeyError(
            "Could not find any of: "
            + str(candidates)
        )

    return None


def extract_frame_number(value):

    text = str(value)

    import re

    matches = re.findall(
        r"(?<!\d)(\d+)(?=\.(?:jpg|jpeg|png)$)",
        text.lower(),
    )

    if matches:

        return int(
            matches[-1]
        )

    matches = re.findall(
        r"\d+",
        text,
    )

    if matches:

        return int(
            matches[-1]
        )

    return np.nan


def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================================
# LOAD FEATURES
# ============================================================================

def load_features():

    print()
    print("=" * 78)
    print(
        "LOADING CURRENT PHYSICAL FEATURES"
    )
    print("=" * 78)

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Raw feature records:",
        len(df),
    )

    sequence_col = find_column(
        df,
        ["sequence"],
    )

    component_col = find_column(
        df,
        ["component"],
    )

    filename_col = find_column(
        df,
        [
            "canonical_image_filename",
            "original_image_filename",
            "source_image",
        ],
    )

    ratio_col = find_column(
        df,
        [
            "crack_area_ratio",
        ],
    )

    df["sequence"] = (
        df[sequence_col]
        .astype(str)
    )

    df["component"] = pd.to_numeric(
        df[component_col],
        errors="coerce",
    )

    df["current_frame"] = (
        df[filename_col]
        .apply(
            extract_frame_number
        )
    )

    df["crack_area_ratio"] = pd.to_numeric(
        df[ratio_col],
        errors="coerce",
    )

    if df[
        "current_frame"
    ].isna().any():

        raise RuntimeError(
            "Frame extraction failed."
        )

    if df[
        "crack_area_ratio"
    ].isna().any():

        raise RuntimeError(
            "Missing crack_area_ratio."
        )

    print(
        "Sequences:",
        df["sequence"].nunique(),
    )

    print(
        "Missing frame numbers:",
        int(
            df["current_frame"].isna().sum()
        ),
    )

    print(
        "Missing damage ratio:",
        int(
            df["crack_area_ratio"].isna().sum()
        ),
    )

    return df


# ============================================================================
# LOAD PHYSICAL TARGET
# ============================================================================

def load_target():

    print()
    print("=" * 78)
    print(
        "LOADING STAGE 3.15 PHYSICAL TARGET"
    )
    print("=" * 78)

    target = pd.read_csv(
        TARGET_FILE
    )

    print(
        "Target records:",
        len(target),
    )

    sequence_col = find_column(
        target,
        ["sequence"],
    )

    frame_col = find_column(
        target,
        ["current_frame"],
    )

    target[
        "sequence"
    ] = (
        target[
            sequence_col
        ]
        .astype(str)
    )

    target[
        "current_frame"
    ] = pd.to_numeric(
        target[
            frame_col
        ],
        errors="coerce",
    )

    target_col = find_column(
        target,
        [
            "future_crack_area_ratio_t1",
        ],
    )

    target[
        "physical_target"
    ] = pd.to_numeric(
        target[
            target_col
        ],
        errors="coerce",
    )

    target = target[
        [
            "sequence",
            "current_frame",
            "physical_target",
        ]
    ].copy()

    target = target.dropna(
        subset=[
            "physical_target",
        ]
    )

    duplicate_count = int(
        target[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate target keys:",
        duplicate_count,
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Duplicate target keys detected."
        )

    print(
        "Valid physical targets:",
        len(target),
    )

    return target


# ============================================================================
# LOAD COMPONENT SPLITS
# ============================================================================

def load_assignments():

    print()
    print("=" * 78)
    print(
        "LOADING LEAKAGE-SAFE COMPONENT SPLITS"
    )
    print("=" * 78)

    assignments = pd.read_csv(
        ASSIGNMENT_FILE
    )

    print(
        "Assignment records:",
        len(assignments),
    )

    component_col = find_column(
        assignments,
        [
            "component",
            "component_id",
        ],
    )

    split_col = find_column(
        assignments,
        [
            "split",
            "ml_split",
            "assigned_split",
        ],
    )

    assignments[
        "component"
    ] = pd.to_numeric(
        assignments[
            component_col
        ],
        errors="coerce",
    )

    assignments[
        "ml_split"
    ] = (
        assignments[
            split_col
        ]
        .astype(str)
        .str.upper()
    )

    assignments = assignments[
        [
            "component",
            "ml_split",
        ]
    ].drop_duplicates()

    print(
        assignments.to_string(
            index=False
        )
    )

    return assignments


# ============================================================================
# BUILD DATASET
# ============================================================================

def build_dataset(
    features,
    target,
    assignments,
):

    print()
    print("=" * 78)
    print(
        "BUILDING PHYSICAL REGRESSION DATASET"
    )
    print("=" * 78)

    data = features.merge(
        target,
        on=[
            "sequence",
            "current_frame",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "After feature-target merge:",
        len(data),
    )

    data = data.merge(
        assignments,
        on="component",
        how="left",
        validate="many_to_one",
    )

    print(
        "Missing split:",
        int(
            data[
                "ml_split"
            ].isna().sum()
        ),
    )

    if data[
        "ml_split"
    ].isna().any():

        raise RuntimeError(
            "Some records have no ML split."
        )

    print()
    print(
        "Split distribution:"
    )

    print(
        data[
            "ml_split"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return data


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_current_features(
    data
):

    print()
    print("=" * 78)
    print(
        "SELECTING CURRENT-STATE FEATURES"
    )
    print("=" * 78)

    excluded = {
        "sequence",
        "component",
        "current_frame",
        "physical_target",
        "ml_split",
    }

    candidate_columns = []

    for column in data.columns:

        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            continue

        candidate_columns.append(
            column
        )

    # Exclude fields that directly encode filenames,
    # source paths, JSON names, or record identifiers.
    forbidden_tokens = [
        "record_id",
        "filename",
        "source_",
        "json",
        "canonical",
        "path",
    ]

    feature_columns = []

    for column in candidate_columns:

        lower = column.lower()

        if any(
            token in lower
            for token in forbidden_tokens
        ):
            continue

        feature_columns.append(
            column
        )

    # Explicitly remove future-looking fields if they
    # happen to exist in the source.
    future_tokens = [
        "future",
        "target",
        "t_plus",
        "t+",
        "delta",
    ]

    safe_features = []

    for column in feature_columns:

        lower = column.lower()

        if any(
            token in lower
            for token in future_tokens
        ):
            continue

        safe_features.append(
            column
        )

    feature_columns = safe_features

    print(
        "Current-state feature count:",
        len(feature_columns),
    )

    print()

    for column in feature_columns:

        print(
            "  ",
            column,
        )

    if len(feature_columns) == 0:

        raise RuntimeError(
            "No valid current-state features."
        )

    return feature_columns


# ============================================================================
# PREPARE MATRICES
# ============================================================================

def prepare_split(
    data,
    feature_columns,
):

    train = data[
        data[
            "ml_split"
        ] == "TRAIN"
    ].copy()

    val = data[
        data[
            "ml_split"
        ] == "VAL"
    ].copy()

    test = data[
        data[
            "ml_split"
        ] == "TEST"
    ].copy()

    X_train = train[
        feature_columns
    ].copy()

    X_val = val[
        feature_columns
    ].copy()

    X_test = test[
        feature_columns
    ].copy()

    y_train = train[
        "physical_target"
    ].copy()

    y_val = val[
        "physical_target"
    ].copy()

    y_test = test[
        "physical_target"
    ].copy()

    # Median imputation based only on training data.
    train_medians = (
        X_train
        .median()
    )

    X_train = X_train.fillna(
        train_medians
    )

    X_val = X_val.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # Remove non-finite values.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(
        train_medians
    )

    X_val = X_val.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(
        train_medians
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(
        train_medians
    )

    print()
    print(
        "TRAIN:",
        len(train),
    )

    print(
        "VAL:",
        len(val),
    )

    print(
        "TEST:",
        len(test),
    )

    return (
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================================
# PERSISTENCE BASELINE
# ============================================================================

def evaluate_persistence(
    train,
    val,
    test,
):

    print()
    print("=" * 78)
    print(
        "PERSISTENCE BASELINE"
    )
    print("=" * 78)

    results = []

    for split_name, frame in [
        ("TRAIN", train),
        ("VAL", val),
        ("TEST", test),
    ]:

        y_true = frame[
            "physical_target"
        ].values

        y_pred = frame[
            "crack_area_ratio"
        ].values

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        results.append(
            {
                "model":
                    "PERSISTENCE",

                "split":
                    split_name,

                "feature_group":
                    "CURRENT_STATE",

                "feature_count":
                    1,

                **metrics,
            }
        )

        print(
            split_name,
            metrics,
        )

    return results


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_count,
):

    print()
    print(
        "-" * 78
    )

    print(
        "MODEL:",
        model_name,
    )

    model.fit(
        X_train,
        y_train,
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

    results = []

    for split_name, y_true, y_pred in [
        (
            "TRAIN",
            y_train,
            train_pred,
        ),
        (
            "VAL",
            y_val,
            val_pred,
        ),
        (
            "TEST",
            y_test,
            test_pred,
        ),
    ]:

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        results.append(
            {
                "model":
                    model_name,

                "split":
                    split_name,

                "feature_group":
                    "CURRENT_STATE",

                "feature_count":
                    feature_count,

                **metrics,
            }
        )

        print(
            split_name,
            metrics,
        )

    return (
        results,
        val_pred,
        test_pred,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.16 - PHYSICAL DAMAGE REGRESSION BASELINE"
    )
    print("=" * 78)

    print()
    print(
        "Project:"
    )

    print(
        PROJECT_ROOT
    )

    # ========================================================================
    # CHECK FILES
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CHECKING INPUT FILES"
    )
    print("=" * 78)

    for label, path in [
        (
            "physical_features",
            FEATURE_FILE,
        ),
        (
            "physical_targets",
            TARGET_FILE,
        ),
        (
            "component_assignments",
            ASSIGNMENT_FILE,
        ),
    ]:

        status = (
            "FOUND"
            if path.exists()
            else "NOT FOUND"
        )

        print(
            f"{label:<24}: {status}"
        )

        if path.exists():

            print(
                "    ",
                path,
            )

        else:

            raise FileNotFoundError(
                str(path)
            )

    # ========================================================================
    # LOAD
    # ========================================================================

    features = load_features()

    target = load_target()

    assignments = load_assignments()

    # ========================================================================
    # DATASET
    # ========================================================================

    data = build_dataset(
        features,
        target,
        assignments,
    )

    # ========================================================================
    # COMPONENT INTEGRITY
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CHECKING COMPONENT INTEGRITY"
    )
    print("=" * 78)

    conflicts = (
        data.groupby(
            "component"
        )[
            "ml_split"
        ]
        .nunique()
    )

    conflicting_components = (
        conflicts[
            conflicts > 1
        ]
    )

    print(
        "Component conflicts:",
        len(
            conflicting_components
        ),
    )

    if len(
        conflicting_components
    ) > 0:

        raise RuntimeError(
            "Component split leakage detected."
        )

    # ========================================================================
    # FEATURES
    # ========================================================================

    feature_columns = (
        select_current_features(
            data
        )
    )

    # ========================================================================
    # MATRICES
    # ========================================================================

    (
        train,
        val,
        test,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = prepare_split(
        data,
        feature_columns,
    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    all_results = []

    persistence_results = (
        evaluate_persistence(
            train,
            val,
            test,
        )
    )

    all_results.extend(
        persistence_results
    )

    # ========================================================================
    # MODELS
    # ========================================================================

    models = [

        (
            "LINEAR_REGRESSION",
            LinearRegression(),
        ),

        (
            "RIDGE",
            Ridge(
                alpha=1.0
            ),
        ),

        (
            "RANDOM_FOREST",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),

        (
            "GRADIENT_BOOSTING",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ]

    prediction_records = []

    # Persistence test predictions.
    persistence_test = test[
        [
            "sequence",
            "component",
            "current_frame",
            "crack_area_ratio",
            "physical_target",
        ]
    ].copy()

    persistence_test[
        "model"
    ] = "PERSISTENCE"

    persistence_test[
        "prediction"
    ] = persistence_test[
        "crack_area_ratio"
    ]

    persistence_test[
        "absolute_error"
    ] = (
        persistence_test[
            "prediction"
        ]
        -
        persistence_test[
            "physical_target"
        ]
    ).abs()

    prediction_records.append(
        persistence_test
    )

    model_objects = {}

    for model_name, model in models:

        (
            results,
            val_pred,
            test_pred,
        ) = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            len(feature_columns),
        )

        all_results.extend(
            results
        )

        model_objects[
            model_name
        ] = model

        model_test = test[
            [
                "sequence",
                "component",
                "current_frame",
                "crack_area_ratio",
                "physical_target",
            ]
        ].copy()

        model_test[
            "model"
        ] = model_name

        model_test[
            "prediction"
        ] = test_pred

        model_test[
            "absolute_error"
        ] = (
            model_test[
                "prediction"
            ]
            -
            model_test[
                "physical_target"
            ]
        ).abs()

        prediction_records.append(
            model_test
        )

    # ========================================================================
    # RESULTS TABLE
    # ========================================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        [
            "split",
            "MAE",
        ]
    )

    print()
    print("=" * 78)
    print(
        "MODEL COMPARISON"
    )
    print("=" * 78)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    # ========================================================================
    # PREDICTIONS
    # ========================================================================

    predictions_df = pd.concat(
        prediction_records,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    # ========================================================================
    # BEST VALIDATION MODEL
    # ========================================================================

    val_results = results_df[
        results_df[
            "split"
        ] == "VAL"
    ].copy()

    val_results = val_results.sort_values(
        "MAE"
    )

    best_validation_model = (
        val_results.iloc[0][
            "model"
        ]
    )

    best_validation_mae = float(
        val_results.iloc[0][
            "MAE"
        ]
    )

    best_validation_r2 = float(
        val_results.iloc[0][
            "R2"
        ]
    )

    test_results = results_df[
        results_df[
            "split"
        ] == "TEST"
    ].copy()

    persistence_test_mae = float(
        test_results[
            test_results[
                "model"
            ] == "PERSISTENCE"
        ].iloc[0][
            "MAE"
        ]
    )

    best_test_row = (
        test_results
        .sort_values(
            "MAE"
        )
        .iloc[0]
    )

    best_test_model = (
        best_test_row[
            "model"
        ]
    )

    best_test_mae = float(
        best_test_row[
            "MAE"
        ]
    )

    best_test_r2 = float(
        best_test_row[
            "R2"
        ]
    )

    # ========================================================================
    # PERSISTENCE IMPROVEMENT
    # ========================================================================

    non_persistence_test = (
        test_results[
            test_results[
                "model"
            ] != "PERSISTENCE"
        ]
        .sort_values(
            "MAE"
        )
    )

    if len(
        non_persistence_test
    ) > 0:

        best_ml_test_mae = float(
            non_persistence_test.iloc[0][
                "MAE"
            ]
        )

        improvement_vs_persistence = (
            (
                persistence_test_mae
                -
                best_ml_test_mae
            )
            /
            persistence_test_mae
            *
            100.0
        )

    else:

        best_ml_test_mae = None
        improvement_vs_persistence = None

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.16",

        "title":
            "Physical Damage Regression Baseline",

        "target":
            "crack_area_ratio(t+1)",

        "records":
            int(len(data)),

        "train_records":
            int(len(train)),

        "validation_records":
            int(len(val)),

        "test_records":
            int(len(test)),

        "sequence_count":
            int(
                data[
                    "sequence"
                ].nunique()
            ),

        "component_count":
            int(
                data[
                    "component"
                ].nunique()
            ),

        "feature_count":
            int(
                len(feature_columns)
            ),

        "feature_columns":
            feature_columns,

        "component_conflicts":
            int(
                len(
                    conflicting_components
                )
            ),

        "best_validation_model":
            str(
                best_validation_model
            ),

        "best_validation_mae":
            best_validation_mae,

        "best_validation_r2":
            best_validation_r2,

        "best_test_model":
            str(
                best_test_model
            ),

        "best_test_mae":
            best_test_mae,

        "best_test_r2":
            best_test_r2,

        "persistence_test_mae":
            persistence_test_mae,

        "best_ml_test_mae":
            best_ml_test_mae,

        "improvement_vs_persistence_percent":
            improvement_vs_persistence,

        "model_results":
            results_df.to_dict(
                orient="records"
            ),

        "decision":
            (
                "BASELINE_COMPLETE"
            ),

        "next_stage":
            (
                "TEMPORAL_PHYSICAL_REGRESSION"
            ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_json(
                summary
            ),
            file,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "STAGE 3.16 COMPLETE"
    )
    print("=" * 78)

    print(
        "Target:",
        "crack_area_ratio(t+1)",
    )

    print(
        "Features:",
        len(feature_columns),
    )

    print(
        "Best validation model:",
        best_validation_model,
    )

    print(
        "Best validation MAE:",
        f"{best_validation_mae:.8f}",
    )

    print(
        "Best test model:",
        best_test_model,
    )

    print(
        "Best test MAE:",
        f"{best_test_mae:.8f}",
    )

    print(
        "Persistence test MAE:",
        f"{persistence_test_mae:.8f}",
    )

    if improvement_vs_persistence is not None:

        print(
            "Best ML improvement vs persistence:",
            f"{improvement_vs_persistence:.4f}%",
        )

    print()
    print(
        "FINAL STATUS: PASS"
    )

    print()
    print(
        "Outputs saved to:"
    )

    print(
        AUDIT_DIR
    )


if __name__ == "__main__":
    main()