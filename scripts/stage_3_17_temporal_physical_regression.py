import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
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
# INPUTS
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
# OUTPUTS
# ============================================================================

FEATURE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_features.csv"
)

RESULT_OUTPUT = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_regression_results.csv"
)

PREDICTION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_predictions.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_regression_summary.json"
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

    import re

    text = str(value)

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


def metrics(
    y_true,
    y_pred,
):

    return {
        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
    }


# ============================================================================
# LOAD PHYSICAL FEATURES
# ============================================================================

def load_features():

    print()
    print("=" * 78)
    print(
        "LOADING PHYSICAL DAMAGE FEATURES"
    )
    print("=" * 78)

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Raw records:",
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
        ["crack_area_ratio"],
    )

    df["sequence"] = (
        df[
            sequence_col
        ]
        .astype(str)
    )

    df["component"] = pd.to_numeric(
        df[
            component_col
        ],
        errors="coerce",
    )

    df["current_frame"] = (
        df[
            filename_col
        ]
        .apply(
            extract_frame_number
        )
    )

    df["crack_area_ratio"] = pd.to_numeric(
        df[
            ratio_col
        ],
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

    df = df[
        [
            "sequence",
            "component",
            "current_frame",
            "crack_area_ratio",
        ]
    ].copy()

    df = df.sort_values(
        [
            "sequence",
            "current_frame",
        ]
    ).reset_index(
        drop=True
    )

    duplicate_keys = int(
        df[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate frame keys:",
        duplicate_keys,
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate frame keys detected."
        )

    print(
        "Sequences:",
        df[
            "sequence"
        ].nunique(),
    )

    return df


# ============================================================================
# LOAD TARGET
# ============================================================================

def load_target():

    print()
    print("=" * 78)
    print(
        "LOADING PHYSICAL T+1 TARGET"
    )
    print("=" * 78)

    target = pd.read_csv(
        TARGET_FILE
    )

    print(
        "Target records:",
        len(target),
    )

    target[
        "sequence"
    ] = (
        target[
            "sequence"
        ]
        .astype(str)
    )

    target[
        "current_frame"
    ] = pd.to_numeric(
        target[
            "current_frame"
        ],
        errors="coerce",
    )

    target[
        "physical_target"
    ] = pd.to_numeric(
        target[
            "future_crack_area_ratio_t1"
        ],
        errors="coerce",
    )

    target = target[
        [
            "sequence",
            "current_frame",
            "physical_target",
        ]
    ].dropna(
        subset=[
            "physical_target"
        ]
    )

    duplicate_keys = int(
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
        duplicate_keys,
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate target keys."
        )

    return target


# ============================================================================
# LOAD SPLITS
# ============================================================================

def load_splits():

    print()
    print("=" * 78)
    print(
        "LOADING COMPONENT SPLITS"
    )
    print("=" * 78)

    assignment = pd.read_csv(
        ASSIGNMENT_FILE
    )

    component_col = find_column(
        assignment,
        [
            "component",
            "component_id",
        ],
    )

    split_col = find_column(
        assignment,
        [
            "split",
            "ml_split",
            "assigned_split",
        ],
    )

    assignment[
        "component"
    ] = pd.to_numeric(
        assignment[
            component_col
        ],
        errors="coerce",
    )

    assignment[
        "ml_split"
    ] = (
        assignment[
            split_col
        ]
        .astype(str)
        .str.upper()
    )

    assignment = assignment[
        [
            "component",
            "ml_split",
        ]
    ].drop_duplicates()

    print(
        assignment.to_string(
            index=False
        )
    )

    return assignment


# ============================================================================
# BUILD TEMPORAL FEATURES
# ============================================================================

def build_temporal_features(
    df
):

    print()
    print("=" * 78)
    print(
        "BUILDING LEAKAGE-SAFE TEMPORAL FEATURES"
    )
    print("=" * 78)

    data = df.copy()

    grouped = data.groupby(
        "sequence"
    )[
        "crack_area_ratio"
    ]

    # ------------------------------------------------------------------------
    # LAGS
    # ------------------------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        5,
    ]:

        data[
            f"lag_{lag}_damage"
        ] = (
            grouped.shift(
                lag
            )
        )

    # ------------------------------------------------------------------------
    # CHANGES
    # ------------------------------------------------------------------------

    for lag in [
        1,
        2,
        3,
    ]:

        data[
            f"delta_{lag}_damage"
        ] = (
            data[
                "crack_area_ratio"
            ]
            -
            data[
                f"lag_{lag}_damage"
            ]
        )

    # ------------------------------------------------------------------------
    # ACCELERATION
    # ------------------------------------------------------------------------

    data[
        "acceleration_1_damage"
    ] = (
        data[
            "delta_1_damage"
        ]
        -
        data.groupby(
            "sequence"
        )[
            "delta_1_damage"
        ].shift(1)
    )

    data[
        "acceleration_2_damage"
    ] = (
        data[
            "delta_2_damage"
        ]
        -
        data.groupby(
            "sequence"
        )[
            "delta_2_damage"
        ].shift(1)
    )

    # ------------------------------------------------------------------------
    # ROLLING WINDOWS
    # ------------------------------------------------------------------------

    shifted = grouped.shift(
        1
    )

    data[
        "rolling3_mean_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            3,
            min_periods=3,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    data[
        "rolling3_std_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            3,
            min_periods=3,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    data[
        "rolling5_mean_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    data[
        "rolling5_std_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            5,
            min_periods=5,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # ------------------------------------------------------------------------
    # RECENT MIN/MAX
    # ------------------------------------------------------------------------

    data[
        "rolling5_min_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            5,
            min_periods=5,
        )
        .min()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    data[
        "rolling5_max_damage"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            5,
            min_periods=5,
        )
        .max()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # ------------------------------------------------------------------------
    # RECENT SLOPE
    # ------------------------------------------------------------------------

    def slope(values):

        values = np.asarray(
            values,
            dtype=float,
        )

        if np.isnan(
            values
        ).any():

            return np.nan

        if len(values) < 3:

            return np.nan

        x = np.arange(
            len(values),
            dtype=float,
        )

        coefficient = np.polyfit(
            x,
            values,
            1,
        )[0]

        return float(
            coefficient
        )

    data[
        "recent3_damage_slope"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            3,
            min_periods=3,
        )
        .apply(
            slope,
            raw=True,
        )
        .reset_index(
            level=0,
            drop=True,
        )
    )

    data[
        "recent5_damage_slope"
    ] = (
        shifted
        .groupby(
            data[
                "sequence"
            ]
        )
        .rolling(
            5,
            min_periods=5,
        )
        .apply(
            slope,
            raw=True,
        )
        .reset_index(
            level=0,
            drop=True,
        )
    )

    # ------------------------------------------------------------------------
    # LEAKAGE CHECK
    # ------------------------------------------------------------------------

    forbidden = []

    for column in data.columns:

        lower = column.lower()

        if (
            "future" in lower
            or "target" in lower
            or "t_plus" in lower
            or "t+" in lower
        ):

            forbidden.append(
                column
            )

    print(
        "Temporal feature count:",
        len(
            [
                c
                for c in data.columns
                if c not in {
                    "sequence",
                    "component",
                    "current_frame",
                    "crack_area_ratio",
                }
            ]
        ),
    )

    print(
        "Future-derived feature names:",
        len(forbidden),
    )

    if forbidden:

        print(
            forbidden
        )

        raise RuntimeError(
            "Future information detected in temporal features."
        )

    return data


# ============================================================================
# MERGE TARGET AND SPLITS
# ============================================================================

def build_dataset(
    temporal,
    target,
    splits,
):

    print()
    print("=" * 78)
    print(
        "MERGING TEMPORAL FEATURES WITH TARGET"
    )
    print("=" * 78)

    data = temporal.merge(
        target,
        on=[
            "sequence",
            "current_frame",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "After target merge:",
        len(data),
    )

    data = data.merge(
        splits,
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
            "Missing split assignments."
        )

    return data


# ============================================================================
# SELECT FEATURES
# ============================================================================

def select_features(
    data
):

    excluded = {
        "sequence",
        "component",
        "current_frame",
        "physical_target",
        "ml_split",
    }

    feature_columns = []

    for column in data.columns:

        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            continue

        if column == "crack_area_ratio":

            feature_columns.append(
                column
            )

            continue

        # Only our deliberately-created temporal features.
        if (
            column.startswith(
                "lag_"
            )
            or column.startswith(
                "delta_"
            )
            or column.startswith(
                "acceleration_"
            )
            or column.startswith(
                "rolling"
            )
            or column.startswith(
                "recent"
            )
        ):

            feature_columns.append(
                column
            )

    print()
    print(
        "Selected temporal features:",
        len(feature_columns),
    )

    for column in feature_columns:

        print(
            "  ",
            column,
        )

    return feature_columns


# ============================================================================
# PREPARE SPLITS
# ============================================================================

def prepare_matrices(
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

    # Only complete temporal history is used.
    train = train.dropna(
        subset=feature_columns
    ).copy()

    val = val.dropna(
        subset=feature_columns
    ).copy()

    test = test.dropna(
        subset=feature_columns
    ).copy()

    print()
    print(
        "Complete temporal records:"
    )

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

    # Safety check.
    for name, matrix in [
        ("TRAIN", X_train),
        ("VAL", X_val),
        ("TEST", X_test),
    ]:

        if not np.isfinite(
            matrix.values
        ).all():

            raise RuntimeError(
                f"Non-finite values in {name}."
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
# PERSISTENCE
# ============================================================================

def persistence_result(
    train,
    val,
    test,
):

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

        result = metrics(
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
                    "PERSISTENCE",

                "feature_count":
                    1,

                **result,
            }
        )

        print(
            split_name,
            result,
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.17 - TEMPORAL PHYSICAL REGRESSION"
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
    # INPUT CHECK
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

        print(
            f"{label:<24}: "
            + (
                "FOUND"
                if path.exists()
                else "NOT FOUND"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )

    # ========================================================================
    # LOAD
    # ========================================================================

    features = load_features()

    target = load_target()

    splits = load_splits()

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================

    temporal = build_temporal_features(
        features
    )

    # ========================================================================
    # DATASET
    # ========================================================================

    data = build_dataset(
        temporal,
        target,
        splits,
    )

    # ========================================================================
    # COMPONENT INTEGRITY
    # ========================================================================

    component_conflicts = (
        data.groupby(
            "component"
        )[
            "ml_split"
        ]
        .nunique()
    )

    conflicts = component_conflicts[
        component_conflicts > 1
    ]

    print()
    print(
        "Component split conflicts:",
        len(conflicts),
    )

    if len(conflicts) > 0:

        raise RuntimeError(
            "Component leakage detected."
        )

    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================

    feature_columns = select_features(
        data
    )

    # ========================================================================
    # PREPARE
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
    ) = prepare_matrices(
        data,
        feature_columns,
    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    results = []

    persistence = persistence_result(
        train,
        val,
        test,
    )

    results.extend(
        persistence
    )

    prediction_records = []

    # Persistence test.
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

    # ========================================================================
    # TEMPORAL MODELS
    # ========================================================================

    models = [

        (
            "TEMPORAL_RIDGE",
            Ridge(
                alpha=1.0
            ),
        ),

        (
            "TEMPORAL_RANDOM_FOREST",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),

        (
            "TEMPORAL_GRADIENT_BOOSTING",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ]

    for model_name, model in models:

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

            result = metrics(
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
                        "TEMPORAL",

                    "feature_count":
                        len(
                            feature_columns
                        ),

                    **result,
                }
            )

            print(
                split_name,
                result,
            )

        test_predictions = test[
            [
                "sequence",
                "component",
                "current_frame",
                "crack_area_ratio",
                "physical_target",
            ]
        ].copy()

        test_predictions[
            "model"
        ] = model_name

        test_predictions[
            "prediction"
        ] = test_pred

        test_predictions[
            "absolute_error"
        ] = (
            test_predictions[
                "prediction"
            ]
            -
            test_predictions[
                "physical_target"
            ]
        ).abs()

        prediction_records.append(
            test_predictions
        )

    # ========================================================================
    # RESULTS
    # ========================================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_OUTPUT,
        index=False,
    )

    predictions_df = pd.concat(
        prediction_records,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "TEMPORAL MODEL COMPARISON"
    )
    print("=" * 78)

    print(
        results_df.sort_values(
            [
                "split",
                "MAE",
            ]
        ).to_string(
            index=False
        )
    )

    # ========================================================================
    # BEST MODELS
    # ========================================================================

    val_results = results_df[
        results_df[
            "split"
        ] == "VAL"
    ].sort_values(
        "MAE"
    )

    test_results = results_df[
        results_df[
            "split"
        ] == "TEST"
    ].sort_values(
        "MAE"
    )

    best_val = val_results.iloc[0]

    best_test = test_results.iloc[0]

    persistence_test = test_results[
        test_results[
            "model"
        ] == "PERSISTENCE"
    ].iloc[0]

    ml_test = test_results[
        test_results[
            "model"
        ] != "PERSISTENCE"
    ].sort_values(
        "MAE"
    ).iloc[0]

    persistence_mae = float(
        persistence_test[
            "MAE"
        ]
    )

    best_ml_mae = float(
        ml_test[
            "MAE"
        ]
    )

    improvement = (
        (
            persistence_mae
            -
            best_ml_mae
        )
        /
        persistence_mae
        *
        100.0
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.17",

        "title":
            "Temporal Physical Regression",

        "target":
            "crack_area_ratio(t+1)",

        "raw_records":
            int(len(features)),

        "temporal_records":
            int(len(temporal)),

        "target_records":
            int(len(target)),

        "complete_temporal_train":
            int(len(train)),

        "complete_temporal_val":
            int(len(val)),

        "complete_temporal_test":
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

        "temporal_feature_count":
            int(
                len(feature_columns)
            ),

        "temporal_features":
            feature_columns,

        "component_conflicts":
            int(
                len(conflicts)
            ),

        "best_validation_model":
            str(
                best_val[
                    "model"
                ]
            ),

        "best_validation_mae":
            float(
                best_val[
                    "MAE"
                ]
            ),

        "best_validation_r2":
            float(
                best_val[
                    "R2"
                ]
            ),

        "best_test_model":
            str(
                best_test[
                    "model"
                ]
            ),

        "best_test_mae":
            float(
                best_test[
                    "MAE"
                ]
            ),

        "best_test_r2":
            float(
                best_test[
                    "R2"
                ]
            ),

        "persistence_test_mae":
            persistence_mae,

        "best_ml_test_mae":
            best_ml_mae,

        "improvement_vs_persistence_percent":
            float(
                improvement
            ),

        "model_results":
            results_df.to_dict(
                orient="records"
            ),

        "decision":
            "TEMPORAL_REGRESSION_BASELINE_COMPLETE",

        "next_stage":
            "TEMPORAL_FEATURE_IMPORTANCE_AND_ABLATION",
    }

    with open(
        SUMMARY_OUTPUT,
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
    # SAVE TEMPORAL FEATURES
    # ========================================================================

    temporal.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "STAGE 3.17 COMPLETE"
    )
    print("=" * 78)

    print(
        "Target:",
        "crack_area_ratio(t+1)",
    )

    print(
        "Temporal features:",
        len(feature_columns),
    )

    print(
        "Complete TRAIN:",
        len(train),
    )

    print(
        "Complete VAL:",
        len(val),
    )

    print(
        "Complete TEST:",
        len(test),
    )

    print(
        "Best validation model:",
        best_val[
            "model"
        ],
    )

    print(
        "Best validation MAE:",
        f"{best_val['MAE']:.8f}",
    )

    print(
        "Best test model:",
        best_test[
            "model"
        ],
    )

    print(
        "Best test MAE:",
        f"{best_test['MAE']:.8f}",
    )

    print(
        "Persistence test MAE:",
        f"{persistence_mae:.8f}",
    )

    print(
        "Best ML improvement vs persistence:",
        f"{improvement:.4f}%",
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