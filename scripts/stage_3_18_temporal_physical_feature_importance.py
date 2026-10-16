import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "stage_3_17_temporal_physical_features.csv"
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

IMPORTANCE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_18_temporal_feature_importance.csv"
)

ABLATION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_18_temporal_feature_ablation.csv"
)

GROUP_OUTPUT = (
    AUDIT_DIR /
    "stage_3_18_temporal_feature_group_importance.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_18_temporal_feature_importance_summary.json"
)


# ============================================================================
# MODEL
# ============================================================================

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "random_state": 42,
}


# ============================================================================
# JSON HELPER
# ============================================================================

def safe_json(value):

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

        if math.isnan(value) or math.isinf(value):
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


# ============================================================================
# COLUMN FINDER
# ============================================================================

def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    raise KeyError(
        "\nCould not find required column.\n"
        "Tried:\n"
        + str(candidates)
        + "\n\nAvailable columns:\n"
        + str(list(df.columns))
    )


# ============================================================================
# UNIQUE ORDERED LIST
# ============================================================================

def unique_preserve_order(columns):

    seen = set()
    result = []

    for column in columns:

        if column not in seen:

            result.append(column)
            seen.add(column)

    return result


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),

        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():

    print()
    print("=" * 78)
    print("LOADING STAGE 3.17 TEMPORAL FEATURES")
    print("=" * 78)

    temporal = pd.read_csv(
        FEATURE_FILE
    )

    target = pd.read_csv(
        TARGET_FILE
    )

    assignments = pd.read_csv(
        ASSIGNMENT_FILE
    )

    print(
        "Temporal feature records:",
        len(temporal),
    )

    print(
        "Target records:",
        len(target),
    )

    print(
        "Assignment records:",
        len(assignments),
    )

    # ------------------------------------------------------------------------
    # CHECK TEMPORAL DUPLICATE COLUMNS
    # ------------------------------------------------------------------------

    duplicate_temporal_columns = (
        temporal.columns[
            temporal.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_temporal_columns:

        raise RuntimeError(
            "Stage 3.17 temporal feature file itself contains duplicate "
            "columns:\n"
            + str(duplicate_temporal_columns)
        )

    # ------------------------------------------------------------------------
    # CHECK TARGET DUPLICATE COLUMNS
    # ------------------------------------------------------------------------

    duplicate_target_columns = (
        target.columns[
            target.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_target_columns:

        raise RuntimeError(
            "Physical target file contains duplicate columns:\n"
            + str(duplicate_target_columns)
        )

    # ------------------------------------------------------------------------
    # TEMPORAL KEYS
    # ------------------------------------------------------------------------

    if "sequence" not in temporal.columns:
        raise KeyError(
            "Stage 3.17 temporal file has no 'sequence' column."
        )

    if "current_frame" not in temporal.columns:
        raise KeyError(
            "Stage 3.17 temporal file has no 'current_frame' column."
        )

    temporal["sequence"] = (
        temporal["sequence"]
        .astype(str)
        .str.strip()
    )

    temporal["current_frame"] = pd.to_numeric(
        temporal["current_frame"],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------------------

    if "sequence" not in target.columns:
        raise KeyError(
            "Physical target file has no 'sequence' column."
        )

    if "current_frame" not in target.columns:
        raise KeyError(
            "Physical target file has no 'current_frame' column."
        )

    target["sequence"] = (
        target["sequence"]
        .astype(str)
        .str.strip()
    )

    target["current_frame"] = pd.to_numeric(
        target["current_frame"],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # FIND PHYSICAL TARGET
    # ------------------------------------------------------------------------

    target_column = find_column(
        target,
        [
            "future_crack_area_ratio_t1",
            "future_damage_t1",
            "future_crack_area_ratio",
            "target_crack_area_ratio",
        ],
    )

    print(
        "Physical target column:",
        target_column,
    )

    target["physical_target"] = pd.to_numeric(
        target[target_column],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # ASSIGNMENT SCHEMA
    # ------------------------------------------------------------------------

    print()
    print("Assignment columns:")

    for column in assignments.columns:
        print("  ", column)

    component_column = find_column(
        assignments,
        [
            "component",
            "component_id",
            "component_index",
            "connected_component",
            "connected_component_id",
            "component_number",
            "component_no",
        ],
    )

    split_column = find_column(
        assignments,
        [
            "ml_split",
            "split",
            "assigned_split",
            "dataset_split",
            "new_split",
        ],
    )

    print()
    print(
        "Detected component column:",
        component_column,
    )

    print(
        "Detected split column:",
        split_column,
    )

    assignments["component"] = pd.to_numeric(
        assignments[component_column],
        errors="coerce",
    )

    assignments["ml_split"] = (
        assignments[split_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print(
        "Missing assignment component:",
        int(
            assignments["component"]
            .isna()
            .sum()
        ),
    )

    print(
        "Missing assignment split:",
        int(
            assignments["ml_split"]
            .isna()
            .sum()
        ),
    )

    if assignments["component"].isna().any():

        raise RuntimeError(
            "Invalid component values in assignment file."
        )

    # ------------------------------------------------------------------------
    # COMPONENT CONFLICT CHECK
    # ------------------------------------------------------------------------

    component_conflicts = (
        assignments
        .groupby("component")["ml_split"]
        .nunique()
    )

    component_conflicts = (
        component_conflicts[
            component_conflicts > 1
        ]
    )

    print(
        "Component assignment conflicts:",
        len(component_conflicts),
    )

    if len(component_conflicts) > 0:

        raise RuntimeError(
            "A component belongs to more than one ML split."
        )

    assignments = assignments[
        [
            "component",
            "ml_split",
        ]
    ].drop_duplicates()

    # ------------------------------------------------------------------------
    # MERGE TARGET
    # ------------------------------------------------------------------------

    data = temporal.merge(
        target[
            [
                "sequence",
                "current_frame",
                "physical_target",
            ]
        ],
        on=[
            "sequence",
            "current_frame",
        ],
        how="inner",
        validate="one_to_one",
    )

    print()
    print(
        "After target merge:",
        len(data),
    )

    # ------------------------------------------------------------------------
    # COMPONENT
    # ------------------------------------------------------------------------

    if "component" not in data.columns:

        # Stage 3.17 may already contain component.
        # If it does not, recover it from the assignment file
        # only if the temporal file contains a usable component ID.

        possible_component = find_column(
            data,
            [
                "component_id",
                "connected_component_id",
                "component_index",
            ],
        )

        data["component"] = pd.to_numeric(
            data[possible_component],
            errors="coerce",
        )

    else:

        data["component"] = pd.to_numeric(
            data["component"],
            errors="coerce",
        )

    if data["component"].isna().any():

        raise RuntimeError(
            "Temporal records contain invalid component values."
        )

    # ------------------------------------------------------------------------
    # MERGE SPLIT
    # ------------------------------------------------------------------------

    data = data.merge(
        assignments,
        on="component",
        how="left",
        validate="many_to_one",
    )

    missing_split = int(
        data["ml_split"]
        .isna()
        .sum()
    )

    print(
        "Missing splits:",
        missing_split,
    )

    if missing_split > 0:

        raise RuntimeError(
            "Some temporal records have no ML split assignment."
        )

    return data


# ============================================================================
# TAXONOMY
# ============================================================================

def build_taxonomy():

    return {

        "CURRENT": [
            "crack_area_ratio",
        ],

        "LAG": [
            "lag_1_damage",
            "lag_2_damage",
            "lag_3_damage",
            "lag_5_damage",
        ],

        "CHANGE": [
            "delta_1_damage",
            "delta_2_damage",
            "delta_3_damage",
        ],

        "ACCELERATION": [
            "acceleration_1_damage",
            "acceleration_2_damage",
        ],

        "ROLLING": [
            "rolling3_mean_damage",
            "rolling3_std_damage",
            "rolling5_mean_damage",
            "rolling5_std_damage",
            "rolling5_min_damage",
            "rolling5_max_damage",
        ],

        "TREND": [
            "recent3_damage_slope",
            "recent5_damage_slope",
        ],
    }


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(data, taxonomy):

    print()
    print("=" * 78)
    print("VALIDATING TEMPORAL FEATURE TAXONOMY")
    print("=" * 78)

    raw_features = []

    for group, columns in taxonomy.items():

        print(
            f"{group:<16}:",
            len(columns),
        )

        raw_features.extend(
            columns
        )

    # ------------------------------------------------------------------------
    # REMOVE DUPLICATE FEATURE NAMES
    # ------------------------------------------------------------------------

    features = unique_preserve_order(
        raw_features
    )

    duplicate_feature_names = (
        len(raw_features)
        -
        len(features)
    )

    print()
    print(
        "Raw taxonomy feature entries:",
        len(raw_features),
    )

    print(
        "Unique taxonomy features:",
        len(features),
    )

    print(
        "Duplicate taxonomy entries:",
        duplicate_feature_names,
    )

    # ------------------------------------------------------------------------
    # MISSING
    # ------------------------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in data.columns
    ]

    print(
        "Missing features:",
        len(missing),
    )

    if missing:

        print(
            missing
        )

        raise RuntimeError(
            "Required Stage 3.17 temporal features are missing."
        )

    # ------------------------------------------------------------------------
    # FUTURE LEAKAGE CHECK
    # ------------------------------------------------------------------------

    future_features = []

    for feature in features:

        lower = feature.lower()

        if (
            "future" in lower
            or "target" in lower
            or "t_plus" in lower
            or "t+" in lower
        ):

            future_features.append(
                feature
            )

    print(
        "Future-named features:",
        len(future_features),
    )

    if future_features:

        print(
            future_features
        )

        raise RuntimeError(
            "Future-derived features detected."
        )

    # ------------------------------------------------------------------------
    # DUPLICATE DATAFRAME COLUMNS
    # ------------------------------------------------------------------------

    duplicate_dataframe_columns = (
        data.columns[
            data.columns.duplicated()
        ]
        .tolist()
    )

    print(
        "Duplicate dataframe columns:",
        len(duplicate_dataframe_columns),
    )

    if duplicate_dataframe_columns:

        print(
            duplicate_dataframe_columns
        )

        raise RuntimeError(
            "Input dataframe contains duplicate columns."
        )

    # ------------------------------------------------------------------------
    # EXPLICIT FEATURE UNIQUENESS CHECK
    # ------------------------------------------------------------------------

    if len(features) != len(set(features)):

        raise RuntimeError(
            "Feature list still contains duplicates."
        )

    print()
    print(
        "FINAL UNIQUE TEMPORAL FEATURES:",
        len(features),
    )

    for i, feature in enumerate(features, 1):

        print(
            f"{i:02d}. {feature}"
        )

    return features


# ============================================================================
# COMMON SUBSET
# ============================================================================

def prepare_common_subset(
    data,
    features,
):

    print()
    print("=" * 78)
    print("BUILDING COMMON COMPLETE SUBSET")
    print("=" * 78)

    required = unique_preserve_order(
        [
            "sequence",
            "current_frame",
            "component",
            "ml_split",
            "physical_target",
        ]
        + features
    )

    subset = data[
        required
    ].copy()

    subset = subset.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    before = len(
        subset
    )

    subset = subset.dropna(
        subset=features + [
            "physical_target",
        ]
    ).copy()

    after = len(
        subset
    )

    print(
        "Records before:",
        before,
    )

    print(
        "Records after:",
        after,
    )

    print(
        "Removed:",
        before - after,
    )

    for split in [
        "TRAIN",
        "VAL",
        "TEST",
    ]:

        count = int(
            (
                subset["ml_split"]
                ==
                split
            ).sum()
        )

        print(
            split,
            count,
        )

    # ------------------------------------------------------------------------
    # CHECK DUPLICATE KEYS
    # ------------------------------------------------------------------------

    duplicate_keys = (
        subset[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate sequence/frame keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate temporal transition keys found."
        )

    # ------------------------------------------------------------------------
    # CHECK COMPONENT INTEGRITY
    # ------------------------------------------------------------------------

    component_split_counts = (
        subset
        .groupby("component")["ml_split"]
        .nunique()
    )

    component_conflicts = (
        component_split_counts[
            component_split_counts > 1
        ]
    )

    print(
        "Component split conflicts:",
        len(component_conflicts),
    )

    if len(component_conflicts) > 0:

        raise RuntimeError(
            "A component crosses ML splits."
        )

    return subset


# ============================================================================
# MODEL
# ============================================================================

def train_model(
    train,
    test,
    features,
):

    # IMPORTANT:
    # Convert explicitly to NumPy.
    #
    # This prevents sklearn/narwhals from inspecting duplicate
    # pandas feature names and makes the model input unambiguous.

    X_train = (
        train[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_train = (
        train[
            "physical_target"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_test = (
        test[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_test = (
        test[
            "physical_target"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    model = GradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        prediction,
    )

    return (
        model,
        prediction,
        metrics,
    )


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================

def calculate_permutation_importance(
    model,
    test,
    features,
):

    print()
    print("=" * 78)
    print("TEST PERMUTATION IMPORTANCE")
    print("=" * 78)

    X_test = (
        test[
            features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    y_test = (
        test[
            "physical_target"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    mapping = {}

    taxonomy = build_taxonomy()

    for group, columns in taxonomy.items():

        for column in columns:

            mapping[column] = group

    importance = pd.DataFrame(
        {
            "feature": features,

            "permutation_importance":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    importance["group"] = (
        importance["feature"]
        .map(mapping)
    )

    importance = importance.sort_values(
        "permutation_importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    print(
        importance.to_string(
            index=False
        )
    )

    return importance


# ============================================================================
# GROUP IMPORTANCE
# ============================================================================

def calculate_group_importance(
    importance
):

    print()
    print("=" * 78)
    print("FEATURE GROUP IMPORTANCE")
    print("=" * 78)

    result = (
        importance
        .groupby("group")
        .agg(
            feature_count=(
                "feature",
                "count",
            ),

            total_importance=(
                "permutation_importance",
                "sum",
            ),

            mean_importance=(
                "permutation_importance",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "total_importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# ABLATIONS
# ============================================================================

def build_ablation_sets(
    taxonomy,
):

    current = taxonomy["CURRENT"]
    lag = taxonomy["LAG"]
    change = taxonomy["CHANGE"]
    acceleration = taxonomy["ACCELERATION"]
    rolling = taxonomy["ROLLING"]
    trend = taxonomy["TREND"]

    full = unique_preserve_order(
        current
        + lag
        + change
        + acceleration
        + rolling
        + trend
    )

    return {

        "FULL_TEMPORAL":
            full,

        "CURRENT_ONLY":
            unique_preserve_order(
                current
            ),

        "CURRENT_PLUS_LAG":
            unique_preserve_order(
                current + lag
            ),

        "CURRENT_PLUS_CHANGE":
            unique_preserve_order(
                current + change
            ),

        "CURRENT_LAG_CHANGE":
            unique_preserve_order(
                current + lag + change
            ),

        "CURRENT_LAG_CHANGE_ACCELERATION":
            unique_preserve_order(
                current
                + lag
                + change
                + acceleration
            ),

        "LAG_CHANGE_ROLLING":
            unique_preserve_order(
                current
                + lag
                + change
                + rolling
            ),

        "LAG_CHANGE_TREND":
            unique_preserve_order(
                current
                + lag
                + change
                + trend
            ),

        "WITHOUT_ACCELERATION":
            [
                feature
                for feature in full
                if feature not in acceleration
            ],

        "WITHOUT_ROLLING":
            [
                feature
                for feature in full
                if feature not in rolling
            ],

        "WITHOUT_TREND":
            [
                feature
                for feature in full
                if feature not in trend
            ],
    }


# ============================================================================
# ABLATION RUN
# ============================================================================

def run_ablation(
    common,
    feature_sets,
):

    print()
    print("=" * 78)
    print("CONTROLLED TEMPORAL ABLATION")
    print("=" * 78)

    train = common[
        common["ml_split"] == "TRAIN"
    ].copy()

    val = common[
        common["ml_split"] == "VAL"
    ].copy()

    test = common[
        common["ml_split"] == "TEST"
    ].copy()

    results = []

    for name, features in feature_sets.items():

        features = unique_preserve_order(
            features
        )

        print()
        print(
            "ABLATION:",
            name,
        )

        print(
            "Feature count:",
            len(features),
        )

        # ---------------------------------------------------------------
        # Explicit duplicate check
        # ---------------------------------------------------------------

        if len(features) != len(set(features)):

            raise RuntimeError(
                "Duplicate features detected in ablation: "
                + name
            )

        # ---------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------

        X_train = (
            train[
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_train = (
            train[
                "physical_target"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        X_val = (
            val[
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_val = (
            val[
                "physical_target"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        X_test = (
            test[
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_test = (
            test[
                "physical_target"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        model = GradientBoostingRegressor(
            **MODEL_PARAMS
        )

        model.fit(
            X_train,
            y_train,
        )

        train_prediction = model.predict(
            X_train
        )

        val_prediction = model.predict(
            X_val
        )

        test_prediction = model.predict(
            X_test
        )

        train_metrics = calculate_metrics(
            y_train,
            train_prediction,
        )

        val_metrics = calculate_metrics(
            y_val,
            val_prediction,
        )

        test_metrics = calculate_metrics(
            y_test,
            test_prediction,
        )

        print(
            "TRAIN MAE:",
            f"{train_metrics['MAE']:.8f}",
        )

        print(
            "VAL MAE:",
            f"{val_metrics['MAE']:.8f}",
        )

        print(
            "TEST MAE:",
            f"{test_metrics['MAE']:.8f}",
        )

        print(
            "TEST R2:",
            f"{test_metrics['R2']:.8f}",
        )

        results.append(
            {
                "ablation":
                    name,

                "feature_count":
                    len(features),

                "train_records":
                    len(train),

                "val_records":
                    len(val),

                "test_records":
                    len(test),

                "train_MAE":
                    train_metrics["MAE"],

                "train_RMSE":
                    train_metrics["RMSE"],

                "train_R2":
                    train_metrics["R2"],

                "val_MAE":
                    val_metrics["MAE"],

                "val_RMSE":
                    val_metrics["RMSE"],

                "val_R2":
                    val_metrics["R2"],

                "test_MAE":
                    test_metrics["MAE"],

                "test_RMSE":
                    test_metrics["RMSE"],

                "test_R2":
                    test_metrics["R2"],
            }
        )

    result = pd.DataFrame(
        results
    )

    result = result.sort_values(
        "test_MAE"
    ).reset_index(
        drop=True
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.18 - TEMPORAL PHYSICAL FEATURE IMPORTANCE"
    )
    print("=" * 78)

    print()
    print("Project:")
    print(PROJECT_ROOT)

    # ------------------------------------------------------------------------
    # FILE CHECK
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("CHECKING INPUT FILES")
    print("=" * 78)

    input_files = [
        (
            "temporal_features",
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
    ]

    for name, path in input_files:

        status = (
            "FOUND"
            if path.exists()
            else "NOT FOUND"
        )

        print(
            f"{name:<24}: {status}"
        )

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    data = load_data()

    # ------------------------------------------------------------------------
    # TAXONOMY
    # ------------------------------------------------------------------------

    taxonomy = build_taxonomy()

    features = validate_features(
        data,
        taxonomy,
    )

    # ------------------------------------------------------------------------
    # COMMON SUBSET
    # ------------------------------------------------------------------------

    common = prepare_common_subset(
        data,
        features,
    )

    train = common[
        common["ml_split"] == "TRAIN"
    ].copy()

    val = common[
        common["ml_split"] == "VAL"
    ].copy()

    test = common[
        common["ml_split"] == "TEST"
    ].copy()

    # ------------------------------------------------------------------------
    # FULL MODEL
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FULL TEMPORAL MODEL")
    print("=" * 78)

    model, prediction, full_metrics = train_model(
        train,
        test,
        features,
    )

    print(
        "TEST MAE:",
        f"{full_metrics['MAE']:.8f}",
    )

    print(
        "TEST RMSE:",
        f"{full_metrics['RMSE']:.8f}",
    )

    print(
        "TEST R2:",
        f"{full_metrics['R2']:.8f}",
    )

    # ------------------------------------------------------------------------
    # PERMUTATION IMPORTANCE
    # ------------------------------------------------------------------------

    importance = (
        calculate_permutation_importance(
            model,
            test,
            features,
        )
    )

    importance.to_csv(
        IMPORTANCE_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # GROUP IMPORTANCE
    # ------------------------------------------------------------------------

    group_importance = (
        calculate_group_importance(
            importance
        )
    )

    group_importance.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # ABLATIONS
    # ------------------------------------------------------------------------

    feature_sets = build_ablation_sets(
        taxonomy
    )

    ablation = run_ablation(
        common,
        feature_sets,
    )

    ablation.to_csv(
        ABLATION_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------------

    persistence_prediction = (
        test[
            "crack_area_ratio"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    persistence_metrics = calculate_metrics(
        test[
            "physical_target"
        ].to_numpy(
            dtype=np.float64
        ),
        persistence_prediction,
    )

    print()
    print("=" * 78)
    print("PERSISTENCE BASELINE")
    print("=" * 78)

    print(
        "TEST MAE:",
        f"{persistence_metrics['MAE']:.8f}",
    )

    print(
        "TEST RMSE:",
        f"{persistence_metrics['RMSE']:.8f}",
    )

    print(
        "TEST R2:",
        f"{persistence_metrics['R2']:.8f}",
    )

    # ------------------------------------------------------------------------
    # FULL MODEL ROW
    # ------------------------------------------------------------------------

    full_row = ablation[
        ablation["ablation"]
        ==
        "FULL_TEMPORAL"
    ]

    if len(full_row) != 1:

        raise RuntimeError(
            "FULL_TEMPORAL row missing from ablation results."
        )

    full_mae = float(
        full_row.iloc[0]["test_MAE"]
    )

    persistence_mae = float(
        persistence_metrics["MAE"]
    )

    improvement = (
        (
            persistence_mae
            -
            full_mae
        )
        /
        persistence_mae
    ) * 100.0

    # ------------------------------------------------------------------------
    # BEST VALIDATION ABLATION
    # ------------------------------------------------------------------------

    best_validation_row = (
        ablation
        .sort_values(
            "val_MAE"
        )
        .iloc[0]
    )

    # ------------------------------------------------------------------------
    # BEST TEST ABLATION
    # ------------------------------------------------------------------------

    best_test_row = (
        ablation
        .sort_values(
            "test_MAE"
        )
        .iloc[0]
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    summary = {

        "stage":
            "3.18",

        "title":
            "Temporal Physical Feature Importance and Ablation",

        "target":
            "crack_area_ratio(t+1)",

        "feature_count":
            int(len(features)),

        "features":
            features,

        "feature_taxonomy":
            taxonomy,

        "common_subset":
            {
                "train":
                    int(len(train)),

                "val":
                    int(len(val)),

                "test":
                    int(len(test)),
            },

        "full_temporal_metrics":
            full_metrics,

        "persistence_metrics":
            persistence_metrics,

        "improvement_vs_persistence_percent":
            float(improvement),

        "best_validation_ablation":
            best_validation_row.to_dict(),

        "best_test_ablation":
            best_test_row.to_dict(),

        "top_features":
            importance.head(
                15
            ).to_dict(
                orient="records"
            ),

        "group_importance":
            group_importance.to_dict(
                orient="records"
            ),

        "duplicate_feature_protection":
            True,

        "future_leakage_features":
            0,

        "decision":
            "TEMPORAL_FEATURE_IMPORTANCE_COMPLETE",

        "next_stage":
            "SPATIAL_PLUS_TEMPORAL_PHYSICAL_REGRESSION",
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_json(summary),
            file,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )

    # ------------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 3.18 COMPLETE")
    print("=" * 78)

    print(
        "Unique temporal features:",
        len(features),
    )

    print(
        "Common TRAIN:",
        len(train),
    )

    print(
        "Common VAL:",
        len(val),
    )

    print(
        "Common TEST:",
        len(test),
    )

    print(
        "Full temporal TEST MAE:",
        f"{full_mae:.8f}",
    )

    print(
        "Persistence TEST MAE:",
        f"{persistence_mae:.8f}",
    )

    print(
        "Full temporal improvement:",
        f"{improvement:.4f}%",
    )

    print()
    print(
        "Best validation ablation:",
        best_validation_row["ablation"],
    )

    print(
        "Best validation MAE:",
        f"{best_validation_row['val_MAE']:.8f}",
    )

    print()
    print(
        "Best test ablation:",
        best_test_row["ablation"],
    )

    print(
        "Best test MAE:",
        f"{best_test_row['test_MAE']:.8f}",
    )

    print()
    print(
        "Outputs:"
    )

    print(
        IMPORTANCE_OUTPUT
    )

    print(
        ABLATION_OUTPUT
    )

    print(
        GROUP_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print()
    print(
        "FINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()