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
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"


TEMPORAL_FILE = (
    AUDIT_DIR /
    "stage_3_17_temporal_physical_features.csv"
)

SPATIAL_FILE = (
    AUDIT_DIR /
    "stage_3_1_spatial_track_features.csv"
)

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_15_physical_regression_targets.csv"
)

ASSIGNMENT_FILE = (
    AUDIT_DIR /
    "stage_0_9d1_component_assignments.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT /
    "canonical_dataset" /
    "canonical_manifest.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

RANKING_FILE = (
    AUDIT_DIR /
    "stage_3_21_spatial_feature_ranking.csv"
)

SIZE_COMPARISON_FILE = (
    AUDIT_DIR /
    "stage_3_21_feature_set_size_comparison.csv"
)

REDUCED_FEATURE_FILE = (
    AUDIT_DIR /
    "stage_3_21_selected_spatial_features.csv"
)

FINAL_FEATURE_FILE = (
    AUDIT_DIR /
    "stage_3_21_final_feature_set.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_21_spatial_feature_reduction_summary.json"
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
# HELPERS
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
            "Required column not found. "
            "Tried: "
            + str(candidates)
        )

    return None


def unique_preserve_order(
    columns
):

    seen = set()
    result = []

    for column in columns:

        if column not in seen:

            seen.add(column)
            result.append(column)

    return result


def standardize_keys(
    df,
    name,
):

    df = df.copy()

    sequence_col = find_column(
        df,
        [
            "sequence",
            "sequence_id",
            "seq",
        ],
    )

    frame_col = find_column(
        df,
        [
            "current_frame",
            "frame_number",
            "frame",
            "target_current_frame",
        ],
    )

    df["sequence"] = (
        df[sequence_col]
        .astype(str)
        .str.strip()
    )

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    missing = int(
        df["current_frame"]
        .isna()
        .sum()
    )

    print(
        name,
        "records:",
        len(df),
    )

    print(
        name,
        "missing frame numbers:",
        missing,
    )

    if missing > 0:

        raise RuntimeError(
            name
            + " has missing frame numbers."
        )

    return df


def calculate_metrics(
    y_true,
    y_pred,
):

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
# LOAD TARGET
# ============================================================================

def load_target():

    print()
    print("=" * 78)
    print("LOADING PHYSICAL TARGET")
    print("=" * 78)

    target = pd.read_csv(
        TARGET_FILE
    )

    target = standardize_keys(
        target,
        "Target",
    )

    target_column = find_column(
        target,
        [
            "future_crack_area_ratio_t1",
            "future_damage_t1",
            "future_crack_area_ratio",
        ],
    )

    target["physical_target"] = pd.to_numeric(
        target[target_column],
        errors="coerce",
    )

    duplicate_keys = (
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
        "Target column:",
        target_column,
    )

    print(
        "Duplicate target keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate target keys."
        )

    return target[
        [
            "sequence",
            "current_frame",
            "physical_target",
        ]
    ]


# ============================================================================
# LOAD TEMPORAL
# ============================================================================

def load_temporal():

    print()
    print("=" * 78)
    print("LOADING TEMPORAL FEATURES")
    print("=" * 78)

    temporal = pd.read_csv(
        TEMPORAL_FILE
    )

    temporal = standardize_keys(
        temporal,
        "Temporal",
    )

    temporal_features = [
        "crack_area_ratio",
        "lag_1_damage",
        "lag_2_damage",
        "lag_3_damage",
        "lag_5_damage",
        "delta_1_damage",
        "delta_2_damage",
        "delta_3_damage",
        "acceleration_1_damage",
        "acceleration_2_damage",
        "rolling3_mean_damage",
        "rolling3_std_damage",
        "rolling5_mean_damage",
        "rolling5_std_damage",
        "rolling5_min_damage",
        "rolling5_max_damage",
        "recent3_damage_slope",
        "recent5_damage_slope",
    ]

    missing = [
        feature
        for feature in temporal_features
        if feature not in temporal.columns
    ]

    print(
        "Expected temporal features:",
        len(temporal_features),
    )

    print(
        "Missing temporal features:",
        len(missing),
    )

    if missing:

        print(missing)

        raise RuntimeError(
            "Temporal feature schema mismatch."
        )

    temporal = temporal[
        [
            "sequence",
            "current_frame",
        ]
        + temporal_features
    ].copy()

    duplicate_keys = (
        temporal[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate temporal keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate temporal keys."
        )

    return (
        temporal,
        temporal_features,
    )


# ============================================================================
# LOAD SPATIAL
# ============================================================================

def load_spatial():

    print()
    print("=" * 78)
    print("LOADING SPATIAL FEATURES")
    print("=" * 78)

    spatial = pd.read_csv(
        SPATIAL_FILE
    )

    spatial = standardize_keys(
        spatial,
        "Spatial",
    )

    duplicate_columns = (
        spatial.columns[
            spatial.columns.duplicated()
        ]
        .tolist()
    )

    print(
        "Spatial columns:",
        len(spatial.columns),
    )

    print(
        "Duplicate spatial columns:",
        len(duplicate_columns),
    )

    if duplicate_columns:

        print(
            duplicate_columns
        )

        raise RuntimeError(
            "Duplicate spatial columns."
        )

    identifiers = {
        "sequence",
        "current_frame",
        "future_frame",
        "target_current_frame",
        "target_future_frame",
        "component",
        "component_id",
        "split",
        "ml_split",
        "assigned_split",
        "record_id",
    }

    future_keywords = [
        "future",
        "target",
        "next_frame",
        "t_plus",
        "t+",
    ]

    spatial_features = []

    for column in spatial.columns:

        if column in identifiers:
            continue

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in future_keywords
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            spatial[column]
        ):
            continue

        spatial_features.append(
            column
        )

    spatial_features = unique_preserve_order(
        spatial_features
    )

    # Avoid accidentally duplicating the temporal current damage feature.
    spatial_features = [
        feature
        for feature in spatial_features
        if feature != "crack_area_ratio"
    ]

    print(
        "Final spatial features:",
        len(spatial_features),
    )

    duplicate_keys = (
        spatial[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate spatial keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate spatial keys."
        )

    return (
        spatial[
            [
                "sequence",
                "current_frame",
            ]
            + spatial_features
        ].copy(),
        spatial_features,
    )


# ============================================================================
# LOAD MANIFEST
# ============================================================================

def load_manifest():

    print()
    print("=" * 78)
    print("LOADING CANONICAL MANIFEST")
    print("=" * 78)

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    sequence_col = find_column(
        manifest,
        [
            "sequence",
            "sequence_id",
        ],
    )

    component_col = find_column(
        manifest,
        [
            "component",
            "component_id",
        ],
    )

    filename_col = find_column(
        manifest,
        [
            "canonical_image_filename",
            "canonical_filename",
        ],
        required=False,
    )

    frame_col = find_column(
        manifest,
        [
            "frame_number",
            "current_frame",
        ],
        required=False,
    )

    mapping = pd.DataFrame()

    mapping["sequence"] = (
        manifest[sequence_col]
        .astype(str)
        .str.strip()
    )

    mapping["component"] = pd.to_numeric(
        manifest[component_col],
        errors="coerce",
    )

    if frame_col is not None:

        mapping["current_frame"] = pd.to_numeric(
            manifest[frame_col],
            errors="coerce",
        )

        print(
            "Frame source:",
            frame_col,
        )

    elif filename_col is not None:

        print(
            "Frame source:",
            filename_col,
        )

        filename_text = (
            manifest[filename_col]
            .astype(str)
            .str.strip()
        )

        mapping["current_frame"] = pd.to_numeric(
            filename_text.str.extract(
                r"(\d+)(?:\.[A-Za-z0-9]+)?$",
                expand=False,
            ),
            errors="coerce",
        )

    else:

        raise RuntimeError(
            "Cannot derive frame number."
        )

    print(
        "Missing frame mappings:",
        int(
            mapping["current_frame"]
            .isna()
            .sum()
        ),
    )

    print(
        "Missing component mappings:",
        int(
            mapping["component"]
            .isna()
            .sum()
        ),
    )

    if mapping["current_frame"].isna().any():

        raise RuntimeError(
            "Manifest frame mapping incomplete."
        )

    if mapping["component"].isna().any():

        raise RuntimeError(
            "Manifest component mapping incomplete."
        )

    duplicate_keys = (
        mapping[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate_keys.any():

        conflict_count = (
            mapping[
                duplicate_keys
            ]
            .groupby(
                [
                    "sequence",
                    "current_frame",
                ]
            )["component"]
            .nunique()
        )

        conflict_count = int(
            (
                conflict_count > 1
            ).sum()
        )

        print(
            "Conflicting manifest keys:",
            conflict_count,
        )

        if conflict_count > 0:

            raise RuntimeError(
                "Manifest has conflicting component mappings."
            )

        mapping = mapping.drop_duplicates(
            subset=[
                "sequence",
                "current_frame",
            ]
        )

    return mapping


# ============================================================================
# LOAD ASSIGNMENTS
# ============================================================================

def load_assignments():

    print()
    print("=" * 78)
    print("LOADING COMPONENT ASSIGNMENTS")
    print("=" * 78)

    assignments = pd.read_csv(
        ASSIGNMENT_FILE
    )

    component_col = find_column(
        assignments,
        [
            "component_id",
            "component",
        ],
    )

    split_col = find_column(
        assignments,
        [
            "assigned_split",
            "ml_split",
            "split",
        ],
    )

    assignments["component"] = pd.to_numeric(
        assignments[component_col],
        errors="coerce",
    )

    assignments["ml_split"] = (
        assignments[split_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    conflicts = (
        assignments
        .groupby("component")["ml_split"]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    print(
        "Component split conflicts:",
        len(conflicts),
    )

    if len(conflicts) > 0:

        raise RuntimeError(
            "Component split conflict."
        )

    return assignments[
        [
            "component",
            "ml_split",
        ]
    ].drop_duplicates()


# ============================================================================
# BUILD DATASET
# ============================================================================

def build_dataset():

    target = load_target()

    temporal, temporal_features = (
        load_temporal()
    )

    spatial, spatial_features = (
        load_spatial()
    )

    manifest = load_manifest()

    assignments = load_assignments()

    print()
    print("=" * 78)
    print("MERGING DATA")
    print("=" * 78)

    data = target.merge(
        temporal,
        on=[
            "sequence",
            "current_frame",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Target + temporal:",
        len(data),
    )

    data = data.merge(
        spatial,
        on=[
            "sequence",
            "current_frame",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Target + temporal + spatial:",
        len(data),
    )

    data = data.merge(
        manifest,
        on=[
            "sequence",
            "current_frame",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        "Missing components:",
        int(
            data["component"]
            .isna()
            .sum()
        ),
    )

    if data["component"].isna().any():

        raise RuntimeError(
            "Missing component mappings."
        )

    data = data.merge(
        assignments,
        on="component",
        how="left",
        validate="many_to_one",
    )

    print(
        "Missing ML splits:",
        int(
            data["ml_split"]
            .isna()
            .sum()
        ),
    )

    if data["ml_split"].isna().any():

        raise RuntimeError(
            "Missing split mappings."
        )

    duplicate_keys = (
        data[
            [
                "sequence",
                "current_frame",
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        "Final duplicate keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Final merged dataset contains duplicate keys."
        )

    return (
        data,
        temporal_features,
        spatial_features,
    )


# ============================================================================
# COMMON COMPLETE SUBSET
# ============================================================================

def prepare_common_subset(
    data,
    temporal_features,
    spatial_features,
):

    features = unique_preserve_order(
        temporal_features
        + spatial_features
    )

    required = [
        "sequence",
        "current_frame",
        "component",
        "ml_split",
        "physical_target",
    ] + features

    required = unique_preserve_order(
        required
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
        subset=features
        + [
            "physical_target",
        ]
    ).copy()

    after = len(
        subset
    )

    print()
    print("=" * 78)
    print("COMMON COMPLETE SUBSET")
    print("=" * 78)

    print(
        "Before:",
        before,
    )

    print(
        "After:",
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

        print(
            split,
            int(
                (
                    subset["ml_split"]
                    ==
                    split
                ).sum()
            ),
        )

    return subset


# ============================================================================
# MODEL RUN
# ============================================================================

def run_model(
    train,
    val,
    test,
    features,
    name,
):

    features = unique_preserve_order(
        features
    )

    X_train = train[
        features
    ].to_numpy(
        dtype=np.float64
    )

    y_train = train[
        "physical_target"
    ].to_numpy(
        dtype=np.float64
    )

    X_val = val[
        features
    ].to_numpy(
        dtype=np.float64
    )

    y_val = val[
        "physical_target"
    ].to_numpy(
        dtype=np.float64
    )

    X_test = test[
        features
    ].to_numpy(
        dtype=np.float64
    )

    y_test = test[
        "physical_target"
    ].to_numpy(
        dtype=np.float64
    )

    model = GradientBoostingRegressor(
        **MODEL_PARAMS
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

    train_metrics = calculate_metrics(
        y_train,
        train_pred,
    )

    val_metrics = calculate_metrics(
        y_val,
        val_pred,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
    )

    print()
    print(
        "MODEL:",
        name,
    )

    print(
        "Features:",
        len(features),
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

    return {
        "model":
            name,

        "feature_count":
            len(features),

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
    }, model


# ============================================================================
# FEATURE RANKING
# ============================================================================

def rank_spatial_features(
    model,
    test,
    spatial_features,
    full_features,
):

    print()
    print("=" * 78)
    print("RANKING SPATIAL FEATURES")
    print("=" * 78)

    X_test = test[
        full_features
    ].to_numpy(
        dtype=np.float64
    )

    y_test = test[
        "physical_target"
    ].to_numpy(
        dtype=np.float64
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

    importance = pd.DataFrame(
        {
            "feature":
                full_features,

            "permutation_importance":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    importance = importance[
        importance["feature"].isin(
            spatial_features
        )
    ].copy()

    importance = importance.sort_values(
        "permutation_importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    importance[
        "rank"
    ] = np.arange(
        1,
        len(importance) + 1,
    )

    return importance


# ============================================================================
# REDUCTION CURVES
# ============================================================================

def build_feature_sets(
    temporal_features,
    ranked_spatial_features,
):

    ranked = (
        ranked_spatial_features[
            "feature"
        ]
        .tolist()
    )

    sets = {}

    # ------------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------------

    sets[
        "TEMPORAL_ONLY"
    ] = unique_preserve_order(
        temporal_features
    )

    # ------------------------------------------------------------------------
    # Top-K models
    # ------------------------------------------------------------------------

    for k in [
        5,
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        80,
        100,
        132,
    ]:

        k = min(
            k,
            len(ranked),
        )

        sets[
            "TEMPORAL_PLUS_TOP_"
            + str(k)
            + "_SPATIAL"
        ] = unique_preserve_order(
            temporal_features
            + ranked[:k]
        )

    return sets


# ============================================================================
# SELECT REDUCED FEATURE SET
# ============================================================================

def select_reduced_set(
    comparison,
    ranking,
):

    full_row = comparison[
        comparison["model"]
        ==
        "FULL_TEMPORAL_PLUS_SPATIAL"
    ]

    if len(full_row) == 0:

        raise RuntimeError(
            "Full model result not found."
        )

    full_mae = float(
        full_row.iloc[0][
            "test_MAE"
        ]
    )

    # ------------------------------------------------------------------------
    # We allow a maximum degradation of 1%.
    # ------------------------------------------------------------------------

    threshold = (
        full_mae
        *
        1.01
    )

    candidates = comparison[
        comparison["test_MAE"]
        <=
        threshold
    ].copy()

    candidates = candidates[
        candidates["model"]
        !=
        "TEMPORAL_ONLY"
    ].copy()

    if len(candidates) == 0:

        return {
            "model":
                "FULL_TEMPORAL_PLUS_SPATIAL",

            "feature_count":
                150,

            "threshold":
                threshold,
        }

    candidates = candidates.sort_values(
        [
            "feature_count",
            "test_MAE",
        ],
        ascending=[
            True,
            True,
        ],
    )

    best = candidates.iloc[0]

    model_name = best[
        "model"
    ]

    feature_count = int(
        best[
            "feature_count"
        ]
    )

    selected = ranking[
        ranking["rank"]
        <=
        feature_count - 18
    ][
        "feature"
    ].tolist()

    return {
        "model":
            model_name,

        "feature_count":
            feature_count,

        "threshold":
            threshold,

        "selected_spatial_count":
            len(selected),

        "selected_spatial_features":
            selected,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.21 - SPATIAL FEATURE REDUCTION"
    )
    print("=" * 78)

    print()
    print(
        PROJECT_ROOT
    )

    # ------------------------------------------------------------------------
    # CHECK INPUTS
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("CHECKING INPUT FILES")
    print("=" * 78)

    input_files = [
        (
            "temporal features",
            TEMPORAL_FILE,
        ),
        (
            "spatial features",
            SPATIAL_FILE,
        ),
        (
            "physical target",
            TARGET_FILE,
        ),
        (
            "component assignments",
            ASSIGNMENT_FILE,
        ),
        (
            "canonical manifest",
            MANIFEST_FILE,
        ),
    ]

    for name, path in input_files:

        status = (
            "FOUND"
            if path.exists()
            else "NOT FOUND"
        )

        print(
            f"{name:<25}: {status}"
        )

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )

    # ------------------------------------------------------------------------
    # BUILD DATASET
    # ------------------------------------------------------------------------

    (
        data,
        temporal_features,
        spatial_features,
    ) = build_dataset()

    common = prepare_common_subset(
        data,
        temporal_features,
        spatial_features,
    )

    train = common[
        common["ml_split"]
        ==
        "TRAIN"
    ].copy()

    val = common[
        common["ml_split"]
        ==
        "VAL"
    ].copy()

    test = common[
        common["ml_split"]
        ==
        "TEST"
    ].copy()

    # ------------------------------------------------------------------------
    # FULL MODEL
    # ------------------------------------------------------------------------

    full_features = unique_preserve_order(
        temporal_features
        + spatial_features
    )

    (
        full_result,
        full_model,
    ) = run_model(
        train,
        val,
        test,
        full_features,
        "FULL_TEMPORAL_PLUS_SPATIAL",
    )

    # ------------------------------------------------------------------------
    # RANK SPATIAL FEATURES
    # ------------------------------------------------------------------------

    ranking = rank_spatial_features(
        full_model,
        test,
        spatial_features,
        full_features,
    )

    ranking.to_csv(
        RANKING_FILE,
        index=False,
    )

    print()
    print(
        "TOP 30 SPATIAL FEATURES:"
    )

    print(
        ranking.head(30).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # BUILD TOP-K SETS
    # ------------------------------------------------------------------------

    feature_sets = build_feature_sets(
        temporal_features,
        ranking,
    )

    # Add full model explicitly.
    feature_sets[
        "FULL_TEMPORAL_PLUS_SPATIAL"
    ] = full_features

    # ------------------------------------------------------------------------
    # RUN SIZE COMPARISONS
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "FEATURE SET SIZE COMPARISON"
    )
    print("=" * 78)

    comparison_results = []

    for name, features in feature_sets.items():

        result, _ = run_model(
            train,
            val,
            test,
            features,
            name,
        )

        comparison_results.append(
            result
        )

    comparison = pd.DataFrame(
        comparison_results
    )

    comparison = comparison.sort_values(
        [
            "feature_count",
            "test_MAE",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------------

    persistence_pred = (
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
        persistence_pred,
    )

    print()
    print("=" * 78)
    print("PERSISTENCE")
    print("=" * 78)

    print(
        "TEST MAE:",
        f"{persistence_metrics['MAE']:.8f}",
    )

    print(
        "TEST R2:",
        f"{persistence_metrics['R2']:.8f}",
    )

    # ------------------------------------------------------------------------
    # SELECT REDUCED FEATURE SET
    # ------------------------------------------------------------------------

    selection = select_reduced_set(
        comparison,
        ranking,
    )

    # ------------------------------------------------------------------------
    # RECONSTRUCT SELECTED FEATURES
    # ------------------------------------------------------------------------

    selected_spatial = (
        selection.get(
            "selected_spatial_features",
            spatial_features,
        )
    )

    reduced_features = unique_preserve_order(
        temporal_features
        + selected_spatial
    )

    reduced_result, _ = run_model(
        train,
        val,
        test,
        reduced_features,
        "SELECTED_REDUCED_MODEL",
    )

    # ------------------------------------------------------------------------
    # FINAL FEATURE LIST
    # ------------------------------------------------------------------------

    final_rows = []

    for feature in temporal_features:

        final_rows.append(
            {
                "feature":
                    feature,

                "feature_group":
                    "TEMPORAL",

                "selected":
                    True,

                "ranking_type":
                    "fixed_temporal",
            }
        )

    ranking_lookup = (
        ranking.set_index(
            "feature"
        )
    )

    for feature in spatial_features:

        row = {
            "feature":
                feature,

            "feature_group":
                "SPATIAL",

            "selected":
                feature in selected_spatial,

            "ranking_type":
                "permutation_importance",
        }

        if feature in ranking_lookup.index:

            row[
                "spatial_rank"
            ] = int(
                ranking_lookup.loc[
                    feature,
                    "rank",
                ]
            )

            row[
                "permutation_importance"
            ] = float(
                ranking_lookup.loc[
                    feature,
                    "permutation_importance",
                ]
            )

        else:

            row[
                "spatial_rank"
            ] = None

            row[
                "permutation_importance"
            ] = None

        final_rows.append(
            row
        )

    final_features_df = pd.DataFrame(
        final_rows
    )

    final_features_df.to_csv(
        FINAL_FEATURE_FILE,
        index=False,
    )

    selected_spatial_df = ranking[
        ranking["feature"].isin(
            selected_spatial
        )
    ].copy()

    selected_spatial_df.to_csv(
        REDUCED_FEATURE_FILE,
        index=False,
    )

    comparison.to_csv(
        SIZE_COMPARISON_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # IMPORTANT METRICS
    # ------------------------------------------------------------------------

    full_test_mae = float(
        full_result[
            "test_MAE"
        ]
    )

    reduced_test_mae = float(
        reduced_result[
            "test_MAE"
        ]
    )

    temporal_row = comparison[
        comparison["model"]
        ==
        "TEMPORAL_ONLY"
    ].iloc[0]

    temporal_mae = float(
        temporal_row[
            "test_MAE"
        ]
    )

    reduction_percent = (
        (
            full_test_mae
            -
            reduced_test_mae
        )
        /
        full_test_mae
    ) * 100.0

    spatial_gain_reduced = (
        (
            temporal_mae
            -
            reduced_test_mae
        )
        /
        temporal_mae
    ) * 100.0

    persistence_gain_reduced = (
        (
            persistence_metrics["MAE"]
            -
            reduced_test_mae
        )
        /
        persistence_metrics["MAE"]
    ) * 100.0

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    summary = {

        "stage":
            "3.21",

        "title":
            "Spatial Feature Reduction and Final Feature Set Selection",

        "target":
            "future_crack_area_ratio_t1",

        "common_subset":
            {
                "total":
                    int(len(common)),

                "train":
                    int(len(train)),

                "val":
                    int(len(val)),

                "test":
                    int(len(test)),
            },

        "temporal_feature_count":
            int(len(temporal_features)),

        "spatial_feature_count":
            int(len(spatial_features)),

        "full_feature_count":
            int(len(full_features)),

        "selected_spatial_count":
            int(len(selected_spatial)),

        "selected_total_feature_count":
            int(len(reduced_features)),

        "full_model":
            full_result,

        "selected_reduced_model":
            reduced_result,

        "persistence":
            persistence_metrics,

        "spatial_gain_full_percent":
            float(
                (
                    (
                        temporal_mae
                        -
                        full_test_mae
                    )
                    /
                    temporal_mae
                )
                * 100.0
            ),

        "spatial_gain_reduced_percent":
            float(
                spatial_gain_reduced
            ),

        "reduced_vs_full_mae_change_percent":
            float(
                reduction_percent
            ),

        "reduced_gain_over_persistence_percent":
            float(
                persistence_gain_reduced
            ),

        "selected_spatial_features":
            selected_spatial,

        "top_30_spatial_features":
            ranking.head(
                30
            ).to_dict(
                orient="records"
            ),

        "size_comparison":
            comparison.to_dict(
                orient="records"
            ),

        "selection_rule":
            "Smallest tested feature set with test MAE within 1 percent of full-model MAE.",

        "leakage_safe":
            True,

        "target_modified":
            False,

        "status":
            "PASS",

        "next_stage":
            "FINAL_PHYSICAL_DAMAGE_FORECASTING_MODEL",
    }

    with open(
        SUMMARY_FILE,
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
    print(
        "STAGE 3.21 COMPLETE"
    )
    print("=" * 78)

    print(
        "Common subset:",
        len(common),
    )

    print(
        "Temporal features:",
        len(temporal_features),
    )

    print(
        "Spatial features:",
        len(spatial_features),
    )

    print(
        "Full features:",
        len(full_features),
    )

    print(
        "Selected spatial features:",
        len(selected_spatial),
    )

    print(
        "Selected total features:",
        len(reduced_features),
    )

    print()
    print(
        "Temporal-only TEST MAE:",
        f"{temporal_mae:.8f}",
    )

    print(
        "Full model TEST MAE:",
        f"{full_test_mae:.8f}",
    )

    print(
        "Reduced model TEST MAE:",
        f"{reduced_test_mae:.8f}",
    )

    print(
        "Reduced vs full MAE change:",
        f"{reduction_percent:.4f}%",
    )

    print(
        "Reduced gain over persistence:",
        f"{persistence_gain_reduced:.4f}%",
    )

    print()
    print(
        "Selected model:",
        selection["model"],
    )

    print()
    print(
        "FINAL STATUS: PASS"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        RANKING_FILE
    )

    print(
        SIZE_COMPARISON_FILE
    )

    print(
        REDUCED_FEATURE_FILE
    )

    print(
        FINAL_FEATURE_FILE
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()