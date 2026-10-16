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
# INPUTS
# ============================================================================

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

IMPORTANCE_FILE = (
    AUDIT_DIR /
    "stage_3_20_spatial_feature_importance.csv"
)

GROUP_FILE = (
    AUDIT_DIR /
    "stage_3_20_spatial_feature_group_importance.csv"
)

ABLATION_FILE = (
    AUDIT_DIR /
    "stage_3_20_spatial_feature_ablation.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_20_spatial_feature_importance_summary.json"
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
            "\nRequired column not found.\n"
            "Tried:\n"
            + str(candidates)
            + "\n\nAvailable columns:\n"
            + str(list(df.columns))
        )

    return None


def unique_preserve_order(columns):

    seen = set()
    result = []

    for column in columns:

        if column not in seen:

            seen.add(column)
            result.append(column)

    return result


def metrics(
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
# FRAME KEY STANDARDIZATION
# ============================================================================

def standardize_frame_key(
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
            + " contains missing frame numbers."
        )

    return df


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

    target = standardize_frame_key(
        target,
        "Physical target",
    )

    target_column = find_column(
        target,
        [
            "future_crack_area_ratio_t1",
            "future_damage_t1",
            "future_crack_area_ratio",
            "target_crack_area_ratio",
        ],
    )

    target["physical_target"] = pd.to_numeric(
        target[target_column],
        errors="coerce",
    )

    print(
        "Target column:",
        target_column,
    )

    duplicates = (
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
        int(duplicates),
    )

    if duplicates > 0:

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

    temporal = standardize_frame_key(
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
        "Temporal features expected:",
        len(temporal_features),
    )

    print(
        "Temporal features missing:",
        len(missing),
    )

    if missing:

        print(missing)

        raise RuntimeError(
            "Temporal schema mismatch."
        )

    temporal = temporal[
        [
            "sequence",
            "current_frame",
        ]
        + temporal_features
    ].copy()

    duplicates = (
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
        int(duplicates),
    )

    if duplicates > 0:

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

    print(
        "Spatial records:",
        len(spatial),
    )

    print(
        "Spatial columns:",
        len(spatial.columns),
    )

    spatial = standardize_frame_key(
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
        "Duplicate spatial columns:",
        len(duplicate_columns),
    )

    if duplicate_columns:

        print(
            duplicate_columns
        )

        raise RuntimeError(
            "Spatial file has duplicate columns."
        )

    # ------------------------------------------------------------------------
    # IDENTIFIER / FUTURE COLUMNS
    # ------------------------------------------------------------------------

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

    # Never duplicate the current temporal damage feature.
    spatial_features = [
        feature
        for feature in spatial_features
        if feature != "crack_area_ratio"
    ]

    print(
        "Final spatial features:",
        len(spatial_features),
    )

    duplicates = (
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
        int(duplicates),
    )

    if duplicates > 0:

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

    print(
        "Manifest records:",
        len(manifest),
    )

    sequence_column = find_column(
        manifest,
        [
            "sequence",
            "sequence_id",
        ],
    )

    component_column = find_column(
        manifest,
        [
            "component",
            "component_id",
            "connected_component",
        ],
    )

    filename_column = find_column(
        manifest,
        [
            "canonical_image_filename",
            "canonical_filename",
        ],
        required=False,
    )

    frame_column = find_column(
        manifest,
        [
            "frame_number",
            "current_frame",
        ],
        required=False,
    )

    mapping = manifest.copy()

    mapping["sequence"] = (
        mapping[sequence_column]
        .astype(str)
        .str.strip()
    )

    mapping["component"] = pd.to_numeric(
        mapping[component_column],
        errors="coerce",
    )

    if frame_column is not None:

        mapping["current_frame"] = pd.to_numeric(
            mapping[frame_column],
            errors="coerce",
        )

        print(
            "Frame source:",
            frame_column,
        )

    elif filename_column is not None:

        print(
            "Frame source:",
            filename_column,
        )

        text = (
            mapping[filename_column]
            .astype(str)
            .str.strip()
        )

        mapping["current_frame"] = pd.to_numeric(
            text.str.extract(
                r"(\d+)(?:\.[A-Za-z0-9]+)?$",
                expand=False,
            ),
            errors="coerce",
        )

    else:

        raise RuntimeError(
            "Cannot derive frame number from manifest."
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

    mapping = mapping[
        [
            "sequence",
            "current_frame",
            "component",
        ]
    ].copy()

    duplicates = (
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

    print(
        "Duplicate sequence/frame mappings:",
        int(duplicates.sum()),
    )

    if duplicates.any():

        conflicts = (
            mapping[
                duplicates
            ]
            .groupby(
                [
                    "sequence",
                    "current_frame",
                ]
            )["component"]
            .nunique()
        )

        conflicts = conflicts[
            conflicts > 1
        ]

        print(
            "Conflicting component mappings:",
            len(conflicts),
        )

        if len(conflicts) > 0:

            raise RuntimeError(
                "Canonical manifest contains conflicting "
                "component mappings."
            )

        mapping = mapping.drop_duplicates(
            subset=[
                "sequence",
                "current_frame",
            ]
        )

    return mapping


# ============================================================================
# LOAD SPLITS
# ============================================================================

def load_assignments():

    print()
    print("=" * 78)
    print("LOADING COMPONENT SPLITS")
    print("=" * 78)

    assignments = pd.read_csv(
        ASSIGNMENT_FILE
    )

    component_column = find_column(
        assignments,
        [
            "component",
            "component_id",
            "component_index",
            "connected_component",
            "connected_component_id",
        ],
    )

    split_column = find_column(
        assignments,
        [
            "ml_split",
            "split",
            "assigned_split",
            "dataset_split",
        ],
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
# BUILD COMMON DATASET
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
            "Component mapping incomplete."
        )

    data = data.merge(
        assignments,
        on="component",
        how="left",
        validate="many_to_one",
    )

    print(
        "Missing splits:",
        int(
            data["ml_split"]
            .isna()
            .sum()
        ),
    )

    if data["ml_split"].isna().any():

        raise RuntimeError(
            "Split mapping incomplete."
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
            "Duplicate merged records."
        )

    return (
        data,
        temporal_features,
        spatial_features,
    )


# ============================================================================
# BUILD COMMON SUBSET
# ============================================================================

def prepare_common_subset(
    data,
    temporal_features,
    spatial_features,
):

    all_features = unique_preserve_order(
        temporal_features
        + spatial_features
    )

    required = unique_preserve_order(
        [
            "sequence",
            "current_frame",
            "component",
            "ml_split",
            "physical_target",
        ]
        + all_features
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
        subset=all_features
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
# SPATIAL TAXONOMY
# ============================================================================

def classify_spatial_feature(
    feature
):

    lower = feature.lower()

    # ------------------------------------------------------------------------
    # Explicit spatial families
    # ------------------------------------------------------------------------

    if (
        "acceleration" in lower
        or "accel" in lower
    ):

        return "SPATIAL_ACCELERATION"

    if (
        "trend" in lower
        or "slope" in lower
        or "velocity" in lower
    ):

        return "SPATIAL_TREND"

    if (
        "rolling" in lower
        or "roll" in lower
    ):

        return "SPATIAL_ROLLING"

    if (
        "delta" in lower
        or "change" in lower
        or "growth" in lower
    ):

        return "SPATIAL_CHANGE"

    if (
        "lag" in lower
    ):

        return "SPATIAL_LAG"

    # ------------------------------------------------------------------------
    # Other spatial measurements
    # ------------------------------------------------------------------------

    if any(
        token in lower
        for token in [
            "iou",
            "overlap",
            "center_distance",
            "area_similarity",
            "persistent_fraction",
            "future_coverage",
            "match_quality",
            "matched_",
        ]
    ):

        return "SPATIAL_MATCH"

    return "SPATIAL_OTHER"


# ============================================================================
# BUILD TAXONOMY
# ============================================================================

def build_taxonomy(
    spatial_features
):

    taxonomy = {}

    for feature in spatial_features:

        group = classify_spatial_feature(
            feature
        )

        taxonomy.setdefault(
            group,
            []
        ).append(
            feature
        )

    return taxonomy


# ============================================================================
# PRINT TAXONOMY
# ============================================================================

def print_taxonomy(
    taxonomy
):

    print()
    print("=" * 78)
    print("SPATIAL FEATURE TAXONOMY")
    print("=" * 78)

    for group, features in sorted(
        taxonomy.items()
    ):

        print(
            f"{group:<28}:",
            len(features),
        )


# ============================================================================
# MODEL
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

    if len(features) != len(
        set(features)
    ):

        raise RuntimeError(
            "Duplicate features in "
            + name
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

    train_metrics = metrics(
        y_train,
        train_pred,
    )

    val_metrics = metrics(
        y_val,
        val_pred,
    )

    test_metrics = metrics(
        y_test,
        test_pred,
    )

    print()
    print(
        "ABLATION:",
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

    result = {

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
    }

    return (
        result,
        model,
    )


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================

def permutation_importance_analysis(
    model,
    test,
    features,
    taxonomy,
):

    print()
    print("=" * 78)
    print("SPATIAL PERMUTATION IMPORTANCE")
    print("=" * 78)

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
                features,

            "permutation_importance":
                result.importances_mean,

            "importance_std":
                result.importances_std,
        }
    )

    mapping = {}

    for group, group_features in taxonomy.items():

        for feature in group_features:

            mapping[
                feature
            ] = group

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

def group_importance(
    importance
):

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

    print()
    print("=" * 78)
    print("SPATIAL GROUP IMPORTANCE")
    print("=" * 78)

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.20 - SPATIAL FEATURE IMPORTANCE AND ABLATION"
    )
    print("=" * 78)

    print()
    print(
        "Project:"
    )
    print(
        PROJECT_ROOT
    )

    # ------------------------------------------------------------------------
    # FILE CHECK
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("CHECKING INPUT FILES")
    print("=" * 78)

    inputs = [
        (
            "temporal_features",
            TEMPORAL_FILE,
        ),
        (
            "spatial_features",
            SPATIAL_FILE,
        ),
        (
            "physical_targets",
            TARGET_FILE,
        ),
        (
            "component_assignments",
            ASSIGNMENT_FILE,
        ),
        (
            "canonical_manifest",
            MANIFEST_FILE,
        ),
    ]

    for name, path in inputs:

        print(
            f"{name:<24}: "
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

    # ------------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------------

    (
        data,
        temporal_features,
        spatial_features,
    ) = build_dataset()

    # ------------------------------------------------------------------------
    # TAXONOMY
    # ------------------------------------------------------------------------

    taxonomy = build_taxonomy(
        spatial_features
    )

    print_taxonomy(
        taxonomy
    )

    # ------------------------------------------------------------------------
    # COMMON SUBSET
    # ------------------------------------------------------------------------

    common = prepare_common_subset(
        data,
        temporal_features,
        spatial_features,
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
    # FULL TEMPORAL + SPATIAL
    # ------------------------------------------------------------------------

    full_features = unique_preserve_order(
        temporal_features
        + spatial_features
    )

    print()
    print("=" * 78)
    print("FULL TEMPORAL + SPATIAL MODEL")
    print("=" * 78)

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
    # PERMUTATION IMPORTANCE
    # ------------------------------------------------------------------------

    importance = (
        permutation_importance_analysis(
            full_model,
            test,
            full_features,
            {
                "TEMPORAL":
                    temporal_features,

                **taxonomy,
            },
        )
    )

    # Keep only spatial features for the dedicated spatial ranking.
    spatial_importance = importance[
        importance["feature"].isin(
            spatial_features
        )
    ].copy()

    spatial_importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # GROUP IMPORTANCE
    # ------------------------------------------------------------------------

    spatial_group_importance = (
        group_importance(
            spatial_importance
        )
    )

    spatial_group_importance.to_csv(
        GROUP_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # ABLATIONS
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SPATIAL GROUP ABLATION")
    print("=" * 78)

    ablation_sets = {}

    # Baseline temporal model.
    ablation_sets[
        "TEMPORAL_ONLY"
    ] = unique_preserve_order(
        temporal_features
    )

    # Add one spatial family at a time.
    for group, group_features in taxonomy.items():

        ablation_sets[
            "TEMPORAL_PLUS_"
            + group
        ] = unique_preserve_order(
            temporal_features
            + group_features
        )

    # Remove one spatial group at a time.
    for group, group_features in taxonomy.items():

        remaining = [
            feature
            for feature in spatial_features
            if feature not in group_features
        ]

        ablation_sets[
            "FULL_WITHOUT_"
            + group
        ] = unique_preserve_order(
            temporal_features
            + remaining
        )

    # Full model.
    ablation_sets[
        "FULL_TEMPORAL_PLUS_SPATIAL"
    ] = full_features

    ablation_results = []

    for name, features in ablation_sets.items():

        result, _ = run_model(
            train,
            val,
            test,
            features,
            name,
        )

        ablation_results.append(
            result
        )

    ablation_df = pd.DataFrame(
        ablation_results
    )

    ablation_df = ablation_df.sort_values(
        "test_MAE"
    ).reset_index(
        drop=True
    )

    ablation_df.to_csv(
        ABLATION_FILE,
        index=False,
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

    persistence_metrics = metrics(
        test[
            "physical_target"
        ].to_numpy(
            dtype=np.float64
        ),
        persistence_pred,
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
        "TEST R2:",
        f"{persistence_metrics['R2']:.8f}",
    )

    # ------------------------------------------------------------------------
    # IMPORTANT COMPARISONS
    # ------------------------------------------------------------------------

    temporal_row = ablation_df[
        ablation_df["model"]
        ==
        "TEMPORAL_ONLY"
    ].iloc[0]

    full_row = ablation_df[
        ablation_df["model"]
        ==
        "FULL_TEMPORAL_PLUS_SPATIAL"
    ].iloc[0]

    temporal_mae = float(
        temporal_row["test_MAE"]
    )

    full_mae = float(
        full_row["test_MAE"]
    )

    persistence_mae = float(
        persistence_metrics["MAE"]
    )

    spatial_gain = (
        (
            temporal_mae
            -
            full_mae
        )
        /
        temporal_mae
    ) * 100.0

    persistence_gain = (
        (
            persistence_mae
            -
            full_mae
        )
        /
        persistence_mae
    ) * 100.0

    # ------------------------------------------------------------------------
    # BEST VALIDATION / TEST
    # ------------------------------------------------------------------------

    best_validation = (
        ablation_df
        .sort_values(
            "val_MAE"
        )
        .iloc[0]
    )

    best_test = (
        ablation_df
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
            "3.20",

        "title":
            "Spatial Feature Importance and Ablation",

        "target":
            "crack_area_ratio(t+1)",

        "common_subset":
            {
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

        "spatial_taxonomy":
            taxonomy,

        "persistence_test":
            persistence_metrics,

        "temporal_only_test":
            {
                "MAE":
                    float(
                        temporal_row["test_MAE"]
                    ),

                "RMSE":
                    float(
                        temporal_row["test_RMSE"]
                    ),

                "R2":
                    float(
                        temporal_row["test_R2"]
                    ),
            },

        "full_temporal_spatial_test":
            {
                "MAE":
                    float(
                        full_row["test_MAE"]
                    ),

                "RMSE":
                    float(
                        full_row["test_RMSE"]
                    ),

                "R2":
                    float(
                        full_row["test_R2"]
                    ),
            },

        "spatial_gain_over_temporal_percent":
            float(
                spatial_gain
            ),

        "combined_gain_over_persistence_percent":
            float(
                persistence_gain
            ),

        "top_spatial_features":
            spatial_importance.head(
                20
            ).to_dict(
                orient="records"
            ),

        "spatial_group_importance":
            spatial_group_importance.to_dict(
                orient="records"
            ),

        "best_validation_ablation":
            best_validation.to_dict(),

        "best_test_ablation":
            best_test.to_dict(),

        "leakage_safe":
            True,

        "target_modified":
            False,

        "decision":
            "SPATIAL_FEATURE_IMPORTANCE_COMPLETE",

        "next_stage":
            "SPATIAL_FEATURE_REDUCTION_AND_FINAL_MODEL",
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
    # FINAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "STAGE 3.20 COMPLETE"
    )
    print("=" * 78)

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

    print()
    print(
        "Temporal-only TEST MAE:",
        f"{temporal_mae:.8f}",
    )

    print(
        "Full temporal + spatial TEST MAE:",
        f"{full_mae:.8f}",
    )

    print(
        "Spatial gain over temporal:",
        f"{spatial_gain:.4f}%",
    )

    print(
        "Combined gain over persistence:",
        f"{persistence_gain:.4f}%",
    )

    print()
    print(
        "Best validation ablation:",
        best_validation["model"],
    )

    print(
        "Best validation MAE:",
        f"{best_validation['val_MAE']:.8f}",
    )

    print()
    print(
        "Best test ablation:",
        best_test["model"],
    )

    print(
        "Best test MAE:",
        f"{best_test['test_MAE']:.8f}",
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
        IMPORTANCE_FILE
    )

    print(
        GROUP_FILE
    )

    print(
        ABLATION_FILE
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()