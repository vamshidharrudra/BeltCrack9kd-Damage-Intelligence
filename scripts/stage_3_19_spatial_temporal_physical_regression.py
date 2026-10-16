import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

CANONICAL_DIR = (
    PROJECT_ROOT / "canonical_dataset"
)


# ============================================================================
# INPUT FILES
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
    CANONICAL_DIR /
    "canonical_manifest.csv"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

RESULTS_FILE = (
    AUDIT_DIR /
    "stage_3_19_spatial_temporal_physical_regression_results.csv"
)

PREDICTIONS_FILE = (
    AUDIT_DIR /
    "stage_3_19_spatial_temporal_physical_predictions.csv"
)

FEATURE_SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_19_spatial_temporal_feature_summary.csv"
)

SUMMARY_FILE = (
    AUDIT_DIR /
    "stage_3_19_spatial_temporal_physical_regression_summary.json"
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
            "\nCould not find required column.\n"
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

            result.append(column)
            seen.add(column)

    return result


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
# STANDARDIZE FRAME KEY
# ============================================================================

def standardize_frame_key(
    df,
    name,
):

    print()
    print(
        "Checking frame key for:",
        name,
    )

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

    df = df.copy()

    df["sequence"] = (
        df[sequence_col]
        .astype(str)
        .str.strip()
    )

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    print(
        "Sequence column:",
        sequence_col,
    )

    print(
        "Frame column:",
        frame_col,
    )

    print(
        "Missing frame numbers:",
        int(
            df["current_frame"]
            .isna()
            .sum()
        ),
    )

    if df["current_frame"].isna().any():

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

    print(
        "Target records:",
        len(target),
    )

    target = standardize_frame_key(
        target,
        "physical target",
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
        "Duplicate target keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate physical target keys found."
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

    print(
        "Temporal records:",
        len(temporal),
    )

    temporal = standardize_frame_key(
        temporal,
        "temporal features",
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

        print(
            missing
        )

        raise RuntimeError(
            "Stage 3.17 temporal feature schema changed."
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
            "Duplicate temporal keys found."
        )

    return temporal, temporal_features


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
        "spatial features",
    )

    duplicate_columns = (
        spatial.columns[
            spatial.columns.duplicated()
        ]
        .tolist()
    )

    print(
        "Duplicate spatial dataframe columns:",
        len(duplicate_columns),
    )

    if duplicate_columns:

        print(
            duplicate_columns
        )

        raise RuntimeError(
            "Spatial feature file contains duplicate columns."
        )

    identifier_columns = {
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

        if column in identifier_columns:
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

    # Never duplicate the temporal current-state feature.
    spatial_features = [
        feature
        for feature in spatial_features
        if feature != "crack_area_ratio"
    ]

    print(
        "Candidate spatial features:",
        len(spatial_features),
    )

    print(
        "Spatial features after safety filtering:",
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
            "Duplicate spatial keys found."
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
# LOAD CANONICAL MANIFEST
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

    print(
        "Manifest columns:"
    )

    for column in manifest.columns:

        print(
            "  ",
            column,
        )

    return manifest


# ============================================================================
# DERIVE COMPONENT FROM MANIFEST
# ============================================================================

def build_frame_component_mapping(
    manifest
):

    print()
    print("=" * 78)
    print("BUILDING FRAME -> COMPONENT MAPPING")
    print("=" * 78)

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
            "connected_component",
        ],
    )

    canonical_frame_col = find_column(
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

    mapping = manifest.copy()

    mapping["sequence"] = (
        mapping[sequence_col]
        .astype(str)
        .str.strip()
    )

    mapping["component"] = pd.to_numeric(
        mapping[component_col],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # FRAME EXTRACTION
    # ------------------------------------------------------------------------

    if frame_col is not None:

        mapping["current_frame"] = pd.to_numeric(
            mapping[frame_col],
            errors="coerce",
        )

        print(
            "Frame source:",
            frame_col,
        )

    elif canonical_frame_col is not None:

        print(
            "Frame source:",
            canonical_frame_col,
        )

        text = (
            mapping[canonical_frame_col]
            .astype(str)
            .str.strip()
        )

        # Extract final numeric stem.
        mapping["current_frame"] = pd.to_numeric(
            text.str.extract(
                r"(\d+)(?:\.[A-Za-z0-9]+)?$",
                expand=False,
            ),
            errors="coerce",
        )

    else:

        raise RuntimeError(
            "Canonical manifest contains neither frame_number "
            "nor canonical_image_filename."
        )

    missing_frames = int(
        mapping["current_frame"]
        .isna()
        .sum()
    )

    missing_components = int(
        mapping["component"]
        .isna()
        .sum()
    )

    print(
        "Missing manifest frames:",
        missing_frames,
    )

    print(
        "Missing manifest components:",
        missing_components,
    )

    if missing_frames > 0:

        raise RuntimeError(
            "Could not derive all frame numbers from canonical manifest."
        )

    if missing_components > 0:

        raise RuntimeError(
            "Could not derive all component IDs from canonical manifest."
        )

    mapping = mapping[
        [
            "sequence",
            "current_frame",
            "component",
        ]
    ].copy()

    # ------------------------------------------------------------------------
    # CHECK KEY UNIQUENESS
    # ------------------------------------------------------------------------

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

    duplicate_count = int(
        duplicate_keys.sum()
    )

    print(
        "Duplicate sequence/frame mappings:",
        duplicate_count,
    )

    if duplicate_count > 0:

        conflicting = (
            mapping[
                duplicate_keys
            ]
            .groupby(
                [
                    "sequence",
                    "current_frame",
                ]
            )[
                "component"
            ]
            .nunique()
        )

        conflicting = conflicting[
            conflicting > 1
        ]

        print(
            "Conflicting component mappings:",
            len(conflicting),
        )

        if len(conflicting) > 0:

            raise RuntimeError(
                "A sequence/frame maps to multiple components."
            )

        mapping = mapping.drop_duplicates(
            subset=[
                "sequence",
                "current_frame",
            ]
        )

    return mapping


# ============================================================================
# LOAD COMPONENT SPLITS
# ============================================================================

def load_assignments():

    print()
    print("=" * 78)
    print("LOADING LEAKAGE-SAFE COMPONENT SPLITS")
    print("=" * 78)

    assignments = pd.read_csv(
        ASSIGNMENT_FILE
    )

    print(
        "Assignment records:",
        len(assignments),
    )

    print(
        "Assignment columns:",
        list(assignments.columns),
    )

    component_column = find_column(
        assignments,
        [
            "component",
            "component_id",
            "component_index",
            "connected_component",
            "connected_component_id",
            "component_number",
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
            "Component split leakage detected."
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

    frame_component = (
        build_frame_component_mapping(
            manifest
        )
    )

    assignments = load_assignments()

    # ------------------------------------------------------------------------
    # TARGET + TEMPORAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("MERGING PHYSICAL REGRESSION DATA")
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
        "After target + temporal:",
        len(data),
    )

    # ------------------------------------------------------------------------
    # + SPATIAL
    # ------------------------------------------------------------------------

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
        "After spatial merge:",
        len(data),
    )

    # ------------------------------------------------------------------------
    # + COMPONENT
    # ------------------------------------------------------------------------

    data = data.merge(
        frame_component,
        on=[
            "sequence",
            "current_frame",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        "Missing component mappings:",
        int(
            data["component"]
            .isna()
            .sum()
        ),
    )

    if data["component"].isna().any():

        missing_examples = (
            data[
                data["component"].isna()
            ][
                [
                    "sequence",
                    "current_frame",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        print(
            "Example missing mappings:",
            missing_examples,
        )

        raise RuntimeError(
            "Some regression records could not be mapped "
            "to a canonical component."
        )

    # ------------------------------------------------------------------------
    # + SPLIT
    # ------------------------------------------------------------------------

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
            "Some components have no leakage-safe ML split."
        )

    # ------------------------------------------------------------------------
    # FINAL INTEGRITY
    # ------------------------------------------------------------------------

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
        "Final duplicate sequence/frame keys:",
        int(duplicate_keys),
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate records after all merges."
        )

    component_conflicts = (
        data
        .groupby("component")["ml_split"]
        .nunique()
    )

    component_conflicts = (
        component_conflicts[
            component_conflicts > 1
        ]
    )

    print(
        "Final component split conflicts:",
        len(component_conflicts),
    )

    if len(component_conflicts) > 0:

        raise RuntimeError(
            "Component leakage detected."
        )

    return (
        data,
        temporal_features,
        spatial_features,
    )


# ============================================================================
# COMMON SUBSET
# ============================================================================

def prepare_common_subset(
    data,
    temporal_features,
    spatial_features,
):

    print()
    print("=" * 78)
    print("BUILDING COMMON COMPLETE SUBSET")
    print("=" * 78)

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
# FEATURE SETS
# ============================================================================

def build_feature_sets(
    temporal_features,
    spatial_features,
):

    current = [
        "crack_area_ratio"
    ]

    temporal = unique_preserve_order(
        temporal_features
    )

    spatial = unique_preserve_order(
        spatial_features
    )

    return {

        "CURRENT_ONLY":
            current,

        "TEMPORAL_ONLY":
            temporal,

        "SPATIAL_ONLY":
            spatial,

        "CURRENT_PLUS_SPATIAL":
            unique_preserve_order(
                current + spatial
            ),

        "TEMPORAL_PLUS_SPATIAL":
            unique_preserve_order(
                temporal + spatial
            ),
    }


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

    print()
    print(
        "MODEL:",
        name,
    )

    print(
        "Feature count:",
        len(features),
    )

    if len(features) != len(set(features)):

        raise RuntimeError(
            "Duplicate features detected in "
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

    result = {

        "model":
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

    return (
        result,
        test_prediction,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.19 - SPATIAL + TEMPORAL PHYSICAL REGRESSION"
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
    # CHECK INPUTS
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
    # BUILD
    # ------------------------------------------------------------------------

    (
        data,
        temporal_features,
        spatial_features,
    ) = build_dataset()

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
    # FEATURE SETS
    # ------------------------------------------------------------------------

    feature_sets = build_feature_sets(
        temporal_features,
        spatial_features,
    )

    print()
    print("=" * 78)
    print("FEATURE SET SUMMARY")
    print("=" * 78)

    feature_summary_records = []

    for name, features in feature_sets.items():

        print(
            f"{name:<28}:",
            len(features),
        )

        feature_summary_records.append(
            {
                "model":
                    name,

                "feature_count":
                    len(features),

                "features":
                    "|".join(features),
            }
        )

    feature_summary = pd.DataFrame(
        feature_summary_records
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # RUN MODELS
    # ------------------------------------------------------------------------

    results = []
    predictions = []

    for name, features in feature_sets.items():

        (
            result,
            test_prediction,
        ) = run_model(
            train,
            val,
            test,
            features,
            name,
        )

        results.append(
            result
        )

        table = test[
            [
                "sequence",
                "current_frame",
                "component",
                "physical_target",
                "crack_area_ratio",
            ]
        ].copy()

        table["model"] = name

        table["prediction"] = (
            test_prediction
        )

        table["absolute_error"] = np.abs(
            table["physical_target"]
            -
            table["prediction"]
        )

        predictions.append(
            table
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

    results.append(
        {
            "model":
                "PERSISTENCE",

            "feature_count":
                1,

            "train_records":
                len(train),

            "val_records":
                len(val),

            "test_records":
                len(test),

            "train_MAE":
                np.nan,

            "train_RMSE":
                np.nan,

            "train_R2":
                np.nan,

            "val_MAE":
                np.nan,

            "val_RMSE":
                np.nan,

            "val_R2":
                np.nan,

            "test_MAE":
                persistence_metrics["MAE"],

            "test_RMSE":
                persistence_metrics["RMSE"],

            "test_R2":
                persistence_metrics["R2"],
        }
    )

    persistence_table = test[
        [
            "sequence",
            "current_frame",
            "component",
            "physical_target",
            "crack_area_ratio",
        ]
    ].copy()

    persistence_table["model"] = (
        "PERSISTENCE"
    )

    persistence_table["prediction"] = (
        persistence_prediction
    )

    persistence_table["absolute_error"] = (
        np.abs(
            persistence_table[
                "physical_target"
            ]
            -
            persistence_table[
                "prediction"
            ]
        )
    )

    predictions.append(
        persistence_table
    )

    # ------------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "test_MAE"
    ).reset_index(
        drop=True
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # COMPARISONS
    # ------------------------------------------------------------------------

    temporal_row = results_df[
        results_df["model"]
        ==
        "TEMPORAL_ONLY"
    ].iloc[0]

    spatial_row = results_df[
        results_df["model"]
        ==
        "SPATIAL_ONLY"
    ].iloc[0]

    combined_row = results_df[
        results_df["model"]
        ==
        "TEMPORAL_PLUS_SPATIAL"
    ].iloc[0]

    persistence_mae = (
        persistence_metrics["MAE"]
    )

    temporal_mae = (
        temporal_row["test_MAE"]
    )

    spatial_mae = (
        spatial_row["test_MAE"]
    )

    combined_mae = (
        combined_row["test_MAE"]
    )

    temporal_improvement = (
        (
            persistence_mae
            -
            temporal_mae
        )
        /
        persistence_mae
    ) * 100.0

    spatial_improvement = (
        (
            persistence_mae
            -
            spatial_mae
        )
        /
        persistence_mae
    ) * 100.0

    combined_improvement = (
        (
            persistence_mae
            -
            combined_mae
        )
        /
        persistence_mae
    ) * 100.0

    spatial_gain_over_temporal = (
        (
            temporal_mae
            -
            combined_mae
        )
        /
        temporal_mae
    ) * 100.0

    # ------------------------------------------------------------------------
    # BEST MODELS
    # ------------------------------------------------------------------------

    ml_results = results_df[
        results_df["model"]
        !=
        "PERSISTENCE"
    ].copy()

    best_validation = (
        ml_results
        .sort_values(
            "val_MAE"
        )
        .iloc[0]
    )

    best_test = (
        results_df
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
            "3.19",

        "title":
            "Spatial + Temporal Physical Damage Regression",

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

        "persistence_test":
            persistence_metrics,

        "temporal_test":
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

        "spatial_test":
            {
                "MAE":
                    float(
                        spatial_row["test_MAE"]
                    ),

                "RMSE":
                    float(
                        spatial_row["test_RMSE"]
                    ),

                "R2":
                    float(
                        spatial_row["test_R2"]
                    ),
            },

        "temporal_plus_spatial_test":
            {
                "MAE":
                    float(
                        combined_row["test_MAE"]
                    ),

                "RMSE":
                    float(
                        combined_row["test_RMSE"]
                    ),

                "R2":
                    float(
                        combined_row["test_R2"]
                    ),
            },

        "temporal_improvement_vs_persistence_percent":
            float(
                temporal_improvement
            ),

        "spatial_improvement_vs_persistence_percent":
            float(
                spatial_improvement
            ),

        "combined_improvement_vs_persistence_percent":
            float(
                combined_improvement
            ),

        "spatial_gain_over_temporal_percent":
            float(
                spatial_gain_over_temporal
            ),

        "best_validation_model":
            str(
                best_validation["model"]
            ),

        "best_validation_MAE":
            float(
                best_validation["val_MAE"]
            ),

        "best_test_model":
            str(
                best_test["model"]
            ),

        "best_test_MAE":
            float(
                best_test["test_MAE"]
            ),

        "component_mapping_source":
            "canonical_manifest.csv",

        "leakage_safe_component_split":
            True,

        "target_modified":
            False,

        "decision":
            "SPATIAL_TEMPORAL_PHYSICAL_REGRESSION_COMPLETE",
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
        "STAGE 3.19 COMPLETE"
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
        "Persistence TEST MAE:",
        f"{persistence_mae:.8f}",
    )

    print(
        "Temporal TEST MAE:",
        f"{temporal_mae:.8f}",
    )

    print(
        "Spatial TEST MAE:",
        f"{spatial_mae:.8f}",
    )

    print(
        "Temporal + Spatial TEST MAE:",
        f"{combined_mae:.8f}",
    )

    print()
    print(
        "Temporal improvement vs persistence:",
        f"{temporal_improvement:.4f}%",
    )

    print(
        "Spatial improvement vs persistence:",
        f"{spatial_improvement:.4f}%",
    )

    print(
        "Combined improvement vs persistence:",
        f"{combined_improvement:.4f}%",
    )

    print(
        "Spatial gain over temporal:",
        f"{spatial_gain_over_temporal:.4f}%",
    )

    print()
    print(
        "Best validation model:",
        best_validation["model"],
    )

    print(
        "Best test model:",
        best_test["model"],
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
        RESULTS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        FEATURE_SUMMARY_FILE
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()