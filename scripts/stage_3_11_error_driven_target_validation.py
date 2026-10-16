import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

PREDICTIONS_FILE = (
    AUDIT_DIR /
    "stage_3_10_test_predictions.csv"
)

TARGETS_FILE = (
    AUDIT_DIR /
    "stage_3_6_persistent_damage_targets.csv"
)

SPATIAL_FILE = (
    AUDIT_DIR /
    "stage_3_1_spatial_track_features.csv"
)

TEMPORAL_FILE = (
    AUDIT_DIR /
    "stage_2_2_temporal_features.csv"
)

DAMAGE_FEATURE_FILE = (
    AUDIT_DIR /
    "stage_1_0_damage_features.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_error_driven_target_validation.json"
)

ERROR_PROFILE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_error_profile.csv"
)

CLASS_PAIR_PROFILE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_class_pair_profiles.csv"
)

TARGET_BOUNDARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_target_boundary_analysis.csv"
)

SEQUENCE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_sequence_error_profile.csv"
)

HIGH_CONF_OUTPUT = (
    AUDIT_DIR /
    "stage_3_11_high_confidence_error_profile.csv"
)


# ============================================================================
# JSON SAFE
# ============================================================================

def to_json_safe(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(value, np.ndarray):

        return to_json_safe(
            value.tolist()
        )

    if isinstance(value, dict):

        return {
            str(k): to_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            to_json_safe(v)
            for v in value
        ]

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    return value


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def resolve_file(
    exact_path,
    required_columns,
    keywords,
):

    if exact_path.exists():
        return exact_path

    candidates = []

    for path in AUDIT_DIR.rglob("*.csv"):

        try:

            columns = pd.read_csv(
                path,
                nrows=0,
            ).columns.tolist()

        except Exception:

            continue

        if not all(
            column in columns
            for column in required_columns
        ):

            continue

        score = 0

        filename = path.name.lower()

        for keyword in keywords:

            if keyword.lower() in filename:
                score += 10

        candidates.append(
            (
                score,
                str(path),
                path,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    return candidates[0][2]


# ============================================================================
# COLUMN HELPERS
# ============================================================================

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
            "Could not find required column: "
            + str(candidates)
        )

    return None


def numeric_series(
    df,
    candidates,
):

    column = find_column(
        df,
        candidates,
        required=False,
    )

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================================
# KEY STANDARDIZATION
# ============================================================================

def standardize_prediction_keys(df):

    df = df.copy()

    sequence_col = find_column(
        df,
        ["sequence"],
    )

    component_col = find_column(
        df,
        ["component"],
    )

    frame_col = find_column(
        df,
        [
            "current_frame",
            "frame_number",
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

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    return df


def standardize_target_keys(df):

    df = df.copy()

    sequence_col = find_column(
        df,
        ["sequence"],
    )

    component_col = find_column(
        df,
        ["component"],
    )

    frame_col = find_column(
        df,
        [
            "current_frame",
            "frame_number",
            "target_current_frame",
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

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    return df


def standardize_spatial_keys(df):

    df = df.copy()

    sequence_col = find_column(
        df,
        ["sequence"],
    )

    component_col = find_column(
        df,
        ["component"],
    )

    frame_col = find_column(
        df,
        [
            "current_frame",
            "frame_number",
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

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    return df


def standardize_temporal_keys(df):

    df = df.copy()

    sequence_col = find_column(
        df,
        ["sequence"],
    )

    component_col = find_column(
        df,
        ["component"],
    )

    frame_col = find_column(
        df,
        [
            "frame_number",
            "current_frame",
            "target_current_frame",
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

    df["current_frame"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    return df


# ============================================================================
# FEATURE SEARCH
# ============================================================================

def find_feature(
    df,
    exact_names=None,
    contains=None,
):

    exact_names = exact_names or []
    contains = contains or []

    for name in exact_names:

        if name in df.columns:
            return name

    for column in df.columns:

        low = column.lower()

        if all(
            token.lower() in low
            for token in contains
        ):

            return column

    return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.11 - ERROR-DRIVEN TARGET VALIDATION"
    )
    print("=" * 78)

    print()
    print("Project:")
    print(PROJECT_ROOT)

    # ========================================================================
    # FILE DISCOVERY
    # ========================================================================

    print()
    print("=" * 78)
    print("CHECKING INPUT FILES")
    print("=" * 78)

    prediction_file = resolve_file(
        PREDICTIONS_FILE,
        [
            "sequence",
            "component",
            "persistent_damage_target",
            "predicted_class",
        ],
        [
            "stage_3_10",
            "prediction",
        ],
    )

    target_file = resolve_file(
        TARGETS_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_3_6",
            "target",
        ],
    )

    spatial_file = resolve_file(
        SPATIAL_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_3_1",
            "spatial",
        ],
    )

    temporal_file = resolve_file(
        TEMPORAL_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_2_2",
            "temporal",
        ],
    )

    damage_file = resolve_file(
        DAMAGE_FEATURE_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_1_0",
            "damage",
            "feature",
        ],
    )

    files = {
        "predictions": prediction_file,
        "targets": target_file,
        "spatial": spatial_file,
        "temporal": temporal_file,
        "damage_features": damage_file,
    }

    for name, path in files.items():

        if path is None:

            print(
                f"{name:<20}: NOT FOUND"
            )

        else:

            print(
                f"{name:<20}: FOUND"
            )

            print(
                "   ",
                path,
            )

    if prediction_file is None:
        raise FileNotFoundError(
            "Stage 3.10 prediction file not found."
        )

    if target_file is None:
        raise FileNotFoundError(
            "Stage 3.6 target file not found."
        )

    if spatial_file is None:
        raise FileNotFoundError(
            "Stage 3.1 spatial feature file not found."
        )

    # ========================================================================
    # LOAD
    # ========================================================================

    print()
    print("=" * 78)
    print("LOADING DATA")
    print("=" * 78)

    predictions = pd.read_csv(
        prediction_file
    )

    targets = pd.read_csv(
        target_file
    )

    spatial = pd.read_csv(
        spatial_file
    )

    temporal = None

    if temporal_file is not None:

        temporal = pd.read_csv(
            temporal_file
        )

    damage_features = None

    if damage_file is not None:

        damage_features = pd.read_csv(
            damage_file
        )

    print(
        "Predictions:",
        len(predictions),
    )

    print(
        "Targets:",
        len(targets),
    )

    print(
        "Spatial records:",
        len(spatial),
    )

    if temporal is not None:

        print(
            "Temporal records:",
            len(temporal),
        )

    if damage_features is not None:

        print(
            "Damage feature records:",
            len(damage_features),
        )

    # ========================================================================
    # SCHEMA REPORT
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "INPUT SCHEMA CHECK"
    )
    print("=" * 78)

    print()
    print(
        "Stage 1.0 columns:"
    )

    if damage_features is not None:

        print(
            ", ".join(
                damage_features.columns
            )
        )

    print()
    print(
        "Important: Stage 1.0 is optional for "
        "Stage 3.11 key merging."
    )

    # ========================================================================
    # STANDARDIZE ONLY REQUIRED DATASETS
    # ========================================================================

    predictions = standardize_prediction_keys(
        predictions
    )

    targets = standardize_target_keys(
        targets
    )

    spatial = standardize_spatial_keys(
        spatial
    )

    if temporal is not None:

        temporal = standardize_temporal_keys(
            temporal
        )

    # ========================================================================
    # DUPLICATE CHECK
    # ========================================================================

    key_columns = [
        "sequence",
        "component",
        "current_frame",
    ]

    print()
    print("=" * 78)
    print(
        "KEY INTEGRITY"
    )
    print("=" * 78)

    prediction_duplicates = int(
        predictions[
            key_columns
        ].duplicated()
        .sum()
    )

    target_duplicates = int(
        targets[
            key_columns
        ].duplicated()
        .sum()
    )

    spatial_duplicates = int(
        spatial[
            key_columns
        ].duplicated()
        .sum()
    )

    print(
        "Prediction duplicate keys:",
        prediction_duplicates,
    )

    print(
        "Target duplicate keys:",
        target_duplicates,
    )

    print(
        "Spatial duplicate keys:",
        spatial_duplicates,
    )

    if prediction_duplicates > 0:
        raise RuntimeError(
            "Duplicate prediction keys detected."
        )

    if target_duplicates > 0:
        raise RuntimeError(
            "Duplicate target keys detected."
        )

    if spatial_duplicates > 0:
        raise RuntimeError(
            "Duplicate spatial keys detected."
        )

    # ========================================================================
    # BUILD BASE ERROR TABLE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "BUILDING ERROR ANALYSIS TABLE"
    )
    print("=" * 78)

    prediction_columns = [
        "sequence",
        "component",
        "current_frame",
        "persistent_damage_target",
        "predicted_class",
        "correct",
        "prediction_confidence",
        "true_class_probability",
        "confidence_margin",
    ]

    available_prediction_columns = [
        column
        for column in prediction_columns
        if column in predictions.columns
    ]

    data = predictions[
        available_prediction_columns
    ].copy()

    # ------------------------------------------------------------------------
    # Verify against Stage 3.6 target
    # ------------------------------------------------------------------------

    target_label_column = find_column(
        targets,
        [
            "persistent_damage_target",
            "target_state",
        ],
    )

    target_subset = targets[
        key_columns
        + [target_label_column]
    ].copy()

    target_subset = target_subset.rename(
        columns={
            target_label_column:
                "verified_target"
        }
    )

    data = data.merge(
        target_subset,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    target_match = (
        data[
            "persistent_damage_target"
        ].astype(str)
        ==
        data[
            "verified_target"
        ].astype(str)
    )

    target_mismatch_count = int(
        (~target_match).sum()
    )

    print(
        "Prediction/target mismatches:",
        target_mismatch_count,
    )

    if target_mismatch_count > 0:

        raise RuntimeError(
            "Stage 3.10 predictions do not match "
            "Stage 3.6 targets."
        )

    data = data.drop(
        columns=[
            "verified_target"
        ]
    )

    # ========================================================================
    # MERGE SPATIAL
    # ========================================================================

    spatial_columns = [
        column
        for column in spatial.columns
        if column not in key_columns
    ]

    spatial_subset = spatial[
        key_columns
        + spatial_columns
    ].copy()

    data = data.merge(
        spatial_subset,
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_spatial",
        ),
    )

    print(
        "Merged records:",
        len(data),
    )

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================

    if temporal is not None:

        print()
        print(
            "ADDING TEMPORAL ERROR CONTEXT"
        )

        temporal_columns = [
            column
            for column in temporal.columns
            if column not in key_columns
        ]

        selected_temporal = []

        preferred_tokens = [
            "crack_area_ratio",
            "crack_count",
            "delta1",
            "delta2",
            "rolling3",
            "growth",
            "slope",
        ]

        for column in temporal_columns:

            low = column.lower()

            if any(
                token in low
                for token in preferred_tokens
            ):

                selected_temporal.append(
                    column
                )

        selected_temporal = list(
            dict.fromkeys(
                selected_temporal
            )
        )

        if len(
            selected_temporal
        ) > 40:

            selected_temporal = (
                selected_temporal[:40]
            )

        temporal_subset = temporal[
            key_columns
            + selected_temporal
        ].copy()

        data = data.merge(
            temporal_subset,
            on=key_columns,
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_temporal",
            ),
        )

        print(
            "Temporal context columns:",
            len(selected_temporal),
        )

    # ========================================================================
    # IDENTIFY USEFUL METRICS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "IDENTIFYING DAMAGE / SPATIAL METRICS"
    )
    print("=" * 78)

    feature_map = {}

    feature_map[
        "crack_area_ratio"
    ] = find_feature(
        data,
        exact_names=[
            "current_crack_area_ratio",
            "crack_area_ratio",
        ],
    )

    feature_map[
        "crack_count"
    ] = find_feature(
        data,
        exact_names=[
            "current_crack_count_spatial",
            "current_crack_count",
            "crack_count",
        ],
    )

    feature_map[
        "spatial_match_quality"
    ] = find_feature(
        data,
        exact_names=[
            "spatial_match_quality",
        ],
    )

    feature_map[
        "mean_match_iou"
    ] = find_feature(
        data,
        exact_names=[
            "current_mean_match_iou",
            "mean_match_iou",
        ],
        contains=[
            "mean_match_iou",
        ],
    )

    feature_map[
        "mean_center_distance"
    ] = find_feature(
        data,
        exact_names=[
            "current_mean_center_distance",
            "mean_center_distance",
        ],
        contains=[
            "mean_center_distance",
        ],
    )

    feature_map[
        "mean_area_similarity"
    ] = find_feature(
        data,
        exact_names=[
            "current_mean_area_similarity",
            "mean_area_similarity",
        ],
        contains=[
            "mean_area_similarity",
        ],
    )

    feature_map[
        "persistent_fraction"
    ] = find_feature(
        data,
        exact_names=[
            "current_persistent_fraction",
            "persistent_fraction",
        ],
        contains=[
            "persistent_fraction",
        ],
    )

    feature_map[
        "future_coverage"
    ] = find_feature(
        data,
        exact_names=[
            "current_future_coverage",
            "future_coverage",
        ],
        contains=[
            "future_coverage",
        ],
    )

    feature_map[
        "matched_area_relative_change"
    ] = find_feature(
        data,
        exact_names=[
            "current_matched_area_relative_change",
            "matched_area_relative_change",
        ],
        contains=[
            "matched_area_relative_change",
        ],
    )

    for name, column in feature_map.items():

        if column is None:

            print(
                f"{name:<35}: NOT FOUND"
            )

        else:

            print(
                f"{name:<35}: {column}"
            )

    # ========================================================================
    # CONSTRUCT DERIVED MAGNITUDES
    # ========================================================================

    data[
        "damage_level"
    ] = numeric_series(
        data,
        [
            feature_map[
                "crack_area_ratio"
            ]
            if feature_map[
                "crack_area_ratio"
            ]
            else "___missing___",
        ],
    )

    # ------------------------------------------------------------------------
    # Temporal delta features
    # ------------------------------------------------------------------------

    delta_columns = [
        column
        for column in data.columns
        if (
            "delta" in column.lower()
            and
            (
                "area" in column.lower()
                or
                "crack" in column.lower()
            )
        )
    ]

    if delta_columns:

        delta_numeric = data[
            delta_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        data[
            "absolute_delta_magnitude"
        ] = (
            delta_numeric
            .abs()
            .max(
                axis=1
            )
        )

        data[
            "signed_delta_mean"
        ] = (
            delta_numeric
            .mean(
                axis=1
            )
        )

    else:

        data[
            "absolute_delta_magnitude"
        ] = np.nan

        data[
            "signed_delta_mean"
        ] = np.nan

    # ------------------------------------------------------------------------
    # Spatial change features
    # ------------------------------------------------------------------------

    spatial_change_columns = [
        column
        for column in data.columns
        if (
            column.startswith(
                "spatial_delta"
            )
        )
    ]

    if spatial_change_columns:

        spatial_change_numeric = (
            data[
                spatial_change_columns
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        data[
            "spatial_change_magnitude"
        ] = (
            spatial_change_numeric
            .abs()
            .mean(
                axis=1
            )
        )

    else:

        data[
            "spatial_change_magnitude"
        ] = np.nan

    # ========================================================================
    # ERROR FLAGS
    # ========================================================================

    data[
        "is_error"
    ] = (
        data[
            "correct"
        ]
        == False
    )

    data[
        "is_high_confidence_error"
    ] = (
        data[
            "is_error"
        ]
        &
        (
            pd.to_numeric(
                data[
                    "prediction_confidence"
                ],
                errors="coerce",
            )
            >= 0.80
        )
    )

    # ========================================================================
    # OVERALL ERROR PROFILE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "OVERALL ERROR PROFILE"
    )
    print("=" * 78)

    numeric_profiles = [
        "damage_level",
        "absolute_delta_magnitude",
        "signed_delta_mean",
        "spatial_change_magnitude",
    ]

    error_profile_records = []

    for feature in numeric_profiles:

        if feature not in data.columns:
            continue

        correct_values = pd.to_numeric(
            data.loc[
                ~data["is_error"],
                feature,
            ],
            errors="coerce",
        ).dropna()

        wrong_values = pd.to_numeric(
            data.loc[
                data["is_error"],
                feature,
            ],
            errors="coerce",
        ).dropna()

        high_conf_values = pd.to_numeric(
            data.loc[
                data[
                    "is_high_confidence_error"
                ],
                feature,
            ],
            errors="coerce",
        ).dropna()

        record = {
            "feature":
                feature,

            "correct_count":
                len(correct_values),

            "wrong_count":
                len(wrong_values),

            "high_conf_wrong_count":
                len(high_conf_values),

            "correct_mean":
                (
                    float(
                        correct_values.mean()
                    )
                    if len(correct_values)
                    else None
                ),

            "wrong_mean":
                (
                    float(
                        wrong_values.mean()
                    )
                    if len(wrong_values)
                    else None
                ),

            "high_conf_wrong_mean":
                (
                    float(
                        high_conf_values.mean()
                    )
                    if len(high_conf_values)
                    else None
                ),

            "correct_median":
                (
                    float(
                        correct_values.median()
                    )
                    if len(correct_values)
                    else None
                ),

            "wrong_median":
                (
                    float(
                        wrong_values.median()
                    )
                    if len(wrong_values)
                    else None
                ),
        }

        error_profile_records.append(
            record
        )

        print()
        print(feature)

        print(
            "  Correct mean:",
            record[
                "correct_mean"
            ],
        )

        print(
            "  Wrong mean:",
            record[
                "wrong_mean"
            ],
        )

        print(
            "  High-confidence wrong mean:",
            record[
                "high_conf_wrong_mean"
            ],
        )

    error_profile_df = pd.DataFrame(
        error_profile_records
    )

    error_profile_df.to_csv(
        ERROR_PROFILE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # IMPORTANT CONFUSION PAIRS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "IMPORTANT CONFUSION PAIRS"
    )
    print("=" * 78)

    important_pairs = [
        (
            "PERSISTENT_GROWTH",
            "PERSISTENT_REDUCTION",
        ),
        (
            "PERSISTENT_REDUCTION",
            "PERSISTENT_GROWTH",
        ),
        (
            "PROBABLE_GROWTH",
            "PROBABLE_REDUCTION",
        ),
        (
            "PROBABLE_REDUCTION",
            "PROBABLE_GROWTH",
        ),
        (
            "SPATIAL_TRANSITION_INCREASE",
            "SPATIAL_TRANSITION_DECREASE",
        ),
        (
            "SPATIAL_TRANSITION_DECREASE",
            "SPATIAL_TRANSITION_INCREASE",
        ),
    ]

    pair_records = []

    for true_class, predicted_class in (
        important_pairs
    ):

        subset = data[
            (
                data[
                    "persistent_damage_target"
                ]
                == true_class
            )
            &
            (
                data[
                    "predicted_class"
                ]
                == predicted_class
            )
        ].copy()

        if len(subset) == 0:
            continue

        record = {
            "true_class":
                true_class,

            "predicted_class":
                predicted_class,

            "count":
                len(subset),
        }

        for feature in numeric_profiles:

            if feature not in subset.columns:
                continue

            values = pd.to_numeric(
                subset[
                    feature
                ],
                errors="coerce",
            ).dropna()

            record[
                feature + "_mean"
            ] = (
                float(
                    values.mean()
                )
                if len(values)
                else None
            )

            record[
                feature + "_median"
            ] = (
                float(
                    values.median()
                )
                if len(values)
                else None
            )

        pair_records.append(
            record
        )

        print()
        print(
            true_class,
            "->",
            predicted_class,
        )

        print(
            "  Count:",
            len(subset),
        )

        for feature in numeric_profiles:

            key = feature + "_mean"

            if key in record:

                print(
                    f"  {feature}:",
                    record[key],
                )

    pair_profile_df = pd.DataFrame(
        pair_records
    )

    pair_profile_df.to_csv(
        CLASS_PAIR_PROFILE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # TARGET-CLASS BOUNDARY ANALYSIS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET CLASS BOUNDARY ANALYSIS"
    )
    print("=" * 78)

    boundary_records = []

    for class_name, group in data.groupby(
        "persistent_damage_target"
    ):

        record = {
            "class":
                class_name,

            "support":
                len(group),

            "errors":
                int(
                    group[
                        "is_error"
                    ].sum()
                ),

            "error_rate":
                float(
                    group[
                        "is_error"
                    ].mean()
                ),
        }

        for feature in numeric_profiles:

            if feature not in group.columns:
                continue

            values = pd.to_numeric(
                group[
                    feature
                ],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            record[
                feature + "_mean"
            ] = float(
                values.mean()
            )

            record[
                feature + "_median"
            ] = float(
                values.median()
            )

            record[
                feature + "_p25"
            ] = float(
                values.quantile(
                    0.25
                )
            )

            record[
                feature + "_p75"
            ] = float(
                values.quantile(
                    0.75
                )
            )

        boundary_records.append(
            record
        )

    boundary_df = pd.DataFrame(
        boundary_records
    )

    boundary_df.to_csv(
        TARGET_BOUNDARY_OUTPUT,
        index=False,
    )

    print(
        boundary_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SEQUENCE ERROR PROFILE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "SEQUENCE ERROR PROFILE"
    )
    print("=" * 78)

    sequence_records = []

    for sequence, group in data.groupby(
        "sequence"
    ):

        n = len(group)

        errors = int(
            group[
                "is_error"
            ].sum()
        )

        high_conf_errors = int(
            group[
                "is_high_confidence_error"
            ].sum()
        )

        sequence_records.append(
            {
                "sequence":
                    sequence,

                "records":
                    n,

                "errors":
                    errors,

                "error_rate":
                    errors / n
                    if n > 0
                    else 0.0,

                "high_confidence_errors":
                    high_conf_errors,

                "high_confidence_error_rate":
                    high_conf_errors / n
                    if n > 0
                    else 0.0,

                "mean_confidence":
                    float(
                        pd.to_numeric(
                            group[
                                "prediction_confidence"
                            ],
                            errors="coerce",
                        ).mean()
                    ),
            }
        )

    sequence_df = pd.DataFrame(
        sequence_records
    )

    sequence_df = (
        sequence_df
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        sequence_df.to_string(
            index=False
        )
    )

    sequence_df.to_csv(
        SEQUENCE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # HIGH-CONFIDENCE ERRORS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "HIGH-CONFIDENCE ERROR PROFILE"
    )
    print("=" * 78)

    high_conf = data[
        data[
            "is_high_confidence_error"
        ]
    ].copy()

    print(
        "High-confidence errors:",
        len(high_conf),
    )

    if len(high_conf) > 0:

        high_conf_profile = (
            high_conf.groupby(
                [
                    "persistent_damage_target",
                    "predicted_class",
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
        )

        print(
            high_conf_profile.to_string(
                index=False
            )
        )

    high_conf.to_csv(
        HIGH_CONF_OUTPUT,
        index=False,
    )

    # ========================================================================
    # TEMPORAL NEIGHBOR CONTEXT
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TEMPORAL NEIGHBOR CONTEXT"
    )
    print("=" * 78)

    data = data.sort_values(
        [
            "sequence",
            "current_frame",
        ]
    ).reset_index(
        drop=True
    )

    data[
        "previous_target"
    ] = (
        data
        .groupby(
            "sequence"
        )[
            "persistent_damage_target"
        ]
        .shift(1)
    )

    data[
        "next_target"
    ] = (
        data
        .groupby(
            "sequence"
        )[
            "persistent_damage_target"
        ]
        .shift(-1)
    )

    error_with_neighbors = data[
        data[
            "is_error"
        ]
    ].copy()

    if len(error_with_neighbors) > 0:

        changed_neighbors = (
            error_with_neighbors[
                "previous_target"
            ]
            !=
            error_with_neighbors[
                "next_target"
            ]
        )

        changed_neighbors = (
            changed_neighbors
            .fillna(False)
        )

        neighbor_change_rate = float(
            changed_neighbors.mean()
        )

        print(
            "Error records whose surrounding "
            "target states differ:",
            f"{neighbor_change_rate:.6f}",
        )

    # ========================================================================
    # GROWTH / REDUCTION CONFUSION
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "GROWTH / REDUCTION CONFUSION DIAGNOSTIC"
    )
    print("=" * 78)

    growth_reduction_pairs = [
        (
            "PERSISTENT_GROWTH",
            "PERSISTENT_REDUCTION",
        ),
        (
            "PERSISTENT_REDUCTION",
            "PERSISTENT_GROWTH",
        ),
        (
            "PROBABLE_GROWTH",
            "PROBABLE_REDUCTION",
        ),
        (
            "PROBABLE_REDUCTION",
            "PROBABLE_GROWTH",
        ),
    ]

    growth_reduction_confusion = 0

    for true_class, predicted_class in (
        growth_reduction_pairs
    ):

        count = int(
            (
                (
                    data[
                        "persistent_damage_target"
                    ]
                    == true_class
                )
                &
                (
                    data[
                        "predicted_class"
                    ]
                    == predicted_class
                )
            ).sum()
        )

        growth_reduction_confusion += count

        print(
            true_class,
            "->",
            predicted_class,
            ":",
            count,
        )

    total_errors = int(
        data[
            "is_error"
        ].sum()
    )

    confusion_fraction = (
        growth_reduction_confusion
        / total_errors
        if total_errors > 0
        else 0.0
    )

    print()
    print(
        "Total errors:",
        total_errors,
    )

    print(
        "Growth/reduction confusion:",
        growth_reduction_confusion,
    )

    print(
        "Fraction of all errors:",
        f"{confusion_fraction:.6f}",
    )

    if confusion_fraction >= 0.50:

        diagnostic_status = (
            "HIGH_GROWTH_REDUCTION_AMBIGUITY"
        )

    elif confusion_fraction >= 0.30:

        diagnostic_status = (
            "MODERATE_GROWTH_REDUCTION_AMBIGUITY"
        )

    else:

        diagnostic_status = (
            "NO_DOMINANT_GROWTH_REDUCTION_AMBIGUITY"
        )

    print(
        "Diagnostic:",
        diagnostic_status,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.11",

        "title":
            "Error-Driven Target Validation",

        "prediction_records":
            len(predictions),

        "target_records":
            len(targets),

        "spatial_records":
            len(spatial),

        "merged_records":
            len(data),

        "target_mismatch_count":
            target_mismatch_count,

        "total_errors":
            total_errors,

        "growth_reduction_confusion":
            growth_reduction_confusion,

        "growth_reduction_error_fraction":
            confusion_fraction,

        "diagnostic_status":
            diagnostic_status,

        "high_confidence_errors":
            int(
                data[
                    "is_high_confidence_error"
                ].sum()
            ),

        "high_confidence_threshold":
            0.80,

        "sequence_count":
            int(
                data[
                    "sequence"
                ].nunique()
            ),

        "outputs": {

            "error_profile":
                str(
                    ERROR_PROFILE_OUTPUT
                ),

            "class_pair_profiles":
                str(
                    CLASS_PAIR_PROFILE_OUTPUT
                ),

            "target_boundary":
                str(
                    TARGET_BOUNDARY_OUTPUT
                ),

            "sequence_profile":
                str(
                    SEQUENCE_OUTPUT
                ),

            "high_confidence_errors":
                str(
                    HIGH_CONF_OUTPUT
                ),
        },
    }

    summary = to_json_safe(
        summary
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
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
        "STAGE 3.11 COMPLETE"
    )
    print("=" * 78)

    print(
        "Diagnostic:",
        diagnostic_status,
    )

    print(
        "Growth/reduction confusion:",
        growth_reduction_confusion,
    )

    print(
        "Fraction of test errors:",
        f"{confusion_fraction:.6f}",
    )

    print()
    print(
        "Outputs saved to:"
    )

    print(
        AUDIT_DIR
    )

    print()
    print(
        "FINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()