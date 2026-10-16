import json
import math
import re
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
# INPUTS
# ============================================================================

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_6_persistent_damage_targets.csv"
)

PREDICTION_FILE = (
    AUDIT_DIR /
    "stage_3_10_test_predictions.csv"
)

FEATURE_FILE = (
    AUDIT_DIR /
    "stage_1_0_damage_features.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_target_stability_summary.json"
)

FRAME_STABILITY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_frame_stability.csv"
)

CLASS_STABILITY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_class_stability.csv"
)

TRANSITION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_target_transitions.csv"
)

AMBIGUOUS_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_ambiguous_cases.csv"
)

STREAK_OUTPUT = (
    AUDIT_DIR /
    "stage_3_12_target_streaks.csv"
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
# FILE HELPERS
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
            "Missing required column: "
            + str(candidates)
        )

    return None


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
# TARGET KEY STANDARDIZATION
# ============================================================================

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


# ============================================================================
# PREDICTION KEY STANDARDIZATION
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


# ============================================================================
# STAGE 1.0 FRAME EXTRACTION
# ============================================================================

def extract_frame_number(value):

    if pd.isna(value):
        return np.nan

    text = str(value)

    # Examples:
    # 639.jpg
    # 639.json
    # ...\639.jpg
    # .../639.jpg

    matches = re.findall(
        r"(?<!\d)(\d+)(?=\.(?:jpg|jpeg|png|json)$)",
        text.lower(),
    )

    if matches:

        return float(
            matches[-1]
        )

    # Fallback: final numeric token
    matches = re.findall(
        r"(\d+)",
        text,
    )

    if matches:

        return float(
            matches[-1]
        )

    return np.nan


def prepare_stage1_features(df):

    df = df.copy()

    print()
    print("=" * 78)
    print(
        "STAGE 1.0 SCHEMA"
    )
    print("=" * 78)

    print(
        "Columns:"
    )

    print(
        ", ".join(
            df.columns
        )
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

    print()
    print(
        "Frame source column:",
        filename_col,
    )

    valid_frames = int(
        df[
            "current_frame"
        ].notna().sum()
    )

    print(
        "Frame numbers extracted:",
        valid_frames,
        "/",
        len(df),
    )

    # Keep only physical damage features that actually exist.
    desired = [
        "crack_count",
        "total_crack_area_px",
        "crack_area_ratio",
        "largest_crack_area_px",
        "largest_crack_area_ratio",
        "mean_crack_area_px",
        "median_crack_area_px",
        "total_crack_perimeter_px",
        "mean_crack_perimeter_px",
        "max_crack_perimeter_px",
    ]

    available = [
        column
        for column in desired
        if column in df.columns
    ]

    print()
    print(
        "Physical features available:"
    )

    for column in available:

        print(
            " ",
            column,
        )

    return df[
        [
            "sequence",
            "component",
            "current_frame",
        ]
        + available
    ].copy()


# ============================================================================
# TARGET DIRECTION
# ============================================================================

def target_direction(label):

    if not isinstance(
        label,
        str,
    ):

        return "UNKNOWN"

    label = label.upper()

    if "GROWTH" in label:
        return "GROWTH"

    if "REDUCTION" in label:
        return "REDUCTION"

    if "STABLE" in label:
        return "STABLE"

    if "TRANSITION" in label:

        if "INCREASE" in label:
            return "GROWTH"

        if "DECREASE" in label:
            return "REDUCTION"

        return "TRANSITION"

    return "UNKNOWN"


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.12 - TARGET STABILITY AUDIT"
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

    target_file = resolve_file(
        TARGET_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_3_6",
            "target",
        ],
    )

    prediction_file = resolve_file(
        PREDICTION_FILE,
        [
            "sequence",
            "component",
            "predicted_class",
        ],
        [
            "stage_3_10",
            "prediction",
        ],
    )

    feature_file = resolve_file(
        FEATURE_FILE,
        [
            "sequence",
            "component",
            "crack_area_ratio",
        ],
        [
            "stage_1_0",
            "damage",
        ],
    )

    if target_file is None:

        raise FileNotFoundError(
            "Stage 3.6 target file not found."
        )

    if feature_file is None:

        raise FileNotFoundError(
            "Stage 1.0 damage feature file not found."
        )

    print(
        "targets         : FOUND"
    )

    print(
        "   ",
        target_file,
    )

    if prediction_file is not None:

        print(
            "predictions     : FOUND"
        )

        print(
            "   ",
            prediction_file,
        )

    else:

        print(
            "predictions     : NOT FOUND"
        )

    print(
        "damage_features : FOUND"
    )

    print(
        "   ",
        feature_file,
    )

    # ========================================================================
    # LOAD
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "LOADING DATA"
    )
    print("=" * 78)

    targets = pd.read_csv(
        target_file
    )

    features = pd.read_csv(
        feature_file
    )

    predictions = None

    if prediction_file is not None:

        predictions = pd.read_csv(
            prediction_file
        )

    print(
        "Targets:",
        len(targets),
    )

    print(
        "Damage features:",
        len(features),
    )

    if predictions is not None:

        print(
            "Test predictions:",
            len(predictions),
        )

    # ========================================================================
    # STANDARDIZE TARGETS ONLY
    # ========================================================================

    targets = standardize_target_keys(
        targets
    )

    # IMPORTANT:
    # Stage 1.0 does NOT contain current_frame.
    # We extract it from the canonical image filename.

    features = prepare_stage1_features(
        features
    )

    if predictions is not None:

        predictions = (
            standardize_prediction_keys(
                predictions
            )
        )

    # ========================================================================
    # DUPLICATE CHECK
    # ========================================================================

    keys = [
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

    target_dup = int(
        targets[
            keys
        ].duplicated()
        .sum()
    )

    feature_dup = int(
        features[
            keys
        ].duplicated()
        .sum()
    )

    print(
        "Target duplicate keys:",
        target_dup,
    )

    print(
        "Feature duplicate keys:",
        feature_dup,
    )

    if target_dup > 0:

        raise RuntimeError(
            "Duplicate target keys detected."
        )

    if feature_dup > 0:

        # Keep first only if duplicate rows are
        # exactly identical in physical features.

        duplicate_groups = (
            features[
                features[
                    keys
                ].duplicated(
                    keep=False
                )
            ]
            .sort_values(keys)
        )

        non_identical = 0

        feature_columns = [
            column
            for column in features.columns
            if column not in keys
        ]

        for _, group in duplicate_groups.groupby(
            keys
        ):

            if len(
                group[
                    feature_columns
                ]
                .drop_duplicates()
            ) > 1:

                non_identical += 1

        print(
            "Non-identical duplicate feature groups:",
            non_identical,
        )

        if non_identical > 0:

            raise RuntimeError(
                "Stage 1.0 contains non-identical "
                "duplicate frame keys."
            )

        features = (
            features
            .drop_duplicates(
                subset=keys
            )
            .copy()
        )

        print(
            "Identical duplicate feature rows removed."
        )

    # ========================================================================
    # TARGET COLUMN
    # ========================================================================

    target_column = find_column(
        targets,
        [
            "persistent_damage_target",
            "target_state",
        ],
    )

    print()
    print(
        "Target column:",
        target_column,
    )

    # ========================================================================
    # MERGE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "BUILDING SEQUENTIAL TARGET TABLE"
    )
    print("=" * 78)

    physical_columns = [
        column
        for column in [
            "crack_count",
            "total_crack_area_px",
            "crack_area_ratio",
            "largest_crack_area_px",
            "largest_crack_area_ratio",
            "mean_crack_area_px",
            "median_crack_area_px",
            "total_crack_perimeter_px",
            "mean_crack_perimeter_px",
            "max_crack_perimeter_px",
        ]
        if column in features.columns
    ]

    data = targets[
        keys
        + [
            target_column
        ]
    ].copy()

    data = data.merge(
        features[
            keys
            + physical_columns
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    print(
        "Merged target/feature records:",
        len(data),
    )

    print(
        "Expected target records:",
        len(targets),
    )

    # ========================================================================
    # PHYSICAL FEATURE COVERAGE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "PHYSICAL FEATURE COVERAGE"
    )
    print("=" * 78)

    for column in physical_columns:

        missing = int(
            data[
                column
            ].isna().sum()
        )

        print(
            f"{column:<35}",
            "missing:",
            missing,
        )

    # ========================================================================
    # SORT
    # ========================================================================

    data = (
        data
        .sort_values(
            [
                "sequence",
                "current_frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================================
    # TEMPORAL NEIGHBORS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "BUILDING TEMPORAL TARGET CONTEXT"
    )
    print("=" * 78)

    data[
        "previous_target"
    ] = (
        data
        .groupby(
            "sequence"
        )[
            target_column
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
            target_column
        ]
        .shift(-1)
    )

    data[
        "previous_frame"
    ] = (
        data
        .groupby(
            "sequence"
        )[
            "current_frame"
        ]
        .shift(1)
    )

    data[
        "next_frame"
    ] = (
        data
        .groupby(
            "sequence"
        )[
            "current_frame"
        ]
        .shift(-1)
    )

    data[
        "previous_frame_gap"
    ] = (
        data[
            "current_frame"
        ]
        -
        data[
            "previous_frame"
        ]
    )

    data[
        "next_frame_gap"
    ] = (
        data[
            "next_frame"
        ]
        -
        data[
            "current_frame"
        ]
    )

    # ========================================================================
    # DAMAGE CHANGES
    # ========================================================================

    if "crack_area_ratio" in data.columns:

        data[
            "area_delta_previous"
        ] = (
            data[
                "crack_area_ratio"
            ]
            -
            data
            .groupby(
                "sequence"
            )[
                "crack_area_ratio"
            ]
            .shift(1)
        )

        data[
            "area_delta_next"
        ] = (
            data
            .groupby(
                "sequence"
            )[
                "crack_area_ratio"
            ]
            .shift(-1)
            -
            data[
                "crack_area_ratio"
            ]
        )

        data[
            "area_direction_previous"
        ] = np.sign(
            data[
                "area_delta_previous"
            ]
        )

        data[
            "area_direction_next"
        ] = np.sign(
            data[
                "area_delta_next"
            ]
        )

        data[
            "area_direction_agreement"
        ] = (
            data[
                "area_direction_previous"
            ]
            ==
            data[
                "area_direction_next"
            ]
        )

        data[
            "area_direction_conflict"
        ] = (
            (
                data[
                    "area_direction_previous"
                ]
                !=
                data[
                    "area_direction_next"
                ]
            )
            &
            data[
                "area_direction_previous"
            ].notna()
            &
            data[
                "area_direction_next"
            ].notna()
        )

    # ========================================================================
    # TARGET DIRECTIONS
    # ========================================================================

    data[
        "target_direction"
    ] = data[
        target_column
    ].map(
        target_direction
    )

    data[
        "previous_direction"
    ] = data[
        "previous_target"
    ].map(
        target_direction
    )

    data[
        "next_direction"
    ] = data[
        "next_target"
    ].map(
        target_direction
    )

    data[
        "neighbor_direction_agreement"
    ] = (
        (
            data[
                "previous_direction"
            ]
            ==
            data[
                "target_direction"
            ]
        )
        &
        (
            data[
                "next_direction"
            ]
            ==
            data[
                "target_direction"
            ]
        )
    )

    data[
        "neighbor_direction_conflict"
    ] = (
        (
            data[
                "previous_direction"
            ]
            !=
            data[
                "next_direction"
            ]
        )
        &
        data[
            "previous_direction"
        ].notna()
        &
        data[
            "next_direction"
        ].notna()
    )

    # ========================================================================
    # TARGET TRANSITIONS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET TRANSITION ANALYSIS"
    )
    print("=" * 78)

    valid_previous = data[
        data[
            "previous_target"
        ].notna()
    ].copy()

    transition_table = (
        valid_previous
        .groupby(
            [
                "previous_target",
                target_column,
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
        transition_table.to_string(
            index=False
        )
    )

    transition_table.to_csv(
        TRANSITION_OUTPUT,
        index=False,
    )

    # ========================================================================
    # CLASS STABILITY
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CLASS STABILITY"
    )
    print("=" * 78)

    class_records = []

    for class_name, group in data.groupby(
        target_column
    ):

        valid = group[
            group[
                "previous_direction"
            ].notna()
            &
            group[
                "next_direction"
            ].notna()
        ]

        if len(valid) > 0:

            agreement = float(
                valid[
                    "neighbor_direction_agreement"
                ].mean()
            )

            conflict = float(
                valid[
                    "neighbor_direction_conflict"
                ].mean()
            )

        else:

            agreement = 0.0
            conflict = 0.0

        class_records.append(
            {
                "class":
                    class_name,

                "records":
                    len(group),

                "valid_neighbor_records":
                    len(valid),

                "neighbor_direction_agreement":
                    agreement,

                "neighbor_direction_conflict":
                    conflict,

                "error_boundary_fraction":
                    conflict,
            }
        )

    class_df = pd.DataFrame(
        class_records
    )

    print(
        class_df.to_string(
            index=False
        )
    )

    class_df.to_csv(
        CLASS_STABILITY_OUTPUT,
        index=False,
    )

    # ========================================================================
    # STREAK ANALYSIS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET STREAK ANALYSIS"
    )
    print("=" * 78)

    streak_records = []

    for sequence, group in data.groupby(
        "sequence"
    ):

        group = group.sort_values(
            "current_frame"
        )

        labels = (
            group[
                target_column
            ]
            .astype(str)
            .tolist()
        )

        frames = (
            group[
                "current_frame"
            ]
            .astype(int)
            .tolist()
        )

        if not labels:
            continue

        start = 0

        while start < len(labels):

            label = labels[start]

            end = start

            while (
                end + 1 < len(labels)
                and labels[
                    end + 1
                ]
                == label
            ):

                end += 1

            streak_records.append(
                {
                    "sequence":
                        sequence,

                    "class":
                        label,

                    "start_frame":
                        frames[start],

                    "end_frame":
                        frames[end],

                    "length":
                        end - start + 1,
                }
            )

            start = end + 1

    streak_df = pd.DataFrame(
        streak_records
    )

    streak_df.to_csv(
        STREAK_OUTPUT,
        index=False,
    )

    if len(streak_df) > 0:

        streak_summary = (
            streak_df
            .groupby(
                "class"
            )[
                "length"
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "max",
                ]
            )
            .reset_index()
        )

        print(
            streak_summary.to_string(
                index=False
            )
        )

    # ========================================================================
    # AMBIGUOUS CASES
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "AMBIGUOUS GROWTH/REDUCTION CASES"
    )
    print("=" * 78)

    ambiguous = data[
        data[
            "target_direction"
        ].isin(
            [
                "GROWTH",
                "REDUCTION",
            ]
        )
        &
        (
            (
                data[
                    "neighbor_direction_conflict"
                ]
                == True
            )
            |
            (
                data[
                    "neighbor_direction_agreement"
                ]
                == False
            )
        )
    ].copy()

    print(
        "Ambiguous records:",
        len(ambiguous),
    )

    ambiguous_columns = [
        "sequence",
        "component",
        "current_frame",
        target_column,
        "previous_target",
        "next_target",
        "previous_direction",
        "target_direction",
        "next_direction",
    ]

    optional_columns = [
        "crack_area_ratio",
        "area_delta_previous",
        "area_delta_next",
        "area_direction_agreement",
        "area_direction_conflict",
    ]

    ambiguous_columns.extend(
        [
            column
            for column in optional_columns
            if column in ambiguous.columns
        ]
    )

    ambiguous[
        ambiguous_columns
    ].to_csv(
        AMBIGUOUS_OUTPUT,
        index=False,
    )

    if len(ambiguous) > 0:

        print(
            ambiguous[
                ambiguous_columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ========================================================================
    # TEST SET STABILITY
    # ========================================================================

    test_stability = None

    if predictions is not None:

        prediction_keys = predictions[
            keys
        ].copy()

        prediction_keys[
            "_test_record"
        ] = True

        test_data = data.merge(
            prediction_keys,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        test_ambiguous = test_data[
            (
                test_data[
                    "target_direction"
                ].isin(
                    [
                        "GROWTH",
                        "REDUCTION",
                    ]
                )
            )
            &
            (
                test_data[
                    "neighbor_direction_conflict"
                ]
                == True
            )
        ]

        test_stability = {

            "test_records":
                len(test_data),

            "ambiguous_growth_reduction_records":
                len(test_ambiguous),

            "ambiguous_fraction":
                (
                    len(test_ambiguous)
                    /
                    len(test_data)
                    if len(test_data)
                    else 0.0
                ),
        }

        print()
        print("=" * 78)
        print(
            "TEST TARGET STABILITY"
        )
        print("=" * 78)

        print(
            "Test records:",
            test_stability[
                "test_records"
            ],
        )

        print(
            "Ambiguous growth/reduction:",
            test_stability[
                "ambiguous_growth_reduction_records"
            ],
        )

        print(
            "Ambiguous fraction:",
            f"{test_stability['ambiguous_fraction']:.6f}",
        )

    # ========================================================================
    # DIAGNOSTIC
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET STABILITY DIAGNOSTIC"
    )
    print("=" * 78)

    growth_reduction = data[
        data[
            "target_direction"
        ].isin(
            [
                "GROWTH",
                "REDUCTION",
            ]
        )
    ].copy()

    valid_growth_reduction = (
        growth_reduction[
            growth_reduction[
                "previous_direction"
            ].notna()
            &
            growth_reduction[
                "next_direction"
            ].notna()
        ]
    )

    if len(
        valid_growth_reduction
    ) > 0:

        direction_conflict = float(
            valid_growth_reduction[
                "neighbor_direction_conflict"
            ].mean()
        )

        direction_agreement = float(
            valid_growth_reduction[
                "neighbor_direction_agreement"
            ].mean()
        )

    else:

        direction_conflict = 0.0
        direction_agreement = 0.0

    if direction_conflict >= 0.30:

        diagnostic = (
            "HIGH_TARGET_DIRECTION_INSTABILITY"
        )

    elif direction_conflict >= 0.15:

        diagnostic = (
            "MODERATE_TARGET_DIRECTION_INSTABILITY"
        )

    else:

        diagnostic = (
            "TARGET_DIRECTION_MOSTLY_STABLE"
        )

    print(
        "Growth/reduction records:",
        len(growth_reduction),
    )

    print(
        "Valid neighbor records:",
        len(valid_growth_reduction),
    )

    print(
        "Direction agreement:",
        f"{direction_agreement:.6f}",
    )

    print(
        "Direction conflict:",
        f"{direction_conflict:.6f}",
    )

    print(
        "Diagnostic:",
        diagnostic,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.12",

        "title":
            "Target Stability Audit",

        "target_records":
            len(targets),

        "merged_records":
            len(data),

        "sequence_count":
            int(
                data[
                    "sequence"
                ].nunique()
            ),

        "growth_reduction_records":
            len(growth_reduction),

        "valid_growth_reduction_neighbor_records":
            len(valid_growth_reduction),

        "direction_agreement":
            direction_agreement,

        "direction_conflict":
            direction_conflict,

        "diagnostic":
            diagnostic,

        "test_stability":
            test_stability,

        "outputs": {

            "frame_stability":
                str(
                    FRAME_STABILITY_OUTPUT
                ),

            "class_stability":
                str(
                    CLASS_STABILITY_OUTPUT
                ),

            "target_transitions":
                str(
                    TRANSITION_OUTPUT
                ),

            "ambiguous_cases":
                str(
                    AMBIGUOUS_OUTPUT
                ),

            "target_streaks":
                str(
                    STREAK_OUTPUT
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
        "STAGE 3.12 COMPLETE"
    )
    print("=" * 78)

    print(
        "Diagnostic:",
        diagnostic,
    )

    print(
        "Direction conflict:",
        f"{direction_conflict:.6f}",
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