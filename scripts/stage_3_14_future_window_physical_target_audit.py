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

FEATURE_FILE = (
    AUDIT_DIR /
    "stage_1_0_damage_features.csv"
)

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_6_persistent_damage_targets.csv"
)

PREDICTION_FILE = (
    AUDIT_DIR /
    "stage_3_10_test_predictions.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

FRAME_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_future_window_targets.csv"
)

HORIZON_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_horizon_comparison.csv"
)

DISTRIBUTION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_target_distribution.csv"
)

STABILITY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_direction_stability.csv"
)

PERSISTENCE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_direction_persistence.csv"
)

ALIGNMENT_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_existing_target_alignment.csv"
)

TEST_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_test_coverage.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_14_future_window_target_summary.json"
)


# ============================================================================
# CONSTANTS
# ============================================================================

HORIZONS = [
    1,
    3,
    5,
    7,
]

STEP_STABLE_EPSILON = 0.00005


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
# COLUMN HELPER
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


# ============================================================================
# FRAME NUMBER EXTRACTION
# ============================================================================

def extract_frame_number(value):

    if pd.isna(value):
        return np.nan

    text = str(value)

    matches = re.findall(
        r"(?<!\d)(\d+)(?=\.(?:jpg|jpeg|png|json)$)",
        text.lower(),
    )

    if matches:

        return float(
            matches[-1]
        )

    matches = re.findall(
        r"(\d+)",
        text,
    )

    if matches:

        return float(
            matches[-1]
        )

    return np.nan


# ============================================================================
# DIRECTION HELPERS
# ============================================================================

def direction_from_delta(
    delta,
    epsilon=STEP_STABLE_EPSILON,
):

    if pd.isna(delta):
        return "UNKNOWN"

    if delta > epsilon:
        return "GROWTH"

    if delta < -epsilon:
        return "REDUCTION"

    return "STABLE"


def direction_from_sign(
    delta,
):

    if pd.isna(delta):
        return "UNKNOWN"

    if delta > 0:
        return "GROWTH"

    if delta < 0:
        return "REDUCTION"

    return "STABLE"


def existing_target_direction(
    label,
):

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

    return "UNKNOWN"


def safe_fraction(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator
        /
        denominator
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_features():

    print()
    print("=" * 78)
    print(
        "PREPARING STAGE 1.0 PHYSICAL DAMAGE FEATURES"
    )
    print("=" * 78)

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Raw feature records:",
        len(features),
    )

    sequence_col = find_column(
        features,
        ["sequence"],
    )

    component_col = find_column(
        features,
        ["component"],
    )

    filename_col = find_column(
        features,
        [
            "canonical_image_filename",
            "original_image_filename",
            "source_image",
        ],
    )

    ratio_col = find_column(
        features,
        [
            "crack_area_ratio",
        ],
    )

    total_area_col = find_column(
        features,
        [
            "total_crack_area_px",
        ],
        required=False,
    )

    crack_count_col = find_column(
        features,
        [
            "crack_count",
        ],
        required=False,
    )

    features["sequence"] = (
        features[
            sequence_col
        ]
        .astype(str)
    )

    features["component"] = pd.to_numeric(
        features[
            component_col
        ],
        errors="coerce",
    )

    features["current_frame"] = (
        features[
            filename_col
        ]
        .apply(
            extract_frame_number
        )
    )

    features[
        "crack_area_ratio"
    ] = pd.to_numeric(
        features[
            ratio_col
        ],
        errors="coerce",
    )

    if total_area_col is not None:

        features[
            "total_crack_area_px"
        ] = pd.to_numeric(
            features[
                total_area_col
            ],
            errors="coerce",
        )

    else:

        features[
            "total_crack_area_px"
        ] = np.nan

    if crack_count_col is not None:

        features[
            "crack_count"
        ] = pd.to_numeric(
            features[
                crack_count_col
            ],
            errors="coerce",
        )

    else:

        features[
            "crack_count"
        ] = np.nan

    keys = [
        "sequence",
        "component",
        "current_frame",
    ]

    duplicate_count = int(
        features[
            keys
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate frame keys:",
        duplicate_count,
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Stage 1.0 contains duplicate "
            "sequence/component/frame keys."
        )

    missing_frame = int(
        features[
            "current_frame"
        ]
        .isna()
        .sum()
    )

    missing_ratio = int(
        features[
            "crack_area_ratio"
        ].isna()
        .sum()
    )

    print(
        "Missing frame numbers:",
        missing_frame,
    )

    print(
        "Missing crack_area_ratio:",
        missing_ratio,
    )

    if missing_frame > 0:

        raise RuntimeError(
            "Could not recover all frame numbers."
        )

    if missing_ratio > 0:

        raise RuntimeError(
            "crack_area_ratio contains missing values."
        )

    features = (
        features[
            [
                "sequence",
                "component",
                "current_frame",
                "crack_area_ratio",
                "total_crack_area_px",
                "crack_count",
            ]
        ]
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

    print(
        "Prepared records:",
        len(features),
    )

    print(
        "Sequences:",
        features[
            "sequence"
        ].nunique(),
    )

    return features


# ============================================================================
# FUTURE TARGETS
# ============================================================================

def build_future_targets(
    features,
):

    print()
    print("=" * 78)
    print(
        "BUILDING FUTURE-WINDOW PHYSICAL TARGETS"
    )
    print("=" * 78)

    data = features.copy()

    for horizon in HORIZONS:

        data[
            f"future_ratio_t{horizon}"
        ] = (
            data
            .groupby(
                "sequence"
            )[
                "crack_area_ratio"
            ]
            .shift(
                -horizon
            )
        )

        data[
            f"future_total_area_t{horizon}"
        ] = (
            data
            .groupby(
                "sequence"
            )[
                "total_crack_area_px"
            ]
            .shift(
                -horizon
            )
        )

        data[
            f"future_crack_count_t{horizon}"
        ] = (
            data
            .groupby(
                "sequence"
            )[
                "crack_count"
            ]
            .shift(
                -horizon
            )
        )

        data[
            f"delta_ratio_t{horizon}"
        ] = (
            data[
                f"future_ratio_t{horizon}"
            ]
            -
            data[
                "crack_area_ratio"
            ]
        )

        data[
            f"abs_delta_ratio_t{horizon}"
        ] = (
            data[
                f"delta_ratio_t{horizon}"
            ]
            .abs()
        )

        data[
            f"direction_t{horizon}"
        ] = data[
            f"delta_ratio_t{horizon}"
        ].apply(
            direction_from_sign
        )

        data[
            f"direction_stable_t{horizon}"
        ] = data[
            f"delta_ratio_t{horizon}"
        ].apply(
            direction_from_delta
        )

        data[
            f"relative_change_t{horizon}"
        ] = np.where(
            data[
                "crack_area_ratio"
            ].abs()
            > 1e-12,

            data[
                f"delta_ratio_t{horizon}"
            ]
            /
            data[
                "crack_area_ratio"
            ].abs(),

            np.nan,
        )

    return data


# ============================================================================
# FUTURE PATH ANALYSIS
# ============================================================================

def add_path_direction_features(
    data,
):

    print()
    print("=" * 78)
    print(
        "ANALYZING FUTURE DAMAGE DIRECTIONAL CONSISTENCY"
    )
    print("=" * 78)

    all_results = []

    for sequence, group in data.groupby(
        "sequence"
    ):

        group = (
            group
            .sort_values(
                "current_frame"
            )
            .reset_index(
                drop=True
            )
        )

        ratios = (
            group[
                "crack_area_ratio"
            ]
            .to_numpy()
        )

        frames = (
            group[
                "current_frame"
            ]
            .to_numpy()
        )

        for index in range(
            len(group)
        ):

            row = {
                "sequence":
                    sequence,

                "current_frame":
                    frames[index],
            }

            for horizon in HORIZONS:

                future_index = (
                    index
                    +
                    horizon
                )

                if future_index >= len(
                    group
                ):

                    row[
                        f"path_growth_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_reduction_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_stable_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_valid_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_direction_t{horizon}"
                    ] = "UNKNOWN"

                    row[
                        f"path_consistency_t{horizon}"
                    ] = np.nan

                    continue

                window_ratios = ratios[
                    index:
                    future_index + 1
                ]

                if len(
                    window_ratios
                ) < 2:

                    row[
                        f"path_growth_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_reduction_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_stable_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_valid_steps_t{horizon}"
                    ] = np.nan

                    row[
                        f"path_direction_t{horizon}"
                    ] = "UNKNOWN"

                    row[
                        f"path_consistency_t{horizon}"
                    ] = np.nan

                    continue

                deltas = np.diff(
                    window_ratios
                )

                growth = int(
                    np.sum(
                        deltas
                        >
                        STEP_STABLE_EPSILON
                    )
                )

                reduction = int(
                    np.sum(
                        deltas
                        <
                        -STEP_STABLE_EPSILON
                    )
                )

                stable = int(
                    len(deltas)
                    -
                    growth
                    -
                    reduction
                )

                valid_steps = (
                    growth
                    +
                    reduction
                )

                if valid_steps == 0:

                    path_direction = (
                        "STABLE"
                    )

                    consistency = 1.0

                elif growth > reduction:

                    path_direction = (
                        "GROWTH"
                    )

                    consistency = (
                        growth
                        /
                        valid_steps
                    )

                elif reduction > growth:

                    path_direction = (
                        "REDUCTION"
                    )

                    consistency = (
                        reduction
                        /
                        valid_steps
                    )

                else:

                    path_direction = (
                        "MIXED"
                    )

                    consistency = (
                        max(
                            growth,
                            reduction,
                        )
                        /
                        valid_steps
                    )

                row[
                    f"path_growth_steps_t{horizon}"
                ] = growth

                row[
                    f"path_reduction_steps_t{horizon}"
                ] = reduction

                row[
                    f"path_stable_steps_t{horizon}"
                ] = stable

                row[
                    f"path_valid_steps_t{horizon}"
                ] = valid_steps

                row[
                    f"path_direction_t{horizon}"
                ] = path_direction

                row[
                    f"path_consistency_t{horizon}"
                ] = consistency

            all_results.append(
                row
            )

    path_df = pd.DataFrame(
        all_results
    )

    print(
        "Path records:",
        len(path_df),
    )

    data = data.merge(
        path_df,
        on=[
            "sequence",
            "current_frame",
        ],
        how="left",
        validate="one_to_one",
    )

    return data


# ============================================================================
# HORIZON SUMMARY
# ============================================================================

def build_horizon_summary(
    data,
):

    print()
    print("=" * 78)
    print(
        "HORIZON COMPARISON"
    )
    print("=" * 78)

    records = []

    for horizon in HORIZONS:

        delta_column = (
            f"delta_ratio_t{horizon}"
        )

        direction_column = (
            f"direction_stable_t{horizon}"
        )

        path_direction_column = (
            f"path_direction_t{horizon}"
        )

        path_consistency_column = (
            f"path_consistency_t{horizon}"
        )

        valid = data[
            data[
                delta_column
            ].notna()
        ].copy()

        count = len(valid)

        growth_count = int(
            (
                valid[
                    direction_column
                ]
                ==
                "GROWTH"
            ).sum()
        )

        reduction_count = int(
            (
                valid[
                    direction_column
                ]
                ==
                "REDUCTION"
            ).sum()
        )

        stable_count = int(
            (
                valid[
                    direction_column
                ]
                ==
                "STABLE"
            ).sum()
        )

        abs_delta = valid[
            delta_column
        ].abs()

        path_valid = valid[
            valid[
                path_direction_column
            ].isin(
                [
                    "GROWTH",
                    "REDUCTION",
                    "STABLE",
                    "MIXED",
                ]
            )
        ]

        mixed_count = int(
            (
                path_valid[
                    path_direction_column
                ]
                ==
                "MIXED"
            ).sum()
        )

        consistent_path = (
            path_valid[
                path_consistency_column
            ]
            >= 0.75
        )

        records.append(
            {
                "horizon":
                    horizon,

                "valid_records":
                    count,

                "coverage":
                    safe_fraction(
                        count,
                        len(data),
                    ),

                "growth_count":
                    growth_count,

                "growth_fraction":
                    safe_fraction(
                        growth_count,
                        count,
                    ),

                "reduction_count":
                    reduction_count,

                "reduction_fraction":
                    safe_fraction(
                        reduction_count,
                        count,
                    ),

                "stable_count":
                    stable_count,

                "stable_fraction":
                    safe_fraction(
                        stable_count,
                        count,
                    ),

                "mean_abs_damage_change":
                    float(
                        abs_delta.mean()
                    ),

                "median_abs_damage_change":
                    float(
                        abs_delta.median()
                    ),

                "p90_abs_damage_change":
                    float(
                        abs_delta.quantile(
                            0.90
                        )
                    ),

                "p95_abs_damage_change":
                    float(
                        abs_delta.quantile(
                            0.95
                        )
                    ),

                "p99_abs_damage_change":
                    float(
                        abs_delta.quantile(
                            0.99
                        )
                    ),

                "path_mixed_fraction":
                    safe_fraction(
                        mixed_count,
                        len(path_valid),
                    ),

                "path_consistency_ge_75_fraction":
                    safe_fraction(
                        int(
                            consistent_path.sum()
                        ),
                        len(path_valid),
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# DIRECTION STABILITY
# ============================================================================

def build_direction_stability(
    data,
):

    print()
    print("=" * 78)
    print(
        "DIRECTION STABILITY"
    )
    print("=" * 78)

    records = []

    for horizon in HORIZONS:

        direction_column = (
            f"direction_stable_t{horizon}"
        )

        path_column = (
            f"path_direction_t{horizon}"
        )

        consistency_column = (
            f"path_consistency_t{horizon}"
        )

        required_columns = [
            direction_column,
            path_column,
            consistency_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            raise RuntimeError(
                "Missing Stage 3.14 stability "
                "columns for horizon "
                + str(horizon)
                + ": "
                + str(missing_columns)
            )

        valid = data[
            data[
                direction_column
            ].isin(
                [
                    "GROWTH",
                    "REDUCTION",
                    "STABLE",
                ]
            )
        ].copy()

        if len(valid) == 0:
            continue

        # IMPORTANT:
        # Compare DATAFRAME COLUMNS, not column-name strings.
        stable_path = valid[
            valid[
                path_column
            ]
            ==
            valid[
                direction_column
            ]
        ]

        high_consistency = valid[
            valid[
                consistency_column
            ]
            >= 0.75
        ]

        very_high_consistency = valid[
            valid[
                consistency_column
            ]
            >= 0.90
        ]

        records.append(
            {
                "horizon":
                    horizon,

                "valid_records":
                    len(valid),

                "path_same_as_net_direction":
                    safe_fraction(
                        len(stable_path),
                        len(valid),
                    ),

                "mean_path_consistency":
                    float(
                        valid[
                            consistency_column
                        ].mean()
                    ),

                "median_path_consistency":
                    float(
                        valid[
                            consistency_column
                        ].median()
                    ),

                "high_consistency_fraction":
                    safe_fraction(
                        len(high_consistency),
                        len(valid),
                    ),

                "very_high_consistency_fraction":
                    safe_fraction(
                        len(very_high_consistency),
                        len(valid),
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# PERSISTENCE
# ============================================================================

def build_persistence_analysis(
    data,
):

    print()
    print("=" * 78)
    print(
        "DIRECTION PERSISTENCE"
    )
    print("=" * 78)

    records = []

    for horizon in HORIZONS:

        column = (
            f"path_consistency_t{horizon}"
        )

        valid = data[
            data[
                column
            ].notna()
        ]

        records.append(
            {
                "horizon":
                    horizon,

                "valid_records":
                    len(valid),

                "consistency_ge_50":
                    float(
                        (
                            valid[column]
                            >= 0.50
                        )
                        .mean()
                    ),

                "consistency_ge_75":
                    float(
                        (
                            valid[column]
                            >= 0.75
                        )
                        .mean()
                    ),

                "consistency_ge_90":
                    float(
                        (
                            valid[column]
                            >= 0.90
                        )
                        .mean()
                    ),

                "mean_consistency":
                    float(
                        valid[column]
                        .mean()
                    ),

                "median_consistency":
                    float(
                        valid[column]
                        .median()
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# EXISTING TARGET ALIGNMENT
# ============================================================================

def build_existing_target_alignment(
    data,
):

    print()
    print("=" * 78)
    print(
        "ALIGNMENT WITH EXISTING STAGE 3.6 TARGET"
    )
    print("=" * 78)

    if not TARGET_FILE.exists():

        print(
            "Stage 3.6 target file not found."
        )

        return pd.DataFrame()

    targets = pd.read_csv(
        TARGET_FILE
    )

    sequence_col = find_column(
        targets,
        ["sequence"],
    )

    component_col = find_column(
        targets,
        ["component"],
    )

    frame_col = find_column(
        targets,
        [
            "current_frame",
            "frame_number",
            "target_current_frame",
        ],
    )

    label_col = find_column(
        targets,
        [
            "persistent_damage_target",
            "target_state",
        ],
    )

    targets["sequence"] = (
        targets[
            sequence_col
        ]
        .astype(str)
    )

    targets["component"] = pd.to_numeric(
        targets[
            component_col
        ],
        errors="coerce",
    )

    targets["current_frame"] = pd.to_numeric(
        targets[
            frame_col
        ],
        errors="coerce",
    )

    targets[
        "existing_direction"
    ] = targets[
        label_col
    ].map(
        existing_target_direction
    )

    target_keys = [
        "sequence",
        "component",
        "current_frame",
    ]

    target_subset = targets[
        target_keys
        + [
            "existing_direction",
        ]
    ].drop_duplicates(
        subset=target_keys
    )

    merged = data.merge(
        target_subset,
        on=target_keys,
        how="left",
        validate="one_to_one",
    )

    records = []

    for horizon in HORIZONS:

        direction_column = (
            f"direction_stable_t{horizon}"
        )

        valid = merged[
            merged[
                direction_column
            ].isin(
                [
                    "GROWTH",
                    "REDUCTION",
                    "STABLE",
                ]
            )
            &
            merged[
                "existing_direction"
            ].isin(
                [
                    "GROWTH",
                    "REDUCTION",
                    "STABLE",
                ]
            )
        ]

        agreement = (
            float(
                (
                    valid[
                        direction_column
                    ]
                    ==
                    valid[
                        "existing_direction"
                    ]
                ).mean()
            )
            if len(valid)
            else 0.0
        )

        records.append(
            {
                "horizon":
                    horizon,

                "valid_records":
                    len(valid),

                "existing_target_agreement":
                    agreement,

                "existing_target_disagreement":
                    1.0 - agreement,
            }
        )

    result = pd.DataFrame(
        records
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================================
# TEST COVERAGE
# ============================================================================

def build_test_coverage(
    data,
):

    print()
    print("=" * 78)
    print(
        "TEST SET FUTURE-WINDOW COVERAGE"
    )
    print("=" * 78)

    if not PREDICTION_FILE.exists():

        return pd.DataFrame()

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    sequence_col = find_column(
        predictions,
        ["sequence"],
    )

    component_col = find_column(
        predictions,
        ["component"],
    )

    frame_col = find_column(
        predictions,
        [
            "current_frame",
            "frame_number",
        ],
    )

    predictions["sequence"] = (
        predictions[
            sequence_col
        ]
        .astype(str)
    )

    predictions["component"] = pd.to_numeric(
        predictions[
            component_col
        ],
        errors="coerce",
    )

    predictions["current_frame"] = pd.to_numeric(
        predictions[
            frame_col
        ],
        errors="coerce",
    )

    keys = [
        "sequence",
        "component",
        "current_frame",
    ]

    test_keys = predictions[
        keys
    ].drop_duplicates()

    test_data = data.merge(
        test_keys,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    records = []

    for horizon in HORIZONS:

        column = (
            f"delta_ratio_t{horizon}"
        )

        valid = test_data[
            test_data[
                column
            ].notna()
        ]

        records.append(
            {
                "horizon":
                    horizon,

                "test_records":
                    len(test_data),

                "future_target_available":
                    len(valid),

                "coverage":
                    safe_fraction(
                        len(valid),
                        len(test_data),
                    ),

                "mean_abs_damage_change":
                    (
                        float(
                            valid[
                                column
                            ]
                            .abs()
                            .mean()
                        )
                        if len(valid)
                        else 0.0
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

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
        "STAGE 3.14 - FUTURE-WINDOW PHYSICAL TARGET AUDIT"
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
    # CHECK
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CHECKING INPUT FILES"
    )
    print("=" * 78)

    for label, path in [
        (
            "damage_features",
            FEATURE_FILE,
        ),
        (
            "existing_target",
            TARGET_FILE,
        ),
        (
            "test_predictions",
            PREDICTION_FILE,
        ),
    ]:

        if path.exists():

            print(
                f"{label:<20}: FOUND"
            )

            print(
                "   ",
                path,
            )

        else:

            print(
                f"{label:<20}: NOT FOUND"
            )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            str(FEATURE_FILE)
        )

    # ========================================================================
    # BUILD
    # ========================================================================

    features = prepare_features()

    data = build_future_targets(
        features
    )

    data = add_path_direction_features(
        data
    )

    # ========================================================================
    # HORIZON
    # ========================================================================

    horizon_df = build_horizon_summary(
        data
    )

    horizon_df.to_csv(
        HORIZON_OUTPUT,
        index=False,
    )

    # ========================================================================
    # STABILITY
    # ========================================================================

    stability_df = build_direction_stability(
        data
    )

    stability_df.to_csv(
        STABILITY_OUTPUT,
        index=False,
    )

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    persistence_df = build_persistence_analysis(
        data
    )

    persistence_df.to_csv(
        PERSISTENCE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # ALIGNMENT
    # ========================================================================

    alignment_df = (
        build_existing_target_alignment(
            data
        )
    )

    if len(alignment_df) > 0:

        alignment_df.to_csv(
            ALIGNMENT_OUTPUT,
            index=False,
        )

    # ========================================================================
    # TEST
    # ========================================================================

    test_df = build_test_coverage(
        data
    )

    if len(test_df) > 0:

        test_df.to_csv(
            TEST_OUTPUT,
            index=False,
        )

    # ========================================================================
    # DISTRIBUTION
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "FUTURE DIRECTION DISTRIBUTION"
    )
    print("=" * 78)

    distribution_records = []

    for horizon in HORIZONS:

        column = (
            f"direction_stable_t{horizon}"
        )

        counts = (
            data[
                column
            ]
            .value_counts(
                dropna=False
            )
        )

        total = len(data)

        for class_name, count in counts.items():

            distribution_records.append(
                {
                    "horizon":
                        horizon,

                    "target_type":
                        "NET_FUTURE_DIRECTION",

                    "class":
                        class_name,

                    "count":
                        int(count),

                    "fraction":
                        safe_fraction(
                            count,
                            total,
                        ),
                }
            )

    distribution_df = pd.DataFrame(
        distribution_records
    )

    print(
        distribution_df.to_string(
            index=False
        )
    )

    distribution_df.to_csv(
        DISTRIBUTION_OUTPUT,
        index=False,
    )

    # ========================================================================
    # FRAME OUTPUT
    # ========================================================================

    output_columns = [
        "sequence",
        "component",
        "current_frame",
        "crack_area_ratio",
        "total_crack_area_px",
        "crack_count",
    ]

    for horizon in HORIZONS:

        output_columns.extend(
            [
                f"future_ratio_t{horizon}",
                f"delta_ratio_t{horizon}",
                f"abs_delta_ratio_t{horizon}",
                f"direction_t{horizon}",
                f"direction_stable_t{horizon}",
                f"path_direction_t{horizon}",
                f"path_consistency_t{horizon}",
                f"path_growth_steps_t{horizon}",
                f"path_reduction_steps_t{horizon}",
                f"path_stable_steps_t{horizon}",
            ]
        )

    frame_columns = [
        column
        for column in output_columns
        if column in data.columns
    ]

    data[
        frame_columns
    ].to_csv(
        FRAME_OUTPUT,
        index=False,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.14",

        "title":
            "Future-Window Physical Target Audit",

        "input_records":
            len(features),

        "sequence_count":
            int(
                features[
                    "sequence"
                ].nunique()
            ),

        "horizons":
            HORIZONS,

        "stable_epsilon":
            STEP_STABLE_EPSILON,

        "purpose":
            (
                "Audit future physical crack-area "
                "trajectory targets without replacing "
                "the existing Stage 3.6 target."
            ),

        "target_replacement":
            False,

        "horizon_summary":
            horizon_df.to_dict(
                orient="records"
            ),

        "direction_stability":
            stability_df.to_dict(
                orient="records"
            ),

        "persistence":
            persistence_df.to_dict(
                orient="records"
            ),

        "existing_target_alignment":
            (
                alignment_df.to_dict(
                    orient="records"
                )
                if len(alignment_df) > 0
                else []
            ),

        "test_coverage":
            (
                test_df.to_dict(
                    orient="records"
                )
                if len(test_df) > 0
                else []
            ),

        "outputs": {

            "frame_level":
                str(
                    FRAME_OUTPUT
                ),

            "horizon_comparison":
                str(
                    HORIZON_OUTPUT
                ),

            "distribution":
                str(
                    DISTRIBUTION_OUTPUT
                ),

            "direction_stability":
                str(
                    STABILITY_OUTPUT
                ),

            "persistence":
                str(
                    PERSISTENCE_OUTPUT
                ),

            "existing_target_alignment":
                str(
                    ALIGNMENT_OUTPUT
                ),

            "test_coverage":
                str(
                    TEST_OUTPUT
                ),

            "summary":
                str(
                    SUMMARY_OUTPUT
                ),
        },
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            to_json_safe(
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
        "STAGE 3.14 COMPLETE"
    )
    print("=" * 78)

    print(
        "Future horizons analyzed:",
        ", ".join(
            str(x)
            for x in HORIZONS
        ),
    )

    print(
        "Physical target construction: PASS"
    )

    print(
        "Existing target modified: NO"
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