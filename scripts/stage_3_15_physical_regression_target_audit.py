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

SPLIT_PLAN_FILE = (
    AUDIT_DIR /
    "stage_0_9d1_optimized_split_plan.json"
)


# ============================================================================
# OUTPUTS
# ============================================================================

FRAME_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_physical_regression_targets.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_physical_regression_summary.json"
)

DISTRIBUTION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_target_distribution.csv"
)

SEQUENCE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_sequence_target_analysis.csv"
)

SPLIT_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_split_target_analysis.csv"
)

EXTREME_OUTPUT = (
    AUDIT_DIR /
    "stage_3_15_extreme_transition_analysis.csv"
)


# ============================================================================
# CONSTANTS
# ============================================================================

HORIZON = 1


# ============================================================================
# HELPERS
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


def describe_series(
    series
):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(series) == 0:

        return {
            "count": 0,
        }

    return {
        "count":
            int(len(series)),

        "mean":
            float(series.mean()),

        "std":
            float(series.std()),

        "min":
            float(series.min()),

        "p01":
            float(series.quantile(0.01)),

        "p05":
            float(series.quantile(0.05)),

        "p25":
            float(series.quantile(0.25)),

        "median":
            float(series.median()),

        "p75":
            float(series.quantile(0.75)),

        "p95":
            float(series.quantile(0.95)),

        "p99":
            float(series.quantile(0.99)),

        "max":
            float(series.max()),
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

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Raw records:",
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

    duplicate_keys = int(
        features[
            keys
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate keys:",
        duplicate_keys,
    )

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate physical feature keys detected."
        )

    missing_frames = int(
        features[
            "current_frame"
        ].isna().sum()
    )

    missing_ratio = int(
        features[
            "crack_area_ratio"
        ].isna().sum()
    )

    print(
        "Missing frame numbers:",
        missing_frames,
    )

    print(
        "Missing crack_area_ratio:",
        missing_ratio,
    )

    if missing_frames > 0:

        raise RuntimeError(
            "Missing frame numbers."
        )

    if missing_ratio > 0:

        raise RuntimeError(
            "Missing crack_area_ratio values."
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
# BUILD REGRESSION TARGETS
# ============================================================================

def build_targets(
    features
):

    print()
    print("=" * 78)
    print(
        "BUILDING PHYSICAL REGRESSION TARGETS"
    )
    print("=" * 78)

    data = features.copy()

    group = data.groupby(
        "sequence"
    )

    data[
        "future_crack_area_ratio_t1"
    ] = (
        group[
            "crack_area_ratio"
        ]
        .shift(-1)
    )

    data[
        "future_total_crack_area_px_t1"
    ] = (
        group[
            "total_crack_area_px"
        ]
        .shift(-1)
    )

    data[
        "future_crack_count_t1"
    ] = (
        group[
            "crack_count"
        ]
        .shift(-1)
    )

    data[
        "target_delta_crack_area_ratio"
    ] = (
        data[
            "future_crack_area_ratio_t1"
        ]
        -
        data[
            "crack_area_ratio"
        ]
    )

    data[
        "target_abs_delta_crack_area_ratio"
    ] = (
        data[
            "target_delta_crack_area_ratio"
        ]
        .abs()
    )

    data[
        "target_relative_change"
    ] = np.where(
        data[
            "crack_area_ratio"
        ].abs()
        > 1e-12,

        data[
            "target_delta_crack_area_ratio"
        ]
        /
        data[
            "crack_area_ratio"
        ].abs(),

        np.nan,
    )

    data[
        "target_delta_total_area_px"
    ] = (
        data[
            "future_total_crack_area_px_t1"
        ]
        -
        data[
            "total_crack_area_px"
        ]
    )

    data[
        "target_delta_crack_count"
    ] = (
        data[
            "future_crack_count_t1"
        ]
        -
        data[
            "crack_count"
        ]
    )

    data[
        "target_direction"
    ] = np.select(
        [
            data[
                "target_delta_crack_area_ratio"
            ]
            > 0,

            data[
                "target_delta_crack_area_ratio"
            ]
            < 0,
        ],
        [
            "GROWTH",
            "REDUCTION",
        ],
        default="STABLE",
    )

    valid = data[
        data[
            "future_crack_area_ratio_t1"
        ].notna()
    ].copy()

    print(
        "Valid t+1 regression targets:",
        len(valid),
    )

    print(
        "Expected:",
        len(features) - features.groupby(
            "sequence"
        ).size().shape[0],
    )

    return data, valid


# ============================================================================
# TARGET DISTRIBUTION
# ============================================================================

def build_distribution(
    valid
):

    print()
    print("=" * 78)
    print(
        "TARGET DISTRIBUTION"
    )
    print("=" * 78)

    records = []

    target_columns = [
        (
            "future_crack_area_ratio_t1",
            "FUTURE_DAMAGE_LEVEL",
        ),
        (
            "target_delta_crack_area_ratio",
            "FUTURE_DAMAGE_CHANGE",
        ),
        (
            "target_relative_change",
            "FUTURE_RELATIVE_CHANGE",
        ),
        (
            "target_delta_total_area_px",
            "FUTURE_TOTAL_AREA_CHANGE",
        ),
        (
            "target_delta_crack_count",
            "FUTURE_CRACK_COUNT_CHANGE",
        ),
    ]

    for column, target_type in target_columns:

        stats = describe_series(
            valid[
                column
            ]
        )

        stats[
            "target_type"
        ] = target_type

        stats[
            "column"
        ] = column

        records.append(
            stats
        )

        print()
        print(
            target_type
        )

        for key, value in stats.items():

            if key not in {
                "target_type",
                "column",
            }:

                print(
                    f"  {key:<10}: {value}"
                )

    return pd.DataFrame(
        records
    )


# ============================================================================
# SEQUENCE ANALYSIS
# ============================================================================

def build_sequence_analysis(
    valid
):

    print()
    print("=" * 78)
    print(
        "SEQUENCE-LEVEL TARGET ANALYSIS"
    )
    print("=" * 78)

    records = []

    for sequence, group in valid.groupby(
        "sequence"
    ):

        group = group.sort_values(
            "current_frame"
        )

        delta = group[
            "target_delta_crack_area_ratio"
        ]

        abs_delta = (
            delta.abs()
        )

        positive_fraction = float(
            (
                delta > 0
            ).mean()
        )

        negative_fraction = float(
            (
                delta < 0
            ).mean()
        )

        records.append(
            {
                "sequence":
                    sequence,

                "records":
                    len(group),

                "mean_current_damage":
                    float(
                        group[
                            "crack_area_ratio"
                        ].mean()
                    ),

                "mean_future_damage":
                    float(
                        group[
                            "future_crack_area_ratio_t1"
                        ].mean()
                    ),

                "mean_delta":
                    float(
                        delta.mean()
                    ),

                "mean_abs_delta":
                    float(
                        abs_delta.mean()
                    ),

                "positive_fraction":
                    positive_fraction,

                "negative_fraction":
                    negative_fraction,

                "max_abs_delta":
                    float(
                        abs_delta.max()
                    ),

                "p95_abs_delta":
                    float(
                        abs_delta.quantile(
                            0.95
                        )
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
# SPLIT ANALYSIS
# ============================================================================

def determine_split(
    data
):

    # First preference: use existing component assignment.
    assignment_file = (
        AUDIT_DIR /
        "stage_0_9d1_component_assignments.csv"
    )

    if assignment_file.exists():

        assignments = pd.read_csv(
            assignment_file
        )

        print(
            "Using optimized component assignments."
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

        mapping = (
            assignments[
                [
                    "component",
                    "ml_split",
                ]
            ]
            .drop_duplicates()
        )

        data = data.merge(
            mapping,
            on="component",
            how="left",
            validate="many_to_one",
        )

        return data

    print(
        "Component assignment file not found."
    )

    return data


def build_split_analysis(
    data
):

    print()
    print("=" * 78)
    print(
        "SPLIT-LEVEL TARGET ANALYSIS"
    )
    print("=" * 78)

    if "ml_split" not in data.columns:

        print(
            "No ML split column available."
        )

        return pd.DataFrame()

    records = []

    for split, group in data.groupby(
        "ml_split"
    ):

        delta = group[
            "target_delta_crack_area_ratio"
        ]

        abs_delta = (
            delta.abs()
        )

        records.append(
            {
                "split":
                    split,

                "records":
                    len(group),

                "mean_current_damage":
                    float(
                        group[
                            "crack_area_ratio"
                        ].mean()
                    ),

                "mean_future_damage":
                    float(
                        group[
                            "future_crack_area_ratio_t1"
                        ].mean()
                    ),

                "mean_delta":
                    float(
                        delta.mean()
                    ),

                "mean_abs_delta":
                    float(
                        abs_delta.mean()
                    ),

                "median_abs_delta":
                    float(
                        abs_delta.median()
                    ),

                "p95_abs_delta":
                    float(
                        abs_delta.quantile(
                            0.95
                        )
                    ),

                "positive_fraction":
                    float(
                        (
                            delta > 0
                        ).mean()
                    ),

                "negative_fraction":
                    float(
                        (
                            delta < 0
                        ).mean()
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
# EXTREME TRANSITIONS
# ============================================================================

def build_extreme_analysis(
    valid
):

    print()
    print("=" * 78)
    print(
        "EXTREME PHYSICAL TRANSITIONS"
    )
    print("=" * 78)

    delta = (
        valid[
            "target_abs_delta_crack_area_ratio"
        ]
    )

    p95 = float(
        delta.quantile(
            0.95
        )
    )

    p99 = float(
        delta.quantile(
            0.99
        )
    )

    records = []

    for threshold_name, threshold in [
        (
            "P90",
            float(
                delta.quantile(
                    0.90
                )
            ),
        ),
        (
            "P95",
            p95,
        ),
        (
            "P99",
            p99,
        ),
    ]:

        extreme = valid[
            delta
            >=
            threshold
        ]

        records.append(
            {
                "threshold":
                    threshold_name,

                "threshold_value":
                    threshold,

                "records":
                    len(extreme),

                "fraction":
                    safe_fraction(
                        len(extreme),
                        len(valid),
                    ),

                "growth_records":
                    int(
                        (
                            extreme[
                                "target_direction"
                            ]
                            ==
                            "GROWTH"
                        ).sum()
                    ),

                "reduction_records":
                    int(
                        (
                            extreme[
                                "target_direction"
                            ]
                            ==
                            "REDUCTION"
                        ).sum()
                    ),

                "mean_signed_delta":
                    float(
                        extreme[
                            "target_delta_crack_area_ratio"
                        ].mean()
                    ),

                "mean_abs_delta":
                    float(
                        extreme[
                            "target_abs_delta_crack_area_ratio"
                        ].mean()
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
# TARGET NOISE / DIRECTION FLIPS
# ============================================================================

def build_direction_flip_analysis(
    valid
):

    print()
    print("=" * 78)
    print(
        "CONSECUTIVE DIRECTION FLIP ANALYSIS"
    )
    print("=" * 78)

    records = []

    for sequence, group in valid.groupby(
        "sequence"
    ):

        group = group.sort_values(
            "current_frame"
        ).copy()

        direction = (
            group[
                "target_direction"
            ]
            .tolist()
        )

        flips = 0
        valid_pairs = 0

        for index in range(
            1,
            len(direction)
        ):

            previous = direction[
                index - 1
            ]

            current = direction[
                index
            ]

            if previous in {
                "GROWTH",
                "REDUCTION",
            } and current in {
                "GROWTH",
                "REDUCTION",
            }:

                valid_pairs += 1

                if previous != current:

                    flips += 1

        records.append(
            {
                "sequence":
                    sequence,

                "valid_direction_pairs":
                    valid_pairs,

                "direction_flips":
                    flips,

                "flip_rate":
                    safe_fraction(
                        flips,
                        valid_pairs,
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
        "STAGE 3.15 - PHYSICAL REGRESSION TARGET AUDIT"
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
            "damage_features",
            FEATURE_FILE,
        ),
        (
            "existing_target",
            TARGET_FILE,
        ),
        (
            "split_assignments",
            SPLIT_PLAN_FILE,
        ),
    ]:

        status = (
            "FOUND"
            if path.exists()
            else "NOT FOUND"
        )

        print(
            f"{label:<20}: {status}"
        )

        if path.exists():

            print(
                "   ",
                path,
            )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            str(FEATURE_FILE)
        )

    # ========================================================================
    # LOAD
    # ========================================================================

    features = load_features()

    data, valid = build_targets(
        features
    )

    # ========================================================================
    # DISTRIBUTION
    # ========================================================================

    distribution_df = build_distribution(
        valid
    )

    distribution_df.to_csv(
        DISTRIBUTION_OUTPUT,
        index=False,
    )

    # ========================================================================
    # SEQUENCE
    # ========================================================================

    sequence_df = build_sequence_analysis(
        valid
    )

    sequence_df.to_csv(
        SEQUENCE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # SPLIT
    # ========================================================================

    split_data = determine_split(
        valid
    )

    split_df = build_split_analysis(
        split_data
    )

    if len(split_df) > 0:

        split_df.to_csv(
            SPLIT_OUTPUT,
            index=False,
        )

    # ========================================================================
    # EXTREMES
    # ========================================================================

    extreme_df = build_extreme_analysis(
        valid
    )

    extreme_df.to_csv(
        EXTREME_OUTPUT,
        index=False,
    )

    # ========================================================================
    # FLIPS
    # ========================================================================

    flip_df = build_direction_flip_analysis(
        valid
    )

    # ========================================================================
    # FRAME OUTPUT
    # ========================================================================

    output_columns = [
        "sequence",
        "component",
        "current_frame",
        "crack_area_ratio",
        "future_crack_area_ratio_t1",
        "target_delta_crack_area_ratio",
        "target_abs_delta_crack_area_ratio",
        "target_relative_change",
        "total_crack_area_px",
        "future_total_crack_area_px_t1",
        "target_delta_total_area_px",
        "crack_count",
        "future_crack_count_t1",
        "target_delta_crack_count",
        "target_direction",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in valid.columns
    ]

    valid[
        output_columns
    ].to_csv(
        FRAME_OUTPUT,
        index=False,
    )

    # ========================================================================
    # OVERALL STATS
    # ========================================================================

    delta = valid[
        "target_delta_crack_area_ratio"
    ]

    abs_delta = valid[
        "target_abs_delta_crack_area_ratio"
    ]

    positive = int(
        (
            delta > 0
        ).sum()
    )

    negative = int(
        (
            delta < 0
        ).sum()
    )

    zero = int(
        (
            delta == 0
        ).sum()
    )

    p90 = float(
        abs_delta.quantile(
            0.90
        )
    )

    p95 = float(
        abs_delta.quantile(
            0.95
        )
    )

    p99 = float(
        abs_delta.quantile(
            0.99
        )
    )

    # ========================================================================
    # BASELINE INFORMATION
    # ========================================================================

    current_target_mean = float(
        valid[
            "crack_area_ratio"
        ].mean()
    )

    future_target_mean = float(
        valid[
            "future_crack_area_ratio_t1"
        ].mean()
    )

    mean_absolute_change = float(
        abs_delta.mean()
    )

    relative_change = valid[
        "target_relative_change"
    ].dropna()

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.15",

        "title":
            "Physical Regression Target Audit",

        "horizon":
            HORIZON,

        "input_records":
            len(features),

        "valid_target_records":
            len(valid),

        "sequence_count":
            int(
                features[
                    "sequence"
                ].nunique()
            ),

        "target":
            "crack_area_ratio_t_plus_1",

        "target_change":
            "future_crack_area_ratio_t_plus_1_minus_current",

        "target_statistics":
            describe_series(
                valid[
                    "future_crack_area_ratio_t1"
                ]
            ),

        "target_change_statistics":
            describe_series(
                valid[
                    "target_delta_crack_area_ratio"
                ]
            ),

        "absolute_change_statistics":
            describe_series(
                valid[
                    "target_abs_delta_crack_area_ratio"
                ]
            ),

        "relative_change_statistics":
            describe_series(
                valid[
                    "target_relative_change"
                ]
            ),

        "current_mean_damage":
            current_target_mean,

        "future_mean_damage":
            future_target_mean,

        "mean_absolute_change":
            mean_absolute_change,

        "positive_change_records":
            positive,

        "negative_change_records":
            negative,

        "zero_change_records":
            zero,

        "positive_fraction":
            safe_fraction(
                positive,
                len(valid),
            ),

        "negative_fraction":
            safe_fraction(
                negative,
                len(valid),
            ),

        "zero_fraction":
            safe_fraction(
                zero,
                len(valid),
            ),

        "p90_absolute_change":
            p90,

        "p95_absolute_change":
            p95,

        "p99_absolute_change":
            p99,

        "sequence_analysis":
            sequence_df.to_dict(
                orient="records"
            ),

        "split_analysis":
            (
                split_df.to_dict(
                    orient="records"
                )
                if len(split_df) > 0
                else []
            ),

        "extreme_analysis":
            extreme_df.to_dict(
                orient="records"
            ),

        "direction_flip_analysis":
            flip_df.to_dict(
                orient="records"
            ),

        "decision":
            (
                "PHYSICAL_T_PLUS_1_REGRESSION_TARGET_READY_FOR_MODEL_AUDIT"
            ),

        "important_note":
            (
                "This stage defines and audits a physical "
                "future damage target. It does not train "
                "or select a final ML model."
            ),

        "outputs": {

            "frame_targets":
                str(
                    FRAME_OUTPUT
                ),

            "distribution":
                str(
                    DISTRIBUTION_OUTPUT
                ),

            "sequence_analysis":
                str(
                    SEQUENCE_OUTPUT
                ),

            "split_analysis":
                str(
                    SPLIT_OUTPUT
                ),

            "extreme_analysis":
                str(
                    EXTREME_OUTPUT
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
        "STAGE 3.15 COMPLETE"
    )
    print("=" * 78)

    print(
        "Target:",
        "crack_area_ratio(t+1)",
    )

    print(
        "Valid targets:",
        len(valid),
    )

    print(
        "Positive change:",
        positive,
        f"({positive / len(valid):.6f})",
    )

    print(
        "Negative change:",
        negative,
        f"({negative / len(valid):.6f})",
    )

    print(
        "Zero change:",
        zero,
        f"({zero / len(valid):.6f})",
    )

    print(
        "Mean absolute physical change:",
        f"{mean_absolute_change:.8f}",
    )

    print(
        "P95 absolute physical change:",
        f"{p95:.8f}",
    )

    print()
    print(
        "FINAL STATUS: PASS"
    )

    print()
    print(
        "Existing Stage 3.6 target modified: NO"
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