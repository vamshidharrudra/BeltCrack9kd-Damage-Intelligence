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
# INPUT FILES
# ============================================================================

TARGET_FILE = (
    AUDIT_DIR /
    "stage_3_6_persistent_damage_targets.csv"
)

FEATURE_FILE = (
    AUDIT_DIR /
    "stage_1_0_damage_features.csv"
)

PREDICTION_FILE = (
    AUDIT_DIR /
    "stage_3_10_test_predictions.csv"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_temporal_target_redesign_summary.json"
)

FRAME_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_frame_level_target_comparison.csv"
)

STABILITY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_target_stability_comparison.csv"
)

CLASS_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_class_distribution_comparison.csv"
)

PHYSICAL_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_physical_alignment_comparison.csv"
)

STREAK_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_streak_comparison.csv"
)

TEST_OUTPUT = (
    AUDIT_DIR /
    "stage_3_13_test_target_comparison.csv"
)


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
        text
    )

    if matches:

        return float(
            matches[-1]
        )

    return np.nan


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

    return "UNKNOWN"


# ============================================================================
# DIRECTION TO SIMPLE CLASS
# ============================================================================

def simple_direction(
    direction
):

    if direction == "GROWTH":
        return "GROWTH"

    if direction == "REDUCTION":
        return "REDUCTION"

    if direction == "STABLE":
        return "STABLE"

    return "UNKNOWN"


# ============================================================================
# MODE / CONSENSUS
# ============================================================================

def majority_direction(
    values,
):

    values = [
        value
        for value in values
        if value in
        {
            "GROWTH",
            "REDUCTION",
            "STABLE",
        }
    ]

    if len(values) == 0:
        return "UNKNOWN"

    counts = pd.Series(
        values
    ).value_counts()

    top_count = counts.iloc[0]

    winners = (
        counts[
            counts == top_count
        ]
        .index
        .tolist()
    )

    # Tie -> UNKNOWN.
    if len(winners) != 1:
        return "UNKNOWN"

    return winners[0]


# ============================================================================
# CENTERED CONSENSUS
# ============================================================================

def build_centered_consensus(
    directions,
    radius,
):

    result = []

    n = len(
        directions
    )

    for i in range(n):

        start = max(
            0,
            i - radius
        )

        end = min(
            n,
            i + radius + 1
        )

        window = directions[
            start:end
        ]

        result.append(
            majority_direction(
                window
            )
        )

    return result


# ============================================================================
# CAUSAL CONSENSUS
# ============================================================================

def build_causal_consensus(
    directions,
    window_size,
):

    result = []

    n = len(
        directions
    )

    for i in range(n):

        start = max(
            0,
            i - window_size + 1
        )

        end = i + 1

        window = directions[
            start:end
        ]

        result.append(
            majority_direction(
                window
            )
        )

    return result


# ============================================================================
# STABILITY METRICS
# ============================================================================

def compute_stability(
    labels
):

    labels = [
        value
        for value in labels
        if value != "UNKNOWN"
    ]

    if len(labels) <= 1:

        return {
            "records": len(labels),
            "transitions": 0,
            "transition_rate": 0.0,
            "same_rate": 0.0,
        }

    transitions = sum(
        labels[i]
        != labels[i - 1]
        for i in range(
            1,
            len(labels)
        )
    )

    total_pairs = len(labels) - 1

    return {
        "records": len(labels),
        "transitions": int(
            transitions
        ),
        "transition_rate":
            float(
                transitions
                /
                total_pairs
            ),
        "same_rate":
            float(
                1.0
                -
                transitions
                /
                total_pairs
            ),
    }


# ============================================================================
# DIRECTION CONFLICT
# ============================================================================

def compute_direction_conflict(
    labels
):

    directions = [
        target_direction(
            value
        )
        for value in labels
    ]

    valid = [
        value
        for value in directions
        if value in
        {
            "GROWTH",
            "REDUCTION",
        }
    ]

    if len(valid) <= 2:

        return 0.0

    conflicts = 0
    total = 0

    for i in range(
        1,
        len(valid) - 1
    ):

        previous = valid[
            i - 1
        ]

        current = valid[
            i
        ]

        following = valid[
            i + 1
        ]

        if (
            previous
            !=
            following
        ):

            conflicts += 1

        total += 1

    if total == 0:
        return 0.0

    return float(
        conflicts
        /
        total
    )


# ============================================================================
# STREAKS
# ============================================================================

def compute_streaks(
    df,
    label_column,
):

    records = []

    for sequence, group in df.groupby(
        "sequence"
    ):

        group = (
            group
            .sort_values(
                "current_frame"
            )
        )

        labels = (
            group[
                label_column
            ]
            .astype(str)
            .tolist()
        )

        frames = (
            group[
                "current_frame"
            ]
            .tolist()
        )

        if not labels:
            continue

        start = 0

        while start < len(labels):

            label = labels[
                start
            ]

            if label == "UNKNOWN":

                start += 1
                continue

            end = start

            while (
                end + 1 < len(labels)
                and labels[
                    end + 1
                ]
                == label
            ):

                end += 1

            records.append(
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

    return pd.DataFrame(
        records
    )


# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

def class_distribution(
    df,
    label_column,
    scheme_name,
):

    counts = (
        df[
            label_column
        ]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    counts.columns = [
        "class",
        "count",
    ]

    total = counts[
        "count"
    ].sum()

    counts[
        "fraction"
    ] = (
        counts[
            "count"
        ]
        /
        total
    )

    counts[
        "scheme"
    ] = scheme_name

    return counts[
        [
            "scheme",
            "class",
            "count",
            "fraction",
        ]
    ]


# ============================================================================
# PHYSICAL ALIGNMENT
# ============================================================================

def physical_alignment(
    df,
    label_column,
):

    work = df.copy()

    if "crack_area_ratio" not in work.columns:

        return {
            "records": 0,
            "growth_positive_fraction": None,
            "reduction_negative_fraction": None,
            "direction_alignment": None,
        }

    work[
        "future_area_ratio"
    ] = (
        work
        .groupby(
            "sequence"
        )[
            "crack_area_ratio"
        ]
        .shift(-1)
    )

    work[
        "area_delta_next"
    ] = (
        work[
            "future_area_ratio"
        ]
        -
        work[
            "crack_area_ratio"
        ]
    )

    work[
        "actual_direction"
    ] = np.select(
        [
            work[
                "area_delta_next"
            ] > 0,

            work[
                "area_delta_next"
            ] < 0,
        ],
        [
            "GROWTH",
            "REDUCTION",
        ],
        default="STABLE",
    )

    work[
        "target_direction"
    ] = (
        work[
            label_column
        ]
        .map(
            target_direction
        )
    )

    valid = work[
        work[
            "actual_direction"
        ].isin(
            [
                "GROWTH",
                "REDUCTION",
            ]
        )
        &
        work[
            "target_direction"
        ].isin(
            [
                "GROWTH",
                "REDUCTION",
            ]
        )
    ]

    if len(valid) == 0:

        return {
            "records": 0,
            "growth_positive_fraction": None,
            "reduction_negative_fraction": None,
            "direction_alignment": None,
        }

    growth = valid[
        valid[
            "target_direction"
        ]
        ==
        "GROWTH"
    ]

    reduction = valid[
        valid[
            "target_direction"
        ]
        ==
        "REDUCTION"
    ]

    growth_positive = (
        float(
            (
                growth[
                    "actual_direction"
                ]
                ==
                "GROWTH"
            )
            .mean()
        )
        if len(growth) > 0
        else 0.0
    )

    reduction_negative = (
        float(
            (
                reduction[
                    "actual_direction"
                ]
                ==
                "REDUCTION"
            )
            .mean()
        )
        if len(reduction) > 0
        else 0.0
    )

    alignment = float(
        (
            valid[
                "actual_direction"
            ]
            ==
            valid[
                "target_direction"
            ]
        )
        .mean()
    )

    return {
        "records":
            len(valid),

        "growth_records":
            len(growth),

        "reduction_records":
            len(reduction),

        "growth_positive_fraction":
            growth_positive,

        "reduction_negative_fraction":
            reduction_negative,

        "direction_alignment":
            alignment,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.13 - TEMPORAL TARGET REDESIGN AUDIT"
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
            "targets",
            TARGET_FILE,
        ),
        (
            "damage_features",
            FEATURE_FILE,
        ),
        (
            "predictions",
            PREDICTION_FILE,
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

    if not TARGET_FILE.exists():

        raise FileNotFoundError(
            str(TARGET_FILE)
        )

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            str(FEATURE_FILE)
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
        TARGET_FILE
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    predictions = None

    if PREDICTION_FILE.exists():

        predictions = pd.read_csv(
            PREDICTION_FILE
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
    # TARGET KEYS
    # ========================================================================

    target_sequence = find_column(
        targets,
        ["sequence"],
    )

    target_component = find_column(
        targets,
        ["component"],
    )

    target_frame = find_column(
        targets,
        [
            "current_frame",
            "frame_number",
            "target_current_frame",
        ],
    )

    target_label = find_column(
        targets,
        [
            "persistent_damage_target",
            "target_state",
        ],
    )

    targets["sequence"] = (
        targets[
            target_sequence
        ]
        .astype(str)
    )

    targets["component"] = pd.to_numeric(
        targets[
            target_component
        ],
        errors="coerce",
    )

    targets["current_frame"] = pd.to_numeric(
        targets[
            target_frame
        ],
        errors="coerce",
    )

    targets[
        "original_target"
    ] = targets[
        target_label
    ].astype(str)

    targets[
        "original_direction"
    ] = targets[
        "original_target"
    ].map(
        target_direction
    )

    # ========================================================================
    # STAGE 1.0 FEATURES
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "PREPARING PHYSICAL FEATURES"
    )
    print("=" * 78)

    feature_sequence = find_column(
        features,
        ["sequence"],
    )

    feature_component = find_column(
        features,
        ["component"],
    )

    feature_filename = find_column(
        features,
        [
            "canonical_image_filename",
            "original_image_filename",
            "source_image",
        ],
    )

    features["sequence"] = (
        features[
            feature_sequence
        ]
        .astype(str)
    )

    features["component"] = pd.to_numeric(
        features[
            feature_component
        ],
        errors="coerce",
    )

    features["current_frame"] = (
        features[
            feature_filename
        ]
        .apply(
            extract_frame_number
        )
    )

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
        ]
        if column in features.columns
    ]

    feature_keys = [
        "sequence",
        "component",
        "current_frame",
    ]

    duplicate_features = int(
        features[
            feature_keys
        ]
        .duplicated()
        .sum()
    )

    print(
        "Feature duplicate keys:",
        duplicate_features,
    )

    if duplicate_features > 0:

        features = (
            features
            .drop_duplicates(
                subset=feature_keys
            )
            .copy()
        )

    features = features[
        feature_keys
        + physical_columns
    ].copy()

    # ========================================================================
    # MERGE
    # ========================================================================

    keys = [
        "sequence",
        "component",
        "current_frame",
    ]

    data = targets[
        keys
        + [
            "original_target",
            "original_direction",
        ]
    ].merge(
        features,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    print(
        "Merged records:",
        len(data),
    )

    if len(data) != len(targets):

        raise RuntimeError(
            "Target/feature merge changed "
            "the expected number of records."
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
    # BUILD ORIGINAL DIRECTION
    # ========================================================================

    directions_by_sequence = {}

    for sequence, group in data.groupby(
        "sequence"
    ):

        group = group.sort_values(
            "current_frame"
        )

        directions_by_sequence[
            sequence
        ] = (
            group[
                "original_direction"
            ]
            .tolist()
        )

    # ========================================================================
    # BUILD TARGET CANDIDATES
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "BUILDING TARGET CANDIDATES"
    )
    print("=" * 78)

    candidate_columns = []

    # ------------------------------------------------------------------------
    # ORIGINAL
    # ------------------------------------------------------------------------

    data[
        "TARGET_ORIGINAL"
    ] = data[
        "original_direction"
    ]

    candidate_columns.append(
        "TARGET_ORIGINAL"
    )

    # ------------------------------------------------------------------------
    # CENTERED 3 / 5 / 7
    # ------------------------------------------------------------------------

    for window in [
        3,
        5,
        7,
    ]:

        radius = (
            window - 1
        ) // 2

        new_column = (
            f"TARGET_CENTERED_{window}"
        )

        result = []

        for sequence, group in data.groupby(
            "sequence"
        ):

            group = group.sort_values(
                "current_frame"
            )

            directions = (
                group[
                    "original_direction"
                ]
                .tolist()
            )

            smoothed = (
                build_centered_consensus(
                    directions,
                    radius,
                )
            )

            result.extend(
                [
                    (
                        sequence,
                        frame,
                        label,
                    )
                    for frame, label
                    in zip(
                        group[
                            "current_frame"
                        ],
                        smoothed,
                    )
                ]
            )

        smoothed_df = pd.DataFrame(
            result,
            columns=[
                "sequence",
                "current_frame",
                new_column,
            ],
        )

        data = data.merge(
            smoothed_df,
            on=[
                "sequence",
                "current_frame",
            ],
            how="left",
            validate="one_to_one",
        )

        candidate_columns.append(
            new_column
        )

    # ------------------------------------------------------------------------
    # CAUSAL 3 / 5 / 7
    # ------------------------------------------------------------------------

    for window in [
        3,
        5,
        7,
    ]:

        new_column = (
            f"TARGET_CAUSAL_{window}"
        )

        result = []

        for sequence, group in data.groupby(
            "sequence"
        ):

            group = group.sort_values(
                "current_frame"
            )

            directions = (
                group[
                    "original_direction"
                ]
                .tolist()
            )

            smoothed = (
                build_causal_consensus(
                    directions,
                    window,
                )
            )

            result.extend(
                [
                    (
                        sequence,
                        frame,
                        label,
                    )
                    for frame, label
                    in zip(
                        group[
                            "current_frame"
                        ],
                        smoothed,
                    )
                ]
            )

        smoothed_df = pd.DataFrame(
            result,
            columns=[
                "sequence",
                "current_frame",
                new_column,
            ],
        )

        data = data.merge(
            smoothed_df,
            on=[
                "sequence",
                "current_frame",
            ],
            how="left",
            validate="one_to_one",
        )

        candidate_columns.append(
            new_column
        )

    print(
        "Candidate target schemes:"
    )

    for column in candidate_columns:

        print(
            " ",
            column,
        )

    # ========================================================================
    # STABILITY COMPARISON
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET STABILITY COMPARISON"
    )
    print("=" * 78)

    stability_records = []

    for column in candidate_columns:

        all_transition_rates = []
        all_conflicts = []

        valid_records = 0

        for sequence, group in data.groupby(
            "sequence"
        ):

            group = group.sort_values(
                "current_frame"
            )

            labels = (
                group[
                    column
                ]
                .astype(str)
                .tolist()
            )

            stability = compute_stability(
                labels
            )

            if stability[
                "records"
            ] > 1:

                all_transition_rates.append(
                    stability[
                        "transition_rate"
                    ]
                )

            valid_records += (
                stability[
                    "records"
                ]
            )

            directions = [
                label
                if label in
                {
                    "GROWTH",
                    "REDUCTION",
                    "STABLE",
                }
                else "UNKNOWN"
                for label in labels
            ]

            valid = [
                value
                for value in directions
                if value in
                {
                    "GROWTH",
                    "REDUCTION",
                }
            ]

            if len(valid) >= 3:

                conflicts = 0
                total = 0

                for i in range(
                    1,
                    len(valid) - 1
                ):

                    if (
                        valid[i - 1]
                        !=
                        valid[i + 1]
                    ):

                        conflicts += 1

                    total += 1

                if total > 0:

                    all_conflicts.append(
                        conflicts
                        /
                        total
                    )

        stability_records.append(
            {
                "scheme":
                    column,

                "records":
                    valid_records,

                "mean_transition_rate":
                    (
                        float(
                            np.mean(
                                all_transition_rates
                            )
                        )
                        if all_transition_rates
                        else 0.0
                    ),

                "median_transition_rate":
                    (
                        float(
                            np.median(
                                all_transition_rates
                            )
                        )
                        if all_transition_rates
                        else 0.0
                    ),

                "mean_direction_conflict":
                    (
                        float(
                            np.mean(
                                all_conflicts
                            )
                        )
                        if all_conflicts
                        else 0.0
                    ),

                "median_direction_conflict":
                    (
                        float(
                            np.median(
                                all_conflicts
                            )
                        )
                        if all_conflicts
                        else 0.0
                    ),
            }
        )

    stability_df = pd.DataFrame(
        stability_records
    )

    print(
        stability_df.to_string(
            index=False
        )
    )

    stability_df.to_csv(
        STABILITY_OUTPUT,
        index=False,
    )

    # ========================================================================
    # CLASS DISTRIBUTION
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CLASS DISTRIBUTION"
    )
    print("=" * 78)

    distribution_frames = []

    for column in candidate_columns:

        distribution_frames.append(
            class_distribution(
                data,
                column,
                column,
            )
        )

    distribution_df = pd.concat(
        distribution_frames,
        ignore_index=True,
    )

    print(
        distribution_df.to_string(
            index=False
        )
    )

    distribution_df.to_csv(
        CLASS_OUTPUT,
        index=False,
    )

    # ========================================================================
    # PHYSICAL ALIGNMENT
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "PHYSICAL DAMAGE ALIGNMENT"
    )
    print("=" * 78)

    physical_records = []

    for column in candidate_columns:

        alignment = physical_alignment(
            data,
            column,
        )

        alignment[
            "scheme"
        ] = column

        physical_records.append(
            alignment
        )

        print(
            column
        )

        print(
            "  valid:",
            alignment[
                "records"
            ]
        )

        print(
            "  direction alignment:",
            (
                f"{alignment['direction_alignment']:.6f}"
                if alignment[
                    "direction_alignment"
                ] is not None
                else "NA"
            ),
        )

    physical_df = pd.DataFrame(
        physical_records
    )

    physical_df.to_csv(
        PHYSICAL_OUTPUT,
        index=False,
    )

    # ========================================================================
    # STREAK COMPARISON
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "STREAK COMPARISON"
    )
    print("=" * 78)

    streak_records = []

    for column in candidate_columns:

        streak_df = compute_streaks(
            data,
            column,
        )

        if len(streak_df) == 0:
            continue

        for class_name, group in streak_df.groupby(
            "class"
        ):

            streak_records.append(
                {
                    "scheme":
                        column,

                    "class":
                        class_name,

                    "streak_count":
                        len(group),

                    "mean_streak":
                        float(
                            group[
                                "length"
                            ].mean()
                        ),

                    "median_streak":
                        float(
                            group[
                                "length"
                            ].median()
                        ),

                    "max_streak":
                        int(
                            group[
                                "length"
                            ].max()
                        ),
                }
            )

    streak_comparison = pd.DataFrame(
        streak_records
    )

    print(
        streak_comparison.to_string(
            index=False
        )
    )

    streak_comparison.to_csv(
        STREAK_OUTPUT,
        index=False,
    )

    # ========================================================================
    # TEST SET ANALYSIS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TEST SET TARGET ANALYSIS"
    )
    print("=" * 78)

    test_df = None

    if predictions is not None:

        pred_sequence = find_column(
            predictions,
            ["sequence"],
        )

        pred_component = find_column(
            predictions,
            ["component"],
        )

        pred_frame = find_column(
            predictions,
            [
                "current_frame",
                "frame_number",
            ],
        )

        predictions["sequence"] = (
            predictions[
                pred_sequence
            ]
            .astype(str)
        )

        predictions["component"] = (
            pd.to_numeric(
                predictions[
                    pred_component
                ],
                errors="coerce",
            )
        )

        predictions["current_frame"] = (
            pd.to_numeric(
                predictions[
                    pred_frame
                ],
                errors="coerce",
            )
        )

        prediction_keys = predictions[
            [
                "sequence",
                "component",
                "current_frame",
            ]
        ].copy()

        prediction_keys[
            "_test_record"
        ] = True

        test_df = data.merge(
            prediction_keys,
            on=[
                "sequence",
                "component",
                "current_frame",
            ],
            how="inner",
            validate="one_to_one",
        )

        print(
            "Test records:",
            len(test_df),
        )

        test_records = []

        for column in candidate_columns:

            labels = (
                test_df[
                    column
                ]
                .astype(str)
            )

            known = (
                labels
                !=
                "UNKNOWN"
            )

            test_records.append(
                {
                    "scheme":
                        column,

                    "test_records":
                        len(test_df),

                    "known_target_fraction":
                        float(
                            known.mean()
                        ),

                    "unknown_records":
                        int(
                            (~known).sum()
                        ),
                }
            )

        test_comparison = pd.DataFrame(
            test_records
        )

        print(
            test_comparison.to_string(
                index=False
            )
        )

        test_comparison.to_csv(
            TEST_OUTPUT,
            index=False,
        )

    else:

        print(
            "No prediction file available."
        )

    # ========================================================================
    # SELECT BEST CANDIDATE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TARGET REDESIGN DECISION"
    )
    print("=" * 78)

    decision_table = (
        stability_df
        .merge(
            physical_df[
                [
                    "scheme",
                    "direction_alignment",
                ]
            ],
            on="scheme",
            how="left",
        )
    )

    print(
        decision_table.to_string(
            index=False
        )
    )

    # We do NOT automatically replace the target.
    # This stage is an audit.
    #
    # Rank primarily by:
    # 1. lower direction conflict
    # 2. higher physical alignment
    #
    ranking = decision_table.copy()

    ranking[
        "conflict_rank"
    ] = ranking[
        "mean_direction_conflict"
    ].rank(
        ascending=True,
        method="min",
    )

    ranking[
        "alignment_rank"
    ] = ranking[
        "direction_alignment"
    ].fillna(
        0.0
    ).rank(
        ascending=False,
        method="min",
    )

    ranking[
        "combined_rank"
    ] = (
        ranking[
            "conflict_rank"
        ]
        +
        ranking[
            "alignment_rank"
        ]
    )

    ranking = ranking.sort_values(
        [
            "combined_rank",
            "mean_direction_conflict",
        ]
    )

    print()
    print(
        "Candidate ranking:"
    )

    print(
        ranking[
            [
                "scheme",
                "mean_direction_conflict",
                "direction_alignment",
                "combined_rank",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================================
    # FRAME-LEVEL OUTPUT
    # ========================================================================

    output_columns = [
        "sequence",
        "component",
        "current_frame",
        "original_target",
        "original_direction",
    ]

    output_columns.extend(
        candidate_columns[1:]
    )

    frame_output = data[
        output_columns
    ].copy()

    frame_output.to_csv(
        FRAME_OUTPUT,
        index=False,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    best_scheme = None

    if len(ranking) > 0:

        best_scheme = str(
            ranking.iloc[0][
                "scheme"
            ]
        )

    original_conflict = float(
        stability_df.loc[
            stability_df[
                "scheme"
            ]
            ==
            "TARGET_ORIGINAL",
            "mean_direction_conflict",
        ].iloc[0]
    )

    best_conflict = float(
        ranking.iloc[0][
            "mean_direction_conflict"
        ]
    )

    if (
        best_scheme
        ==
        "TARGET_ORIGINAL"
    ):

        diagnostic = (
            "CURRENT_TARGET_RETAINS_BEST_AUDIT_BALANCE"
        )

    elif (
        best_conflict
        <
        original_conflict
        *
        0.80
    ):

        diagnostic = (
            "TEMPORAL_REDESIGN_STRONGLY_SUPPORTED"
        )

    elif (
        best_conflict
        <
        original_conflict
    ):

        diagnostic = (
            "TEMPORAL_REDESIGN_PARTIALLY_SUPPORTED"
        )

    else:

        diagnostic = (
            "NO_CLEAR_TEMPORAL_REDESIGN_ADVANTAGE"
        )

    summary = {

        "stage":
            "3.13",

        "title":
            "Temporal Target Smoothing and Redesign Audit",

        "input_target_records":
            len(targets),

        "merged_records":
            len(data),

        "sequence_count":
            int(
                data[
                    "sequence"
                ].nunique()
            ),

        "candidate_schemes":
            candidate_columns,

        "original_direction_conflict":
            original_conflict,

        "best_candidate":
            best_scheme,

        "best_direction_conflict":
            best_conflict,

        "diagnostic":
            diagnostic,

        "decision_note":
            (
                "This stage is an audit only. "
                "No target replacement is performed."
            ),

        "outputs": {

            "frame_comparison":
                str(
                    FRAME_OUTPUT
                ),

            "stability_comparison":
                str(
                    STABILITY_OUTPUT
                ),

            "class_distribution":
                str(
                    CLASS_OUTPUT
                ),

            "physical_alignment":
                str(
                    PHYSICAL_OUTPUT
                ),

            "streak_comparison":
                str(
                    STREAK_OUTPUT
                ),

            "test_comparison":
                str(
                    TEST_OUTPUT
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
        "STAGE 3.13 COMPLETE"
    )
    print("=" * 78)

    print(
        "Original direction conflict:",
        f"{original_conflict:.6f}",
    )

    print(
        "Best candidate:",
        best_scheme,
    )

    print(
        "Best direction conflict:",
        f"{best_conflict:.6f}",
    )

    print(
        "Diagnostic:",
        diagnostic,
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