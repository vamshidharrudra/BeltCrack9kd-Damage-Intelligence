import os
import json
import re
import math
import numpy as np
import pandas as pd


# ============================================================
# STAGE 1.1
# SEQUENTIAL / TEMPORAL DAMAGE ANALYSIS
#
# Purpose:
#   Analyze how the physical crack/damage features change
#   across ordered frames within each conveyor-belt sequence.
#
# Important:
#   This stage does NOT assume that temporal change automatically
#   means physical damage progression.
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_1_0_damage_features.csv"
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

FRAME_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_1_sequence_temporal_analysis.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_1_sequence_summary.csv"
)

JSON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_1_temporal_analysis.json"
)


# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def parse_frame_number(filename):

    """
    Extract the final numeric portion of the image filename.

    Examples:

        0.jpg      -> 0
        639.jpg    -> 639
        1234.jpg   -> 1234

    The frame number is only used inside its own sequence.
    """

    if pd.isna(filename):
        return np.nan

    name = os.path.basename(str(filename))

    stem = os.path.splitext(name)[0]

    match = re.search(r"(\d+)$", stem)

    if match:

        return int(match.group(1))

    return np.nan


def safe_corr(x, y):

    """
    Pearson correlation.
    """

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]

    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def safe_spearman(x, y):

    """
    Spearman correlation using pandas ranks.
    """

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]

    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    xr = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy()
    )

    yr = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy()
    )

    return safe_corr(xr, yr)


def linear_slope(x, y):

    """
    Least-squares linear slope.
    """

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]

    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    try:

        slope = np.polyfit(
            x,
            y,
            1
        )[0]

        return float(slope)

    except Exception:

        return np.nan


def monotonicity_score(values):

    """
    Directional consistency.

    +1 = all changes increasing
     0 = mixed/no directional change
    -1 = all changes decreasing
    """

    values = np.asarray(
        values,
        dtype=float
    )

    valid = np.isfinite(values)

    values = values[valid]

    if len(values) < 2:
        return np.nan

    differences = np.diff(values)

    positive = np.sum(
        differences > 0
    )

    negative = np.sum(
        differences < 0
    )

    total = positive + negative

    if total == 0:

        return 0.0

    return float(
        (positive - negative)
        / total
    )


def normalized_volatility(values):

    """
    Mean absolute frame-to-frame change divided by
    mean absolute signal magnitude.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    valid = np.isfinite(values)

    values = values[valid]

    if len(values) < 2:
        return np.nan

    differences = np.diff(values)

    numerator = np.mean(
        np.abs(differences)
    )

    denominator = np.mean(
        np.abs(values)
    )

    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


def trend_label(
    slope,
    correlation
):

    """
    Conservative trend label.
    """

    if (
        not np.isfinite(slope)
        or
        not np.isfinite(correlation)
    ):

        return "INSUFFICIENT_DATA"

    if correlation >= 0.30:

        return "INCREASING"

    if correlation <= -0.30:

        return "DECREASING"

    return "WEAK_OR_MIXED"


def progression_label(
    spearman,
    monotonicity,
    start_value,
    end_value,
    volatility
):

    """
    Conservative interpretation of crack-area progression.

    This is NOT a physical-damage ground truth label.
    """

    if not np.isfinite(spearman):

        return "INSUFFICIENT_DATA"

    if (
        spearman >= 0.50
        and
        monotonicity >= 0.50
        and
        end_value > start_value
    ):

        if (
            np.isfinite(volatility)
            and
            volatility < 0.50
        ):

            return "STRONG_INCREASING_SIGNAL"

        return "INCREASING_BUT_VARIABLE"

    if (
        spearman <= -0.50
        and
        monotonicity <= -0.50
        and
        end_value < start_value
    ):

        if (
            np.isfinite(volatility)
            and
            volatility < 0.50
        ):

            return "STRONG_DECREASING_SIGNAL"

        return "DECREASING_BUT_VARIABLE"

    return "MIXED_OR_NON_MONOTONIC"


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.1 - SEQUENTIAL / TEMPORAL DAMAGE ANALYSIS")
print("=" * 70)
print("")


# ------------------------------------------------------------
# CHECK INPUT FILES
# ------------------------------------------------------------

if not os.path.exists(FEATURE_FILE):

    print("ERROR: Stage 1.0 feature file not found:")

    print(FEATURE_FILE)

    raise SystemExit(1)


if not os.path.exists(MANIFEST_FILE):

    print("ERROR: Canonical manifest not found:")

    print(MANIFEST_FILE)

    raise SystemExit(1)


# ------------------------------------------------------------
# LOAD STAGE 1.0 FEATURES
# ------------------------------------------------------------

print("Loading Stage 1.0 feature file...")

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature rows:",
    len(features)
)


# ------------------------------------------------------------
# SHOW ACTUAL COLUMNS
# ------------------------------------------------------------

print("")
print("Stage 1.0 feature columns:")

for column in features.columns:

    print(
        "  -",
        column
    )


# ------------------------------------------------------------
# LOAD MANIFEST
# ------------------------------------------------------------

print("")
print("Loading canonical manifest...")

manifest = pd.read_csv(
    MANIFEST_FILE
)

print(
    "Manifest rows:",
    len(manifest)
)


# ------------------------------------------------------------
# REQUIRED CORE COLUMNS
#
# Only require columns that are actually needed.
# Do NOT assume optional geometry columns exist.
# ------------------------------------------------------------

required_columns = [
    "record_id",
    "split",
    "component",
    "sequence",
    "original_image_filename",
    "source_image",
    "crack_count",
    "total_crack_area_px",
    "crack_area_ratio"
]


missing_columns = [
    column
    for column in required_columns
    if column not in features.columns
]


if missing_columns:

    print("")
    print("ERROR: Required Stage 1.0 columns are missing:")

    for column in missing_columns:

        print(
            "  -",
            column
        )

    print("")
    print(
        "The script was stopped because the Stage 1.0 "
        "feature schema does not match the expected core fields."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# OPTIONAL FEATURES
#
# These are used only if they actually exist in Stage 1.0.
# ------------------------------------------------------------

optional_metrics = [
    "largest_crack_area_px",
    "smallest_crack_area_px",
    "mean_crack_area_px",
    "median_crack_area_px",
    "total_perimeter_px",
    "mean_perimeter_px",
    "max_perimeter_px",
    "mean_bbox_width_px",
    "mean_bbox_height_px",
    "mean_bbox_area_px",
    "max_bbox_width_px",
    "max_bbox_height_px",
    "max_bbox_area_px",
    "mean_centroid_x_norm",
    "mean_centroid_y_norm",
    "centroid_std_x_norm",
    "centroid_std_y_norm",
    "union_bbox_width_px",
    "union_bbox_height_px",
    "image_area_px",
    "image_width",
    "image_height",
    "json_width",
    "json_height",
    "annotation_shape_count",
    "valid_crack_polygon_count",
    "invalid_shape_count"
]


available_optional_metrics = [
    metric
    for metric in optional_metrics
    if metric in features.columns
]


print("")
print("Optional temporal metrics available:")

for metric in available_optional_metrics:

    print(
        "  -",
        metric
    )


if not available_optional_metrics:

    print(
        "  None"
    )


# ------------------------------------------------------------
# FINAL METRIC LIST
# ------------------------------------------------------------

metrics = [
    "crack_count",
    "total_crack_area_px",
    "crack_area_ratio"
]

for metric in available_optional_metrics:

    if metric not in metrics:

        metrics.append(metric)


print("")
print("Total temporal metrics:", len(metrics))


# ------------------------------------------------------------
# FRAME NUMBER EXTRACTION
# ------------------------------------------------------------

print("")
print("Constructing sequence-aware frame ordering...")


features["frame_number"] = (
    features[
        "original_image_filename"
    ]
    .apply(parse_frame_number)
)


missing_frame_numbers = int(
    features[
        "frame_number"
    ]
    .isna()
    .sum()
)


print(
    "Records without numeric frame number:",
    missing_frame_numbers
)


# ------------------------------------------------------------
# IMPORTANT:
# DO NOT sort globally by filename.
#
# Sort by:
#   sequence
#   frame_number
#   filename
# ------------------------------------------------------------

features = features.sort_values(
    by=[
        "sequence",
        "frame_number",
        "original_image_filename"
    ],
    kind="stable"
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# PROCESS SEQUENCES
# ------------------------------------------------------------

sequences = sorted(
    features[
        "sequence"
    ]
    .dropna()
    .unique()
)


print("")
print(
    "Sequences detected:",
    len(sequences)
)

print("")


frame_records = []

sequence_summaries = []


for sequence_index, sequence in enumerate(
    sequences,
    start=1
):

    sequence_df = features[
        features[
            "sequence"
        ] == sequence
    ].copy()


    # --------------------------------------------------------
    # Remove frames for which frame number could not be found
    # --------------------------------------------------------

    sequence_df = sequence_df[
        np.isfinite(
            sequence_df[
                "frame_number"
            ]
        )
    ].copy()


    if sequence_df.empty:

        continue


    sequence_df = sequence_df.sort_values(
        by="frame_number",
        kind="stable"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Frame information
    # --------------------------------------------------------

    frame_numbers = (
        sequence_df[
            "frame_number"
        ]
        .to_numpy(
            dtype=float
        )
    )


    frame_gaps = np.diff(
        frame_numbers
    )


    if len(frame_gaps) > 0:

        consecutive_fraction = float(
            np.mean(
                frame_gaps == 1
            )
        )

        minimum_gap = int(
            np.min(frame_gaps)
        )

        maximum_gap = int(
            np.max(frame_gaps)
        )

    else:

        consecutive_fraction = np.nan

        minimum_gap = 0

        maximum_gap = 0


    # --------------------------------------------------------
    # FRAME-LEVEL OUTPUT
    # --------------------------------------------------------

    frame_output = sequence_df[
        [
            "record_id",
            "split",
            "component",
            "sequence",
            "original_image_filename",
            "source_image",
            "frame_number"
        ]
    ].copy()


    frame_output[
        "sequence_frame_index"
    ] = np.arange(
        len(frame_output)
    )


    frame_output[
        "frame_gap"
    ] = frame_output[
        "frame_number"
    ].diff()


    frame_output[
        "is_consecutive_frame"
    ] = (
        frame_output[
            "frame_gap"
        ] == 1
    )


    # --------------------------------------------------------
    # FRAME-TO-FRAME CHANGES
    # --------------------------------------------------------

    for metric in metrics:

        values = (
            pd.to_numeric(
                sequence_df[
                    metric
                ],
                errors="coerce"
            )
            .to_numpy(
                dtype=float
            )
        )


        previous_values = np.roll(
            values,
            1
        )


        previous_values[0] = np.nan


        absolute_change = (
            values
            -
            previous_values
        )


        relative_change = np.full(
            len(values),
            np.nan,
            dtype=float
        )


        valid_previous = (
            np.isfinite(
                previous_values
            )
            &
            np.isfinite(
                values
            )
            &
            (
                np.abs(
                    previous_values
                )
                >
                1e-12
            )
        )


        relative_change[
            valid_previous
        ] = (
            (
                values[
                    valid_previous
                ]
                -
                previous_values[
                    valid_previous
                ]
            )
            /
            np.abs(
                previous_values[
                    valid_previous
                ]
            )
        )


        frame_output[
            metric + "_change"
        ] = absolute_change


        frame_output[
            metric + "_relative_change"
        ] = relative_change


    frame_records.append(
        frame_output
    )


    # --------------------------------------------------------
    # SEQUENCE SUMMARY
    # --------------------------------------------------------

    summary = {

        "sequence": sequence,

        "split": sequence_df[
            "split"
        ].iloc[0],

        "component": int(
            sequence_df[
                "component"
            ].iloc[0]
        ),

        "frame_count": int(
            len(sequence_df)
        ),

        "first_frame_number": int(
            frame_numbers[0]
        ),

        "last_frame_number": int(
            frame_numbers[-1]
        ),

        "consecutive_frame_fraction":
            consecutive_fraction,

        "minimum_frame_gap":
            minimum_gap,

        "maximum_frame_gap":
            maximum_gap
    }


    # --------------------------------------------------------
    # TEMPORAL STATISTICS FOR EVERY AVAILABLE METRIC
    # --------------------------------------------------------

    for metric in metrics:

        values = (
            pd.to_numeric(
                sequence_df[
                    metric
                ],
                errors="coerce"
            )
            .to_numpy(
                dtype=float
            )
        )


        valid_values = values[
            np.isfinite(values)
        ]


        if len(valid_values) == 0:

            summary[
                metric + "_mean"
            ] = np.nan

            summary[
                metric + "_std"
            ] = np.nan

            summary[
                metric + "_min"
            ] = np.nan

            summary[
                metric + "_max"
            ] = np.nan

            summary[
                metric + "_start"
            ] = np.nan

            summary[
                metric + "_end"
            ] = np.nan

            summary[
                metric + "_absolute_change_mean"
            ] = np.nan

            summary[
                metric + "_relative_change_mean"
            ] = np.nan

            summary[
                metric + "_slope"
            ] = np.nan

            summary[
                metric + "_pearson"
            ] = np.nan

            summary[
                metric + "_spearman"
            ] = np.nan

            summary[
                metric + "_monotonicity"
            ] = np.nan

            summary[
                metric + "_volatility"
            ] = np.nan

            summary[
                metric + "_trend"
            ] = "INSUFFICIENT_DATA"

            continue


        summary[
            metric + "_mean"
        ] = float(
            np.mean(
                valid_values
            )
        )


        summary[
            metric + "_std"
        ] = float(
            np.std(
                valid_values
            )
        )


        summary[
            metric + "_min"
        ] = float(
            np.min(
                valid_values
            )
        )


        summary[
            metric + "_max"
        ] = float(
            np.max(
                valid_values
            )
        )


        summary[
            metric + "_start"
        ] = safe_float(
            values[0]
        )


        summary[
            metric + "_end"
        ] = safe_float(
            values[-1]
        )


        # ----------------------------------------------------
        # Absolute frame-to-frame change
        # ----------------------------------------------------

        differences = np.diff(
            values
        )

        differences = differences[
            np.isfinite(
                differences
            )
        ]


        if len(differences) > 0:

            summary[
                metric + "_absolute_change_mean"
            ] = float(
                np.mean(
                    np.abs(
                        differences
                    )
                )
            )

        else:

            summary[
                metric + "_absolute_change_mean"
            ] = np.nan


        # ----------------------------------------------------
        # Relative frame-to-frame change
        # ----------------------------------------------------

        relative_changes = []


        for i in range(
            1,
            len(values)
        ):

            previous = values[
                i - 1
            ]

            current = values[
                i
            ]


            if (
                np.isfinite(previous)
                and
                np.isfinite(current)
                and
                abs(previous) > 1e-12
            ):

                relative_changes.append(
                    abs(
                        current - previous
                    )
                    /
                    abs(previous)
                )


        if relative_changes:

            summary[
                metric + "_relative_change_mean"
            ] = float(
                np.mean(
                    relative_changes
                )
            )

        else:

            summary[
                metric + "_relative_change_mean"
            ] = np.nan


        # ----------------------------------------------------
        # Temporal trend
        # ----------------------------------------------------

        temporal_index = np.arange(
            len(values),
            dtype=float
        )


        summary[
            metric + "_slope"
        ] = linear_slope(
            temporal_index,
            values
        )


        summary[
            metric + "_pearson"
        ] = safe_corr(
            temporal_index,
            values
        )


        summary[
            metric + "_spearman"
        ] = safe_spearman(
            temporal_index,
            values
        )


        summary[
            metric + "_monotonicity"
        ] = monotonicity_score(
            values
        )


        summary[
            metric + "_volatility"
        ] = normalized_volatility(
            values
        )


        summary[
            metric + "_trend"
        ] = trend_label(
            summary[
                metric + "_slope"
            ],
            summary[
                metric + "_spearman"
            ]
        )


    # --------------------------------------------------------
    # SPECIAL DAMAGE-PROGRESSION ANALYSIS
    #
    # Primary physical-damage feature:
    #     crack_area_ratio
    # --------------------------------------------------------

    damage_values = (
        pd.to_numeric(
            sequence_df[
                "crack_area_ratio"
            ],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )


    damage_spearman = safe_spearman(
        np.arange(
            len(damage_values)
        ),
        damage_values
    )


    damage_monotonicity = (
        monotonicity_score(
            damage_values
        )
    )


    damage_volatility = (
        normalized_volatility(
            damage_values
        )
    )


    damage_start = safe_float(
        damage_values[0]
    )


    damage_end = safe_float(
        damage_values[-1]
    )


    summary[
        "damage_area_absolute_change"
    ] = (
        damage_end
        -
        damage_start
        if (
            np.isfinite(
                damage_start
            )
            and
            np.isfinite(
                damage_end
            )
        )
        else np.nan
    )


    if (
        np.isfinite(
            damage_start
        )
        and
        abs(
            damage_start
        ) > 1e-12
        and
        np.isfinite(
            damage_end
        )
    ):

        summary[
            "damage_area_relative_change"
        ] = float(
            (
                damage_end
                -
                damage_start
            )
            /
            damage_start
        )

    else:

        summary[
            "damage_area_relative_change"
        ] = np.nan


    summary[
        "damage_progression_class"
    ] = progression_label(
        damage_spearman,
        damage_monotonicity,
        damage_start,
        damage_end,
        damage_volatility
    )


    sequence_summaries.append(
        summary
    )


    if (
        sequence_index % 5 == 0
        or
        sequence_index == len(sequences)
    ):

        print(
            "Processed sequences:",
            sequence_index,
            "/",
            len(sequences)
        )


# ------------------------------------------------------------
# COMBINE FRAME RESULTS
# ------------------------------------------------------------

if frame_records:

    frame_analysis = pd.concat(
        frame_records,
        ignore_index=True
    )

else:

    frame_analysis = pd.DataFrame()


summary_df = pd.DataFrame(
    sequence_summaries
)


# ------------------------------------------------------------
# SORT RESULTS
# ------------------------------------------------------------

if not frame_analysis.empty:

    frame_analysis = frame_analysis.sort_values(
        by=[
            "sequence",
            "frame_number"
        ],
        kind="stable"
    ).reset_index(
        drop=True
    )


if not summary_df.empty:

    summary_df = summary_df.sort_values(
        by="sequence",
        kind="stable"
    ).reset_index(
        drop=True
    )


# ------------------------------------------------------------
# SAVE FRAME ANALYSIS
# ------------------------------------------------------------

print("")
print("Saving frame-level temporal analysis...")

frame_analysis.to_csv(
    FRAME_OUTPUT,
    index=False
)

print(
    "Saved:",
    FRAME_OUTPUT
)


# ------------------------------------------------------------
# SAVE SEQUENCE SUMMARY
# ------------------------------------------------------------

print("")
print("Saving sequence-level summary...")

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print(
    "Saved:",
    SUMMARY_OUTPUT
)


# ------------------------------------------------------------
# OVERALL STATISTICS
# ------------------------------------------------------------

overall = {}


if not summary_df.empty:

    progression_counts = (
        summary_df[
            "damage_progression_class"
        ]
        .value_counts()
        .to_dict()
    )


    overall[
        "progression_class_counts"
    ] = {
        str(key): int(value)
        for key, value
        in progression_counts.items()
    }


    overall[
        "mean_damage_area_spearman"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_spearman"
        ].mean()
    )


    overall[
        "median_damage_area_spearman"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_spearman"
        ].median()
    )


    overall[
        "mean_damage_area_monotonicity"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_monotonicity"
        ].mean()
    )


    overall[
        "median_damage_area_monotonicity"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_monotonicity"
        ].median()
    )


    overall[
        "mean_damage_area_volatility"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_volatility"
        ].mean()
    )


    overall[
        "median_damage_area_volatility"
    ] = safe_float(
        summary_df[
            "crack_area_ratio_volatility"
        ].median()
    )


    overall[
        "increasing_sequences"
    ] = int(
        summary_df[
            "damage_progression_class"
        ]
        .astype(str)
        .str.startswith(
            "STRONG_INCREASING"
        )
        .sum()
    )


    overall[
        "decreasing_sequences"
    ] = int(
        summary_df[
            "damage_progression_class"
        ]
        .astype(str)
        .str.startswith(
            "STRONG_DECREASING"
        )
        .sum()
    )


    overall[
        "mixed_sequences"
    ] = int(
        (
            summary_df[
                "damage_progression_class"
            ]
            ==
            "MIXED_OR_NON_MONOTONIC"
        ).sum()
    )


    overall[
        "variable_increasing_sequences"
    ] = int(
        (
            summary_df[
                "damage_progression_class"
            ]
            ==
            "INCREASING_BUT_VARIABLE"
        ).sum()
    )


    overall[
        "variable_decreasing_sequences"
    ] = int(
        (
            summary_df[
                "damage_progression_class"
            ]
            ==
            "DECREASING_BUT_VARIABLE"
        ).sum()
    )


else:

    overall[
        "progression_class_counts"
    ] = {}


# ------------------------------------------------------------
# BUILD JSON REPORT
# ------------------------------------------------------------

report = {

    "stage": "1.1",

    "title":
        "Sequential / Temporal Damage Analysis",

    "status":
        "PASS",

    "purpose": [

        "Analyze physical damage features across ordered frames.",

        "Measure frame-to-frame temporal changes.",

        "Measure temporal trends.",

        "Measure monotonicity and volatility.",

        "Determine whether damage features show temporal signal.",

        "Avoid treating temporal correlation as proof of physical damage progression."
    ],


    "input": {

        "feature_file":
            FEATURE_FILE,

        "manifest_file":
            MANIFEST_FILE,

        "feature_records":
            int(len(features))
    },


    "dataset": {

        "sequence_count":
            int(len(sequences)),

        "frame_records_analyzed":
            int(len(frame_analysis)),

        "sequence_summaries":
            int(len(summary_df))
    },


    "available_metrics":
        metrics,


    "optional_metrics_found":
        available_optional_metrics,


    "overall_results":
        overall,


    "interpretation": {

        "important_warning":
            "Temporal changes in annotation-derived features do not automatically prove physical conveyor-belt damage progression. Camera movement, viewpoint, illumination, annotation variation, and other factors may contribute to observed changes.",

        "primary_damage_signal":
            "crack_area_ratio",

        "next_decision":
            "Use Stage 1.1 results to determine whether the dataset contains sufficient temporal signal for sequential damage prediction before constructing prediction targets and training models."
    },


    "outputs": {

        "frame_analysis":
            FRAME_OUTPUT,

        "sequence_summary":
            SUMMARY_OUTPUT,

        "report":
            JSON_OUTPUT
    }
}


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2,
        allow_nan=False
    )


# ------------------------------------------------------------
# FINAL CONSOLE REPORT
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.1 SUMMARY")
print("=" * 70)

print(
    "Sequences analyzed:",
    len(summary_df)
)

print(
    "Frame records analyzed:",
    len(frame_analysis)
)


if not summary_df.empty:

    print("")
    print(
        "Damage progression classes:"
    )


    progression_counts = (
        summary_df[
            "damage_progression_class"
        ]
        .value_counts()
    )


    for label, count in progression_counts.items():

        print(
            "  ",
            label,
            ":",
            int(count)
        )


    print("")


    mean_spearman = (
        summary_df[
            "crack_area_ratio_spearman"
        ].mean()
    )


    median_spearman = (
        summary_df[
            "crack_area_ratio_spearman"
        ].median()
    )


    mean_monotonicity = (
        summary_df[
            "crack_area_ratio_monotonicity"
        ].mean()
    )


    mean_volatility = (
        summary_df[
            "crack_area_ratio_volatility"
        ].mean()
    )


    print(
        "Mean crack-area Spearman:",
        round(
            float(mean_spearman),
            4
        )
    )


    print(
        "Median crack-area Spearman:",
        round(
            float(median_spearman),
            4
        )
    )


    print(
        "Mean crack-area monotonicity:",
        round(
            float(mean_monotonicity),
            4
        )
    )


    print(
        "Mean crack-area volatility:",
        round(
            float(mean_volatility),
            4
        )
    )


print("")
print("Frame analysis:")
print(FRAME_OUTPUT)

print("")
print("Sequence summary:")
print(SUMMARY_OUTPUT)

print("")
print("JSON report:")
print(JSON_OUTPUT)

print("")
print("FINAL STATUS: PASS")
print("")