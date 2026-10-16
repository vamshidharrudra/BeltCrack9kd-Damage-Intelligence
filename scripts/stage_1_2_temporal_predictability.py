import os
import json
import re
import math
import numpy as np
import pandas as pd


# ============================================================
# STAGE 1.2
# TEMPORAL PREDICTABILITY ANALYSIS
#
# Purpose:
#   Determine whether the current physical damage state
#   contains information about future damage states.
#
# Tested horizons:
#   t + 1
#   t + 3
#   t + 5
#   t + 10
#
# IMPORTANT:
#   No ML model is trained in this stage.
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

PAIR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_2_temporal_pairs.csv"
)

SEQUENCE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_2_sequence_predictability.csv"
)

OVERALL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_2_overall_predictability.csv"
)

JSON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_2_temporal_predictability.json"
)


# ------------------------------------------------------------
# TEMPORAL HORIZONS
# ------------------------------------------------------------

HORIZONS = [
    1,
    3,
    5,
    10
]


# ------------------------------------------------------------
# CORE FEATURES
# ------------------------------------------------------------

CORE_FEATURES = [
    "crack_count",
    "total_crack_area_px",
    "crack_area_ratio",
    "largest_crack_area_px",
    "largest_crack_area_ratio",
    "smallest_crack_area_px",
    "mean_crack_area_px",
    "median_crack_area_px",
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "max_crack_perimeter_px",
    "mean_crack_bbox_width_px",
    "mean_crack_bbox_height_px",
    "mean_crack_bbox_area_px",
    "max_crack_bbox_width_px",
    "max_crack_bbox_height_px",
    "mean_crack_centroid_x_px",
    "mean_crack_centroid_y_px",
    "mean_crack_centroid_x_norm",
    "mean_crack_centroid_y_norm",
    "crack_centroid_x_std_px",
    "crack_centroid_y_std_px",
    "crack_centroid_x_std_norm",
    "crack_centroid_y_std_norm",
    "crack_bbox_union_width_px",
    "crack_bbox_union_height_px"
]


# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def parse_frame_number(filename):

    if pd.isna(filename):
        return np.nan

    name = os.path.basename(
        str(filename)
    )

    stem = os.path.splitext(
        name
    )[0]

    match = re.search(
        r"(\d+)$",
        stem
    )

    if match:
        return int(match.group(1))

    return np.nan


def safe_float(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def pearson_correlation(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

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
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def spearman_correlation(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

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

    x_rank = (
        pd.Series(x)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    y_rank = (
        pd.Series(y)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    return pearson_correlation(
        x_rank,
        y_rank
    )


def mean_absolute_error(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                y[valid] - x[valid]
            )
        )
    )


def mean_relative_change(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        (np.abs(x) > 1e-12)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    y[valid] - x[valid]
                )
                /
                np.abs(
                    x[valid]
                )
            )
        )
    )


def direction_agreement(x, y):

    """
    Measures whether the future change tends to have
    the same direction as the immediate/current change.

    This is calculated only when enough values exist.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]

    y = y[valid]

    if len(x) < 2:
        return np.nan

    changes = y - x

    positive = np.sum(
        changes > 0
    )

    negative = np.sum(
        changes < 0
    )

    total = positive + negative

    if total == 0:
        return 0.0

    return float(
        (positive - negative)
        / total
    )


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.2 - TEMPORAL PREDICTABILITY ANALYSIS")
print("=" * 70)
print("")


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

if not os.path.exists(
    FEATURE_FILE
):

    print(
        "ERROR: Stage 1.0 feature file not found:"
    )

    print(
        FEATURE_FILE
    )

    raise SystemExit(1)


if not os.path.exists(
    MANIFEST_FILE
):

    print(
        "ERROR: Canonical manifest not found:"
    )

    print(
        MANIFEST_FILE
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# LOAD FEATURES
# ------------------------------------------------------------

print(
    "Loading Stage 1.0 features..."
)

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature records:",
    len(features)
)


# ------------------------------------------------------------
# LOAD MANIFEST
# ------------------------------------------------------------

print("")
print(
    "Loading canonical manifest..."
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

print(
    "Manifest records:",
    len(manifest)
)


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "record_id",
    "split",
    "component",
    "sequence",
    "original_image_filename",
    "source_image"
]


missing = [
    column
    for column in required_columns
    if column not in features.columns
]


if missing:

    print("")
    print(
        "ERROR: Missing required columns:"
    )

    for column in missing:
        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ------------------------------------------------------------
# DETERMINE AVAILABLE DAMAGE FEATURES
# ------------------------------------------------------------

available_features = []

for feature in CORE_FEATURES:

    if feature in features.columns:

        available_features.append(
            feature
        )


print("")
print(
    "Damage features available:",
    len(available_features)
)

for feature in available_features:

    print(
        "  -",
        feature
    )


if not available_features:

    print("")
    print(
        "ERROR: No damage features available."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# FRAME NUMBER
# ------------------------------------------------------------

print("")
print(
    "Extracting frame numbers..."
)

features[
    "frame_number"
] = (
    features[
        "original_image_filename"
    ]
    .apply(
        parse_frame_number
    )
)


missing_frame_numbers = int(
    features[
        "frame_number"
    ]
    .isna()
    .sum()
)


print(
    "Missing frame numbers:",
    missing_frame_numbers
)


if missing_frame_numbers > 0:

    print("")
    print(
        "WARNING: Records without frame numbers "
        "will be excluded from temporal pairing."
    )


# ------------------------------------------------------------
# SORT WITHIN SEQUENCE
# ------------------------------------------------------------

features = features[
    np.isfinite(
        features[
            "frame_number"
        ]
    )
].copy()


features = features.sort_values(
    by=[
        "sequence",
        "frame_number"
    ],
    kind="stable"
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# TEMPORAL PAIR RECORDS
# ------------------------------------------------------------

pair_records = []


sequences = sorted(
    features[
        "sequence"
    ]
    .dropna()
    .unique()
)


print("")
print(
    "Sequences:",
    len(sequences)
)

print(
    "Horizon values:",
    HORIZONS
)

print("")


# ------------------------------------------------------------
# PROCESS EACH SEQUENCE
# ------------------------------------------------------------

for sequence_index, sequence in enumerate(
    sequences,
    start=1
):

    seq = features[
        features[
            "sequence"
        ] == sequence
    ].copy()


    seq = seq.sort_values(
        by="frame_number",
        kind="stable"
    ).reset_index(
        drop=True
    )


    if len(seq) == 0:
        continue


    # --------------------------------------------------------
    # For every temporal horizon
    # --------------------------------------------------------

    for horizon in HORIZONS:

        if len(seq) <= horizon:
            continue


        current_rows = seq.iloc[
            :-horizon
        ].copy()


        future_rows = seq.iloc[
            horizon:
        ].copy()


        # ----------------------------------------------------
        # Verify actual frame gap
        # ----------------------------------------------------

        current_frames = (
            current_rows[
                "frame_number"
            ]
            .to_numpy(
                dtype=int
            )
        )


        future_frames = (
            future_rows[
                "frame_number"
            ]
            .to_numpy(
                dtype=int
            )
        )


        actual_gaps = (
            future_frames
            -
            current_frames
        )


        valid_gap = (
            actual_gaps == horizon
        )


        if not np.any(
            valid_gap
        ):

            continue


        current_rows = (
            current_rows
            .iloc[
                valid_gap
            ]
            .reset_index(
                drop=True
            )
        )


        future_rows = (
            future_rows
            .iloc[
                valid_gap
            ]
            .reset_index(
                drop=True
            )
        )


        # ----------------------------------------------------
        # Create one row per temporal pair
        # ----------------------------------------------------

        for row_index in range(
            len(current_rows)
        ):

            current = (
                current_rows.iloc[
                    row_index
                ]
            )

            future = (
                future_rows.iloc[
                    row_index
                ]
            )


            record = {

                "sequence":
                    sequence,

                "split":
                    current[
                        "split"
                    ],

                "component":
                    int(
                        current[
                            "component"
                        ]
                    ),

                "horizon":
                    int(
                        horizon
                    ),

                "current_frame":
                    int(
                        current[
                            "frame_number"
                        ]
                    ),

                "future_frame":
                    int(
                        future[
                            "frame_number"
                        ]
                    ),

                "current_record_id":
                    int(
                        current[
                            "record_id"
                        ]
                    ),

                "future_record_id":
                    int(
                        future[
                            "record_id"
                        ]
                    )
            }


            # ------------------------------------------------
            # Store feature values
            # ------------------------------------------------

            for feature in available_features:

                current_value = safe_float(
                    current[
                        feature
                    ]
                )

                future_value = safe_float(
                    future[
                        feature
                    ]
                )


                record[
                    feature + "_current"
                ] = current_value


                record[
                    feature + "_future"
                ] = future_value


                if (
                    np.isfinite(
                        current_value
                    )
                    and
                    np.isfinite(
                        future_value
                    )
                ):

                    record[
                        feature + "_future_change"
                    ] = (
                        future_value
                        -
                        current_value
                    )


                    if abs(
                        current_value
                    ) > 1e-12:

                        record[
                            feature
                            + "_future_relative_change"
                        ] = (
                            (
                                future_value
                                -
                                current_value
                            )
                            /
                            abs(
                                current_value
                            )
                        )

                    else:

                        record[
                            feature
                            + "_future_relative_change"
                        ] = np.nan

                else:

                    record[
                        feature + "_future_change"
                    ] = np.nan

                    record[
                        feature
                        + "_future_relative_change"
                    ] = np.nan


            pair_records.append(
                record
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
# CREATE PAIR DATAFRAME
# ------------------------------------------------------------

pair_df = pd.DataFrame(
    pair_records
)


if pair_df.empty:

    print("")
    print(
        "ERROR: No temporal pairs were generated."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# SAVE PAIR DATA
# ------------------------------------------------------------

print("")
print(
    "Temporal pairs generated:",
    len(pair_df)
)


pair_df.to_csv(
    PAIR_OUTPUT,
    index=False
)


print(
    "Saved:",
    PAIR_OUTPUT
)


# ------------------------------------------------------------
# SEQUENCE-LEVEL PREDICTABILITY
# ------------------------------------------------------------

sequence_records = []


for sequence in sequences:

    seq = pair_df[
        pair_df[
            "sequence"
        ] == sequence
    ]


    if seq.empty:
        continue


    for horizon in HORIZONS:

        horizon_df = seq[
            seq[
                "horizon"
            ] == horizon
        ]


        if horizon_df.empty:
            continue


        for feature in available_features:

            current_column = (
                feature
                + "_current"
            )

            future_column = (
                feature
                + "_future"
            )


            x = pd.to_numeric(
                horizon_df[
                    current_column
                ],
                errors="coerce"
            ).to_numpy(
                dtype=float
            )


            y = pd.to_numeric(
                horizon_df[
                    future_column
                ],
                errors="coerce"
            ).to_numpy(
                dtype=float
            )


            valid = (
                np.isfinite(x)
                &
                np.isfinite(y)
            )


            x_valid = x[
                valid
            ]

            y_valid = y[
                valid
            ]


            sequence_record = {

                "sequence":
                    sequence,

                "split":
                    horizon_df[
                        "split"
                    ].iloc[0],

                "component":
                    int(
                        horizon_df[
                            "component"
                        ].iloc[0]
                    ),

                "horizon":
                    int(
                        horizon
                    ),

                "feature":
                    feature,

                "pair_count":
                    int(
                        len(x_valid)
                    ),

                "pearson":
                    pearson_correlation(
                        x_valid,
                        y_valid
                    ),

                "spearman":
                    spearman_correlation(
                        x_valid,
                        y_valid
                    ),

                "mean_absolute_future_change":
                    mean_absolute_error(
                        x_valid,
                        y_valid
                    ),

                "mean_relative_future_change":
                    mean_relative_change(
                        x_valid,
                        y_valid
                    ),

                "future_change_direction":
                    direction_agreement(
                        x_valid,
                        y_valid
                    )
            }


            sequence_records.append(
                sequence_record
            )


sequence_df = pd.DataFrame(
    sequence_records
)


# ------------------------------------------------------------
# SAVE SEQUENCE RESULTS
# ------------------------------------------------------------

sequence_df.to_csv(
    SEQUENCE_OUTPUT,
    index=False
)


print("")
print(
    "Sequence-level predictability saved:"
)

print(
    SEQUENCE_OUTPUT
)


# ------------------------------------------------------------
# OVERALL RESULTS
# ------------------------------------------------------------

overall_records = []


for horizon in HORIZONS:

    horizon_data = sequence_df[
        sequence_df[
            "horizon"
        ] == horizon
    ]


    if horizon_data.empty:
        continue


    for feature in available_features:

        data = horizon_data[
            horizon_data[
                "feature"
            ] == feature
        ]


        if data.empty:
            continue


        pearson_values = pd.to_numeric(
            data[
                "pearson"
            ],
            errors="coerce"
        )


        spearman_values = pd.to_numeric(
            data[
                "spearman"
            ],
            errors="coerce"
        )


        valid_pearson = (
            pearson_values
            .dropna()
        )


        valid_spearman = (
            spearman_values
            .dropna()
        )


        # ----------------------------------------------------
        # Sequence-level median correlations
        # ----------------------------------------------------

        record = {

            "horizon":
                int(
                    horizon
                ),

            "feature":
                feature,

            "sequences_analyzed":
                int(
                    len(data)
                ),

            "sequences_with_valid_pearson":
                int(
                    len(
                        valid_pearson
                    )
                ),

            "sequences_with_valid_spearman":
                int(
                    len(
                        valid_spearman
                    )
                ),

            "mean_pearson":
                safe_float(
                    valid_pearson.mean()
                    if len(
                        valid_pearson
                    ) > 0
                    else np.nan
                ),

            "median_pearson":
                safe_float(
                    valid_pearson.median()
                    if len(
                        valid_pearson
                    ) > 0
                    else np.nan
                ),

            "mean_spearman":
                safe_float(
                    valid_spearman.mean()
                    if len(
                        valid_spearman
                    ) > 0
                    else np.nan
                ),

            "median_spearman":
                safe_float(
                    valid_spearman.median()
                    if len(
                        valid_spearman
                    ) > 0
                    else np.nan
                ),

            "mean_absolute_future_change":
                safe_float(
                    data[
                        "mean_absolute_future_change"
                    ].mean()
                ),

            "mean_relative_future_change":
                safe_float(
                    data[
                        "mean_relative_future_change"
                    ].mean()
                )
        }


        # ----------------------------------------------------
        # Count useful temporal relationships
        # ----------------------------------------------------

        record[
            "strong_positive_sequences"
        ] = int(
            (
                valid_spearman >= 0.70
            ).sum()
        )


        record[
            "moderate_positive_sequences"
        ] = int(
            (
                (
                    valid_spearman >= 0.40
                )
                &
                (
                    valid_spearman < 0.70
                )
            ).sum()
        )


        record[
            "weak_or_mixed_sequences"
        ] = int(
            (
                valid_spearman < 0.40
            ).sum()
        )


        overall_records.append(
            record
        )


overall_df = pd.DataFrame(
    overall_records
)


# ------------------------------------------------------------
# SAVE OVERALL CSV
# ------------------------------------------------------------

overall_df.to_csv(
    OVERALL_OUTPUT,
    index=False
)


print("")
print(
    "Overall predictability saved:"
)

print(
    OVERALL_OUTPUT
)


# ------------------------------------------------------------
# PRIMARY DAMAGE FEATURE SUMMARY
# ------------------------------------------------------------

damage_feature = "crack_area_ratio"


damage_summary = []


if damage_feature in available_features:

    damage_data = overall_df[
        overall_df[
            "feature"
        ] == damage_feature
    ].copy()


    for _, row in damage_data.iterrows():

        damage_summary.append({

            "horizon":
                int(
                    row[
                        "horizon"
                    ]
                ),

            "mean_spearman":
                safe_float(
                    row[
                        "mean_spearman"
                    ]
                ),

            "median_spearman":
                safe_float(
                    row[
                        "median_spearman"
                    ]
                ),

            "strong_positive_sequences":
                int(
                    row[
                        "strong_positive_sequences"
                    ]
                ),

            "moderate_positive_sequences":
                int(
                    row[
                        "moderate_positive_sequences"
                    ]
                ),

            "weak_or_mixed_sequences":
                int(
                    row[
                        "weak_or_mixed_sequences"
                    ]
                )
        })


# ------------------------------------------------------------
# DECISION LOGIC
# ------------------------------------------------------------

decision = "UNDETERMINED"


if damage_summary:

    max_median_spearman = max(
        [
            item[
                "median_spearman"
            ]
            for item in damage_summary
            if np.isfinite(
                item[
                    "median_spearman"
                ]
            )
        ],
        default=np.nan
    )


    max_mean_spearman = max(
        [
            item[
                "mean_spearman"
            ]
            for item in damage_summary
            if np.isfinite(
                item[
                    "mean_spearman"
                ]
            )
        ],
        default=np.nan
    )


    # --------------------------------------------------------
    # Conservative thresholds
    # --------------------------------------------------------

    if (
        np.isfinite(
            max_median_spearman
        )
        and
        max_median_spearman >= 0.70
    ):

        decision = (
            "STRONG_TEMPORAL_PREDICTABILITY"
        )

    elif (
        np.isfinite(
            max_median_spearman
        )
        and
        max_median_spearman >= 0.40
    ):

        decision = (
            "MODERATE_TEMPORAL_PREDICTABILITY"
        )

    else:

        decision = (
            "WEAK_TEMPORAL_PREDICTABILITY"
        )


# ------------------------------------------------------------
# JSON REPORT
# ------------------------------------------------------------

report = {

    "stage":
        "1.2",

    "title":
        "Temporal Predictability Analysis",

    "status":
        "PASS",

    "purpose": [

        "Determine whether current damage features contain information about future damage features.",

        "Evaluate multiple future horizons.",

        "Keep temporal pairs strictly inside each sequence.",

        "Avoid training a predictive model before temporal predictability is established."
    ],


    "input": {

        "feature_file":
            FEATURE_FILE,

        "manifest_file":
            MANIFEST_FILE,

        "feature_records":
            int(
                len(features)
            )
    },


    "dataset": {

        "sequences":
            int(
                len(sequences)
            ),

        "available_features":
            available_features,

        "horizons":
            HORIZONS,

        "temporal_pairs":
            int(
                len(pair_df)
            )
    },


    "primary_damage_feature":
        damage_feature,


    "damage_area_results":
        damage_summary,


    "decision":
        decision,


    "interpretation": {

        "strong":
            "Median sequence-level Spearman correlation >= 0.70 at at least one tested horizon indicates strong short-term temporal predictability.",

        "moderate":
            "Median sequence-level Spearman correlation >= 0.40 but below 0.70 indicates moderate temporal predictability.",

        "weak":
            "Median sequence-level Spearman correlation below 0.40 indicates weak temporal predictability.",

        "warning":
            "Predictability of annotation-derived damage features does not by itself prove that the model will predict future physical damage in a real conveyor belt."
    },


    "outputs": {

        "temporal_pairs":
            PAIR_OUTPUT,

        "sequence_predictability":
            SEQUENCE_OUTPUT,

        "overall_predictability":
            OVERALL_OUTPUT,

        "json_report":
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
# FINAL TERMINAL OUTPUT
# ------------------------------------------------------------

print("")
print("=" * 70)
print("STAGE 1.2 SUMMARY")
print("=" * 70)

print(
    "Sequences analyzed:",
    len(sequences)
)

print(
    "Temporal pairs:",
    len(pair_df)
)

print(
    "Features analyzed:",
    len(available_features)
)

print(
    "Horizons:",
    HORIZONS
)

print("")


if damage_summary:

    print(
        "PRIMARY FEATURE:",
        damage_feature
    )

    print("")

    for item in damage_summary:

        print(
            "Horizon t+"
            + str(
                item[
                    "horizon"
                ]
            )
            + ":"
        )

        print(
            "  Mean Spearman:",
            round(
                item[
                    "mean_spearman"
                ],
                4
            )
        )

        print(
            "  Median Spearman:",
            round(
                item[
                    "median_spearman"
                ],
                4
            )
        )

        print(
            "  Strong positive sequences:",
            item[
                "strong_positive_sequences"
            ]
        )

        print(
            "  Moderate positive sequences:",
            item[
                "moderate_positive_sequences"
            ]
        )

        print(
            "  Weak/mixed sequences:",
            item[
                "weak_or_mixed_sequences"
            ]
        )

        print("")


print(
    "DECISION:",
    decision
)

print("")

print(
    "Temporal pair file:"
)

print(
    PAIR_OUTPUT
)

print("")

print(
    "Sequence predictability file:"
)

print(
    SEQUENCE_OUTPUT
)

print("")

print(
    "Overall predictability file:"
)

print(
    OVERALL_OUTPUT
)

print("")

print(
    "JSON report:"
)

print(
    JSON_OUTPUT
)

print("")
print("FINAL STATUS: PASS")
print("")