import os
import json
import numpy as np
import pandas as pd

from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

RAW_DATASET_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd"

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

RAPID_CASES = os.path.join(
    AUDIT_DIR,
    "stage_2_6_rapid_damage_cases.csv"
)

VALIDATION_CSV = os.path.join(
    AUDIT_DIR,
    "stage_2_7_extreme_transition_validation.csv"
)

OUTPUT_CASES = os.path.join(
    AUDIT_DIR,
    "stage_2_8_visual_transition_consistency.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_2_8_visual_transition_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_N = 30

RESIZE_WIDTH = 320
RESIZE_HEIGHT = 180


# ============================================================
# HELPERS
# ============================================================

def resolve_dataset_path(path_value):

    if path_value is None:
        return None

    path_value = str(
        path_value
    ).strip()

    if not path_value:
        return None

    if os.path.isabs(
        path_value
    ):

        return os.path.normpath(
            path_value
        )

    normalized = path_value.replace(
        "/",
        os.sep
    )

    return os.path.normpath(
        os.path.join(
            RAW_DATASET_ROOT,
            normalized
        )
    )


def load_gray_image(
    path
):

    image = Image.open(
        path
    ).convert(
        "L"
    )

    image = image.resize(
        (
            RESIZE_WIDTH,
            RESIZE_HEIGHT
        )
    )

    return np.asarray(
        image,
        dtype=np.float32
    )


def calculate_image_metrics(
    current,
    future
):

    current_flat = (
        current
        .reshape(-1)
    )

    future_flat = (
        future
        .reshape(-1)
    )

    current_mean = (
        current_flat.mean()
    )

    future_mean = (
        future_flat.mean()
    )

    current_std = (
        current_flat.std()
    )

    future_std = (
        future_flat.std()
    )

    # Pearson correlation.
    if (
        current_std > 1e-8
        and
        future_std > 1e-8
    ):

        correlation = float(
            np.corrcoef(
                current_flat,
                future_flat
            )[0, 1]
        )

    else:

        correlation = np.nan


    # Mean absolute pixel difference.
    mean_absolute_difference = float(
        np.mean(
            np.abs(
                current
                -
                future
            )
        )
    )


    # Normalized difference.
    normalized_difference = (
        mean_absolute_difference
        /
        255.0
    )


    # RMSE in pixel intensity.
    rmse = float(
        np.sqrt(
            np.mean(
                (
                    current
                    -
                    future
                )
                ** 2
            )
        )
    )


    # Structural-ish normalized difference.
    # This is not SSIM; it is a simple normalized
    # pixel difference for dataset auditing.
    denominator = (
        np.mean(
            np.abs(
                current
            )
        )
        +
        1e-8
    )

    relative_pixel_difference = float(
        mean_absolute_difference
        /
        denominator
    )


    return {

        "image_correlation":
            correlation,

        "mean_absolute_pixel_difference":
            mean_absolute_difference,

        "normalized_pixel_difference":
            normalized_difference,

        "pixel_rmse":
            rmse,

        "relative_pixel_difference":
            relative_pixel_difference,

        "current_mean_intensity":
            float(current_mean),

        "future_mean_intensity":
            float(future_mean),

        "current_intensity_std":
            float(current_std),

        "future_intensity_std":
            float(future_std)
    }


def classify_visual_similarity(
    correlation
):

    if pd.isna(
        correlation
    ):

        return "UNKNOWN"

    if correlation >= 0.95:

        return "VERY_HIGH_SIMILARITY"

    elif correlation >= 0.90:

        return "HIGH_SIMILARITY"

    elif correlation >= 0.75:

        return "MODERATE_SIMILARITY"

    else:

        return "LOW_SIMILARITY"


def classify_annotation_change(
    value
):

    value = abs(
        float(value)
    )

    if value < 0.10:

        return "LOW"

    elif value < 0.50:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 2.8 - VISUAL TRANSITION CONSISTENCY ANALYSIS"
)
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

if not os.path.exists(
    VALIDATION_CSV
):

    raise FileNotFoundError(
        "Stage 2.7 validation CSV not found:\n"
        +
        VALIDATION_CSV
    )


if not os.path.exists(
    RAPID_CASES
):

    raise FileNotFoundError(
        "Stage 2.6 rapid cases not found:\n"
        +
        RAPID_CASES
    )


# ============================================================
# LOAD STAGE 2.7
# ============================================================

print()
print(
    "Loading Stage 2.7 validation results..."
)

validation = pd.read_csv(
    VALIDATION_CSV
)

print(
    f"Validated events: "
    f"{len(validation)}"
)


if len(validation) == 0:

    raise ValueError(
        "Stage 2.7 contains zero validated events."
    )


# ============================================================
# SELECT TOP EVENTS
# ============================================================

validation = validation.sort_values(
    "absolute_damage_change",
    ascending=False
).head(
    TOP_N
).copy()


# ============================================================
# ANALYZE
# ============================================================

results = []

print()
print("=" * 70)
print(
    "ANALYZING IMAGE-TO-IMAGE TRANSITIONS"
)
print("=" * 70)


for _, row in validation.iterrows():

    current_path = resolve_dataset_path(
        row["current_image"]
    )

    future_path = resolve_dataset_path(
        row["future_image"]
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(
        current_path
    ):

        print(
            "WARNING: missing current image:"
        )

        print(
            current_path
        )

        continue


    if not os.path.exists(
        future_path
    ):

        print(
            "WARNING: missing future image:"
        )

        print(
            future_path
        )

        continue


    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    try:

        current_image = (
            load_gray_image(
                current_path
            )
        )

        future_image = (
            load_gray_image(
                future_path
            )
        )

    except Exception as exc:

        print(
            f"WARNING: image read failed: "
            f"{exc}"
        )

        continue


    # --------------------------------------------------------
    # IMAGE METRICS
    # --------------------------------------------------------

    metrics = calculate_image_metrics(
        current_image,
        future_image
    )


    # --------------------------------------------------------
    # ANNOTATION CHANGE
    # --------------------------------------------------------

    current_annotation_ratio = float(
        row[
            "current_annotation_area_ratio"
        ]
    )

    future_annotation_ratio = float(
        row[
            "future_annotation_area_ratio"
        ]
    )

    annotation_ratio_change = (
        future_annotation_ratio
        -
        current_annotation_ratio
    )

    absolute_annotation_ratio_change = abs(
        annotation_ratio_change
    )


    # --------------------------------------------------------
    # DAMAGE CHANGE
    # --------------------------------------------------------

    damage_change = float(
        row[
            "damage_change"
        ]
    )

    absolute_damage_change = abs(
        damage_change
    )


    # --------------------------------------------------------
    # RELATION BETWEEN VISUAL AND ANNOTATION CHANGE
    # --------------------------------------------------------

    visual_similarity = classify_visual_similarity(
        metrics[
            "image_correlation"
        ]
    )

    annotation_change_class = (
        classify_annotation_change(
            (
                absolute_annotation_ratio_change
                /
                (
                    abs(
                        current_annotation_ratio
                    )
                    +
                    1e-8
                )
            )
        )
    )


    # --------------------------------------------------------
    # HIGH-SIMILARITY / HIGH-ANNOTATION-CHANGE FLAG
    # --------------------------------------------------------

    suspicious_transition = (
        (
            not pd.isna(
                metrics[
                    "image_correlation"
                ]
            )
        )
        and
        metrics[
            "image_correlation"
        ] >= 0.95
        and
        annotation_change_class
        == "HIGH"
    )


    # --------------------------------------------------------
    # LOW-SIMILARITY / HIGH-CHANGE
    # --------------------------------------------------------

    visually_different_transition = (
        (
            not pd.isna(
                metrics[
                    "image_correlation"
                ]
            )
        )
        and
        metrics[
            "image_correlation"
        ] < 0.75
        and
        annotation_change_class
        == "HIGH"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "sequence":
            row["sequence"],

        "component":
            int(
                row["component"]
            ),

        "split":
            row["split"],

        "current_frame":
            int(
                row["current_frame"]
            ),

        "future_frame":
            int(
                row["future_frame"]
            ),

        "current_damage":
            float(
                row["current_damage"]
            ),

        "future_damage":
            float(
                row["future_damage"]
            ),

        "damage_change":
            damage_change,

        "absolute_damage_change":
            absolute_damage_change,

        "model_absolute_error":
            float(
                row[
                    "model_absolute_error"
                ]
            ),

        "current_polygon_count":
            int(
                row[
                    "current_polygon_count"
                ]
            ),

        "future_polygon_count":
            int(
                row[
                    "future_polygon_count"
                ]
            ),

        "polygon_count_change":
            int(
                row[
                    "polygon_count_change"
                ]
            ),

        "current_annotation_area_ratio":
            current_annotation_ratio,

        "future_annotation_area_ratio":
            future_annotation_ratio,

        "annotation_ratio_change":
            annotation_ratio_change,

        "absolute_annotation_ratio_change":
            absolute_annotation_ratio_change,

        "image_correlation":
            metrics[
                "image_correlation"
            ],

        "mean_absolute_pixel_difference":
            metrics[
                "mean_absolute_pixel_difference"
            ],

        "normalized_pixel_difference":
            metrics[
                "normalized_pixel_difference"
            ],

        "pixel_rmse":
            metrics[
                "pixel_rmse"
            ],

        "relative_pixel_difference":
            metrics[
                "relative_pixel_difference"
            ],

        "current_mean_intensity":
            metrics[
                "current_mean_intensity"
            ],

        "future_mean_intensity":
            metrics[
                "future_mean_intensity"
            ],

        "current_intensity_std":
            metrics[
                "current_intensity_std"
            ],

        "future_intensity_std":
            metrics[
                "future_intensity_std"
            ],

        "visual_similarity_class":
            visual_similarity,

        "annotation_change_class":
            annotation_change_class,

        "high_similarity_high_annotation_change":
            suspicious_transition,

        "low_similarity_high_annotation_change":
            visually_different_transition,

        "current_image":
            current_path,

        "future_image":
            future_path
    }

    results.append(
        result
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if len(results_df) == 0:

    raise RuntimeError(
        "No visual transitions could be analyzed."
    )


# ============================================================
# SAVE CASE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CASES,
    index=False
)


# ============================================================
# SUMMARY COUNTS
# ============================================================

total = len(
    results_df
)

very_high = int(
    (
        results_df[
            "visual_similarity_class"
        ]
        ==
        "VERY_HIGH_SIMILARITY"
    ).sum()
)

high = int(
    (
        results_df[
            "visual_similarity_class"
        ]
        ==
        "HIGH_SIMILARITY"
    ).sum()
)

moderate = int(
    (
        results_df[
            "visual_similarity_class"
        ]
        ==
        "MODERATE_SIMILARITY"
    ).sum()
)

low = int(
    (
        results_df[
            "visual_similarity_class"
        ]
        ==
        "LOW_SIMILARITY"
    ).sum()
)

suspicious_count = int(
    results_df[
        "high_similarity_high_annotation_change"
    ].sum()
)

visually_different_count = int(
    results_df[
        "low_similarity_high_annotation_change"
    ].sum()
)


# ============================================================
# CORRELATIONS
# ============================================================

valid_visual = results_df[
    [
        "image_correlation",
        "absolute_annotation_ratio_change",
        "absolute_damage_change",
        "model_absolute_error"
    ]
].dropna()


if len(valid_visual) >= 3:

    correlation_image_vs_annotation = float(
        valid_visual[
            "image_correlation"
        ].corr(
            valid_visual[
                "absolute_annotation_ratio_change"
            ]
        )
    )

    correlation_image_vs_damage = float(
        valid_visual[
            "image_correlation"
        ].corr(
            valid_visual[
                "absolute_damage_change"
            ]
        )
    )

    correlation_image_vs_error = float(
        valid_visual[
            "image_correlation"
        ].corr(
            valid_visual[
                "model_absolute_error"
            ]
        )
    )

else:

    correlation_image_vs_annotation = None
    correlation_image_vs_damage = None
    correlation_image_vs_error = None


# ============================================================
# HIGH-SIMILARITY GROUP
# ============================================================

high_similarity_group = results_df[
    results_df[
        "image_correlation"
    ] >= 0.95
]


if len(
    high_similarity_group
) > 0:

    high_similarity_mean_annotation_change = float(
        high_similarity_group[
            "absolute_annotation_ratio_change"
        ].mean()
    )

    high_similarity_mean_damage_change = float(
        high_similarity_group[
            "absolute_damage_change"
        ].mean()
    )

else:

    high_similarity_mean_annotation_change = None
    high_similarity_mean_damage_change = None


# ============================================================
# LOW-SIMILARITY GROUP
# ============================================================

low_similarity_group = results_df[
    results_df[
        "image_correlation"
    ] < 0.75
]


if len(
    low_similarity_group
) > 0:

    low_similarity_mean_annotation_change = float(
        low_similarity_group[
            "absolute_annotation_ratio_change"
        ].mean()
    )

    low_similarity_mean_damage_change = float(
        low_similarity_group[
            "absolute_damage_change"
        ].mean()
    )

else:

    low_similarity_mean_annotation_change = None
    low_similarity_mean_damage_change = None


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "2.8",

    "status":
        "PASS",

    "events_analyzed":
        total,

    "visual_similarity": {

        "very_high_similarity":
            very_high,

        "high_similarity":
            high,

        "moderate_similarity":
            moderate,

        "low_similarity":
            low
    },

    "high_similarity_high_annotation_change":
        {

            "records":
                suspicious_count,

            "fraction":
                suspicious_count / total
        },

    "low_similarity_high_annotation_change":
        {

            "records":
                visually_different_count,

            "fraction":
                visually_different_count / total
        },

    "mean_image_correlation":
        float(
            results_df[
                "image_correlation"
            ].mean()
        ),

    "median_image_correlation":
        float(
            results_df[
                "image_correlation"
            ].median()
        ),

    "mean_absolute_damage_change":
        float(
            results_df[
                "absolute_damage_change"
            ].mean()
        ),

    "mean_absolute_annotation_ratio_change":
        float(
            results_df[
                "absolute_annotation_ratio_change"
            ].mean()
        ),

    "high_similarity_group":
        {

            "records":
                int(
                    len(
                        high_similarity_group
                    )
                ),

            "mean_absolute_annotation_ratio_change":
                high_similarity_mean_annotation_change,

            "mean_absolute_damage_change":
                high_similarity_mean_damage_change
        },

    "low_similarity_group":
        {

            "records":
                int(
                    len(
                        low_similarity_group
                    )
                ),

            "mean_absolute_annotation_ratio_change":
                low_similarity_mean_annotation_change,

            "mean_absolute_damage_change":
                low_similarity_mean_damage_change
        },

    "correlations":
        {

            "image_correlation_vs_annotation_change":
                correlation_image_vs_annotation,

            "image_correlation_vs_damage_change":
                correlation_image_vs_damage,

            "image_correlation_vs_model_error":
                correlation_image_vs_error
        },

    "interpretation":
        "This stage quantifies whether large annotation "
        "changes are accompanied by corresponding image "
        "changes. High image similarity combined with "
        "large annotation change is a potential dataset "
        "or annotation consistency concern and should "
        "be manually reviewed.",

    "output":
        OUTPUT_CASES
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# DISPLAY MOST IMPORTANT CASES
# ============================================================

print()
print("=" * 70)
print(
    "TOP VISUAL / ANNOTATION TRANSITIONS"
)
print("=" * 70)

display_columns = [
    "sequence",
    "current_frame",
    "future_frame",
    "current_damage",
    "future_damage",
    "absolute_damage_change",
    "image_correlation",
    "absolute_annotation_ratio_change",
    "model_absolute_error",
    "visual_similarity_class",
    "annotation_change_class",
    "high_similarity_high_annotation_change"
]

print(
    results_df.sort_values(
        "absolute_damage_change",
        ascending=False
    )[
        display_columns
    ].head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STAGE 2.8 SUMMARY")
print("=" * 70)

print(
    f"Events analyzed: {total}"
)

print(
    f"Very high visual similarity: "
    f"{very_high}"
)

print(
    f"High visual similarity: "
    f"{high}"
)

print(
    f"Moderate visual similarity: "
    f"{moderate}"
)

print(
    f"Low visual similarity: "
    f"{low}"
)

print()
print(
    "High similarity + high annotation change: "
    f"{suspicious_count}/{total}"
)

print(
    "Low similarity + high annotation change: "
    f"{visually_different_count}/{total}"
)

print()
print(
    f"Mean image correlation: "
    f"{results_df['image_correlation'].mean():.6f}"
)

print(
    f"Median image correlation: "
    f"{results_df['image_correlation'].median():.6f}"
)

print(
    f"Mean absolute damage change: "
    f"{results_df['absolute_damage_change'].mean():.8f}"
)

print(
    f"Mean absolute annotation ratio change: "
    f"{results_df['absolute_annotation_ratio_change'].mean():.8f}"
)

print()
print(
    "Image correlation vs annotation change: "
    f"{correlation_image_vs_annotation}"
)

print(
    "Image correlation vs damage change: "
    f"{correlation_image_vs_damage}"
)

print(
    "Image correlation vs model error: "
    f"{correlation_image_vs_error}"
)

print()
print(
    "Output:"
)

print(
    OUTPUT_CASES
)

print(
    OUTPUT_SUMMARY
)

print()
print(
    "FINAL STATUS: PASS"
)