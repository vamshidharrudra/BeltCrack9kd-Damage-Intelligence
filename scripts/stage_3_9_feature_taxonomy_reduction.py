import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.9
# FEATURE TAXONOMY & REDUCED FEATURE SET VALIDATION
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")


# ============================================================
# INPUT FILES
# ============================================================

FEATURE_SUMMARY_PATH = os.path.join(
    AUDIT_DIR,
    "stage_3_7_feature_summary.csv"
)

CLASSIFICATION_RESULTS_PATH = os.path.join(
    AUDIT_DIR,
    "stage_3_7_classification_results.csv"
)

TEMPORAL_FEATURE_PATH = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

SPATIAL_FEATURE_PATH = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_features.csv"
)

TARGET_PATH = os.path.join(
    AUDIT_DIR,
    "stage_3_6_persistent_damage_targets.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_TAXONOMY = os.path.join(
    AUDIT_DIR,
    "stage_3_9_feature_taxonomy.csv"
)

OUTPUT_REDUNDANCY = os.path.join(
    AUDIT_DIR,
    "stage_3_9_redundancy_analysis.csv"
)

OUTPUT_REDUCED = os.path.join(
    AUDIT_DIR,
    "stage_3_9_reduced_feature_set.csv"
)

OUTPUT_COMPARISON = os.path.join(
    AUDIT_DIR,
    "stage_3_9_model_comparison.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_9_feature_selection_summary.json"
)

OUTPUT_AUDIT = os.path.join(
    AUDIT_DIR,
    "stage_3_9_feature_selection_audit.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CORRELATION_THRESHOLD = 0.98

MIN_NON_NULL_RATIO = 0.95

NEAR_CONSTANT_THRESHOLD = 0.001

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced_subsample",
}


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def detect_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def is_numeric_series(series):
    return pd.api.types.is_numeric_dtype(series)


def classify_feature(name):
    """
    Scientific feature taxonomy.

    This taxonomy is deliberately based on the actual
    feature naming conventions used in Stages 2.2 and 3.1.
    """

    n = str(name)

    # --------------------------------------------------------
    # CURRENT PHYSICAL DAMAGE FEATURES
    # --------------------------------------------------------

    current_exact = {
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
        "crack_bbox_union_height_px",
    }

    if n in current_exact:
        return "CURRENT"

    # --------------------------------------------------------
    # SPATIAL FEATURES
    # --------------------------------------------------------

    if n.startswith("spatial_"):

        if "accel" in n:
            return "SPATIAL_ACCELERATION"

        if (
            "trend" in n
            or "slope" in n
        ):
            return "SPATIAL_TREND"

        if "rolling" in n:
            return "SPATIAL_ROLLING"

        if "lag" in n:
            return "SPATIAL_LAG"

        if (
            "delta" in n
            or "change" in n
            or "velocity" in n
        ):
            return "SPATIAL_CHANGE"

        return "SPATIAL_OTHER"

    # --------------------------------------------------------
    # TEMPORAL ACCELERATION
    # --------------------------------------------------------

    if n.startswith("accel_"):
        return "TEMPORAL_ACCELERATION"

    # --------------------------------------------------------
    # TEMPORAL ROLLING
    # --------------------------------------------------------

    if n.startswith("rolling"):
        return "TEMPORAL_ROLLING"

    # --------------------------------------------------------
    # TEMPORAL TREND
    # --------------------------------------------------------

    if n.startswith("recent_"):

        if (
            "slope" in n
            or "growth" in n
            or "trend" in n
        ):
            return "TEMPORAL_TREND"

    # --------------------------------------------------------
    # TEMPORAL CHANGE
    # --------------------------------------------------------

    if n.startswith("delta"):
        return "TEMPORAL_CHANGE"

    if (
        n.startswith("change_")
        or n.endswith("_change")
    ):
        return "TEMPORAL_CHANGE"

    # --------------------------------------------------------
    # TEMPORAL LAG
    # --------------------------------------------------------

    if n.startswith("lag"):
        return "TEMPORAL_LAG"

    # --------------------------------------------------------
    # OTHER TEMPORAL HISTORY
    # --------------------------------------------------------

    temporal_history_tokens = [
        "lag1_",
        "lag2_",
        "lag3_",
        "recent_",
        "rolling3_",
        "rolling5_",
    ]

    for token in temporal_history_tokens:
        if token in n:
            return "TEMPORAL_HISTORY"

    return "OTHER"


def feature_priority(name, group):

    n = str(name)

    score = 0

    if group == "CURRENT":
        score += 100

    if group == "SPATIAL_CHANGE":
        score += 90

    if group == "SPATIAL_LAG":
        score += 85

    if group == "SPATIAL_TREND":
        score += 84

    if group == "SPATIAL_ROLLING":
        score += 70

    if group == "SPATIAL_ACCELERATION":
        score += 65

    if group == "SPATIAL_OTHER":
        score += 50

    if group == "TEMPORAL_CHANGE":
        score += 80

    if group == "TEMPORAL_LAG":
        score += 75

    if group == "TEMPORAL_TREND":
        score += 78

    if group == "TEMPORAL_ROLLING":
        score += 65

    if group == "TEMPORAL_ACCELERATION":
        score += 60

    if group == "TEMPORAL_HISTORY":
        score += 55

    preferred_terms = [
        "crack_area_ratio",
        "crack_count",
        "mean_match_iou",
        "spatial_match_quality",
        "persistent_fraction",
        "future_coverage",
        "mean_center_distance",
        "mean_area_similarity",
        "matched_crack_count",
        "unmatched_current_fraction",
        "mean_match_score",
        "crack_perimeter",
        "largest_crack",
    ]

    for term in preferred_terms:
        if term in n:
            score += 3

    return score


# ============================================================
# START
# ============================================================

print_header(
    "STAGE 3.9 - FEATURE TAXONOMY & REDUCED FEATURE SET VALIDATION"
)

print("Project:")
print(PROJECT_ROOT)


# ============================================================
# CHECK INPUTS
# ============================================================

print_header("CHECKING INPUT FILES")

input_files = {
    "feature_summary": FEATURE_SUMMARY_PATH,
    "classification_results": CLASSIFICATION_RESULTS_PATH,
    "temporal_features": TEMPORAL_FEATURE_PATH,
    "spatial_features": SPATIAL_FEATURE_PATH,
    "targets": TARGET_PATH,
}

missing = []

for name, path in input_files.items():

    exists = os.path.exists(path)

    print(
        f"{name:25s}: "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:
        missing.append(name)

if missing:

    raise FileNotFoundError(
        "Missing required files: "
        + ", ".join(missing)
    )


# ============================================================
# LOAD STAGE 3.7 FEATURE SUMMARY
# ============================================================

print_header(
    "LOADING STAGE 3.7 FEATURE SUMMARY"
)

feature_summary = pd.read_csv(
    FEATURE_SUMMARY_PATH
)

print(
    "Feature summary rows:",
    len(feature_summary)
)

print("Feature summary columns:")

for col in feature_summary.columns:
    print("  - " + str(col))


feature_name_column = detect_column(
    feature_summary,
    [
        "feature",
        "feature_name",
        "name",
        "Feature",
        "Feature_Name",
    ]
)

if feature_name_column is None:

    raise ValueError(
        "Could not identify feature-name column."
    )

all_features = (
    feature_summary[
        feature_name_column
    ]
    .astype(str)
    .str.strip()
    .tolist()
)

all_features = [
    x for x in all_features
    if x and x.lower() != "nan"
]

all_features = list(
    dict.fromkeys(all_features)
)

print(
    "Unique Stage 3.7 features:",
    len(all_features)
)


# ============================================================
# LOAD TEMPORAL FEATURES
# ============================================================

print_header(
    "LOADING TEMPORAL FEATURES"
)

temporal = pd.read_csv(
    TEMPORAL_FEATURE_PATH
)

print(
    "Temporal records:",
    len(temporal)
)

print(
    "Temporal columns:",
    len(temporal.columns)
)


# ============================================================
# LOAD SPATIAL FEATURES
# ============================================================

print_header(
    "LOADING SPATIAL FEATURES"
)

spatial = pd.read_csv(
    SPATIAL_FEATURE_PATH
)

print(
    "Spatial records:",
    len(spatial)
)

print(
    "Spatial columns:",
    len(spatial.columns)
)


# ============================================================
# LOAD TARGETS
# ============================================================

print_header(
    "LOADING PERSISTENT DAMAGE TARGETS"
)

targets = pd.read_csv(
    TARGET_PATH
)

print(
    "Target records:",
    len(targets)
)

print("Target columns:")

for col in targets.columns:
    print("  - " + str(col))


# ============================================================
# IMPORTANT CORRECTION
# ============================================================
#
# The actual Stage 3.6 target schema shows:
#
# persistent_damage_target
#
# This is the classification target.
#
# It is NOT called:
# target_state
# persistent_damage_state
# damage_state
#
# The other target-file columns such as future_total_area,
# damage_future, progression_state, etc. are target-construction
# information and MUST NOT become model predictor features.
# ============================================================

TARGET_COLUMN = "persistent_damage_target"

if TARGET_COLUMN not in targets.columns:

    raise ValueError(
        "Expected target column "
        "'persistent_damage_target' "
        "was not found in Stage 3.6 target file."
    )

print()
print(
    "Classification target:",
    TARGET_COLUMN
)

print()
print("Target distribution:")

print(
    targets[
        TARGET_COLUMN
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# DETERMINE JOIN KEYS
# ============================================================

print_header(
    "DETERMINING LEAKAGE-SAFE JOIN KEYS"
)

temporal_sequence = detect_column(
    temporal,
    ["sequence"]
)

temporal_component = detect_column(
    temporal,
    ["component"]
)

temporal_frame = detect_column(
    temporal,
    [
        "frame_number",
        "current_frame",
    ]
)

spatial_sequence = detect_column(
    spatial,
    ["sequence"]
)

spatial_component = detect_column(
    spatial,
    ["component"]
)

spatial_frame = detect_column(
    spatial,
    [
        "current_frame",
        "frame_number",
    ]
)

target_sequence = detect_column(
    targets,
    ["sequence"]
)

target_component = detect_column(
    targets,
    ["component"]
)

target_frame = detect_column(
    targets,
    [
        "current_frame",
        "target_current_frame",
        "frame_number",
    ]
)

print(
    "Temporal sequence:",
    temporal_sequence
)

print(
    "Temporal component:",
    temporal_component
)

print(
    "Temporal frame:",
    temporal_frame
)

print(
    "Spatial sequence:",
    spatial_sequence
)

print(
    "Spatial component:",
    spatial_component
)

print(
    "Spatial frame:",
    spatial_frame
)

print(
    "Target sequence:",
    target_sequence
)

print(
    "Target component:",
    target_component
)

print(
    "Target frame:",
    target_frame
)


if None in [
    temporal_sequence,
    temporal_component,
    temporal_frame,
    spatial_sequence,
    spatial_component,
    spatial_frame,
    target_sequence,
    target_component,
    target_frame,
]:

    raise ValueError(
        "Could not determine sequence/component/frame keys."
    )


# ============================================================
# STANDARDIZE JOIN KEYS
# ============================================================

temporal = temporal.copy()
spatial = spatial.copy()
targets = targets.copy()

temporal["_sequence_key"] = (
    temporal[
        temporal_sequence
    ].astype(str)
)

temporal["_component_key"] = pd.to_numeric(
    temporal[
        temporal_component
    ],
    errors="coerce"
)

temporal["_frame_key"] = pd.to_numeric(
    temporal[
        temporal_frame
    ],
    errors="coerce"
)

spatial["_sequence_key"] = (
    spatial[
        spatial_sequence
    ].astype(str)
)

spatial["_component_key"] = pd.to_numeric(
    spatial[
        spatial_component
    ],
    errors="coerce"
)

spatial["_frame_key"] = pd.to_numeric(
    spatial[
        spatial_frame
    ],
    errors="coerce"
)

targets["_sequence_key"] = (
    targets[
        target_sequence
    ].astype(str)
)

targets["_component_key"] = pd.to_numeric(
    targets[
        target_component
    ],
    errors="coerce"
)

targets["_frame_key"] = pd.to_numeric(
    targets[
        target_frame
    ],
    errors="coerce"
)

JOIN_KEYS = [
    "_sequence_key",
    "_component_key",
    "_frame_key",
]


# ============================================================
# KEY INTEGRITY
# ============================================================

print_header(
    "KEY INTEGRITY CHECK"
)

temporal_duplicates = temporal.duplicated(
    JOIN_KEYS
).sum()

spatial_duplicates = spatial.duplicated(
    JOIN_KEYS
).sum()

target_duplicates = targets.duplicated(
    JOIN_KEYS
).sum()

print(
    "Temporal duplicate keys:",
    temporal_duplicates
)

print(
    "Spatial duplicate keys:",
    spatial_duplicates
)

print(
    "Target duplicate keys:",
    target_duplicates
)

if (
    temporal_duplicates > 0
    or spatial_duplicates > 0
    or target_duplicates > 0
):

    raise ValueError(
        "Duplicate sequence/component/frame keys detected."
    )


# ============================================================
# LEAKAGE-SAFE FEATURE SELECTION
# ============================================================

print_header(
    "BUILDING LEAKAGE-SAFE FEATURE MATRIX"
)

# Important:
# Predictor features come ONLY from Stage 2.2 temporal
# features and Stage 3.1 spatial features.
#
# NOTHING is taken from the Stage 3.6 target file except
# persistent_damage_target.
#
# This prevents target-construction information from
# entering the model.

excluded_exact = set(
    JOIN_KEYS
    + [
        "frame_number",
        "current_frame",
        "future_frame",
        "target_current_frame",
        "target_future_frame",
        "target_t_plus_1",
        "target_t_plus_3",
        "target_t_plus_5",
        "target",
    ]
)

excluded_tokens = [
    "target",
    "future",
    "next",
]

candidate_features = []

for col in list(
    dict.fromkeys(
        list(temporal.columns)
        + list(spatial.columns)
    )
):

    if col in excluded_exact:
        continue

    if str(col).startswith("_"):
        continue

    lower = str(col).lower()

    if any(
        token in lower
        for token in excluded_tokens
    ):
        continue

    if col in temporal.columns:

        series = temporal[col]

    elif col in spatial.columns:

        series = spatial[col]

    else:

        continue

    if not is_numeric_series(series):
        continue

    candidate_features.append(
        str(col)
    )

candidate_features = list(
    dict.fromkeys(candidate_features)
)

print(
    "Numeric leakage-safe candidate features:",
    len(candidate_features)
)


# ============================================================
# SEPARATE TEMPORAL AND SPATIAL FEATURES
# ============================================================

temporal_feature_cols = [
    c
    for c in candidate_features
    if c in temporal.columns
]

spatial_feature_cols = [
    c
    for c in candidate_features
    if c in spatial.columns
]

print(
    "Temporal feature columns:",
    len(temporal_feature_cols)
)

print(
    "Spatial feature columns:",
    len(spatial_feature_cols)
)


# ============================================================
# BUILD TEMPORAL TABLE
# ============================================================

temporal_small = temporal[
    JOIN_KEYS
    + temporal_feature_cols
].copy()


# ============================================================
# BUILD SPATIAL TABLE
# ============================================================

spatial_small = spatial[
    JOIN_KEYS
    + spatial_feature_cols
].copy()


# ============================================================
# HANDLE SAME-NAME FEATURES
# ============================================================

overlap_features = (
    set(temporal_feature_cols)
    &
    set(spatial_feature_cols)
)

print(
    "Overlapping feature names:",
    len(overlap_features)
)

for feature in sorted(
    overlap_features
):

    temporal_values = (
        temporal_small[
            feature
        ]
    )

    spatial_values = (
        spatial_small[
            feature
        ]
    )

    comparable = (
        temporal_values.notna()
        &
        spatial_values.notna()
    )

    if comparable.any():

        max_diff = np.nanmax(
            np.abs(
                temporal_values[
                    comparable
                ].to_numpy()
                -
                spatial_values[
                    comparable
                ].to_numpy()
            )
        )

    else:

        max_diff = 0.0

    if max_diff > 1e-10:

        new_name = (
            "spatial_source_"
            + feature
        )

        spatial_small = (
            spatial_small.rename(
                columns={
                    feature: new_name
                }
            )
        )

    else:

        spatial_small = (
            spatial_small.drop(
                columns=[feature]
            )
        )


# ============================================================
# MERGE TEMPORAL + SPATIAL
# ============================================================

merged = temporal_small.merge(
    spatial_small,
    on=JOIN_KEYS,
    how="inner",
    suffixes=(
        "",
        "_spatial_duplicate"
    )
)

# Remove accidental duplicate columns.
duplicate_suffix_columns = [
    c
    for c in merged.columns
    if c.endswith(
        "_spatial_duplicate"
    )
]

if duplicate_suffix_columns:

    merged = merged.drop(
        columns=duplicate_suffix_columns
    )


# ============================================================
# ADD ONLY THE CLASSIFICATION TARGET
# ============================================================

target_small = targets[
    JOIN_KEYS
    + [
        TARGET_COLUMN
    ]
].copy()

merged = merged.merge(
    target_small,
    on=JOIN_KEYS,
    how="inner"
)

print(
    "Merged records:",
    len(merged)
)


# ============================================================
# FINAL FEATURE COLUMNS
# ============================================================

feature_columns = []

for col in merged.columns:

    if col in JOIN_KEYS:
        continue

    if col == TARGET_COLUMN:
        continue

    if str(col).startswith("_"):
        continue

    lower = str(col).lower()

    if any(
        token in lower
        for token in [
            "target",
            "future",
            "next",
        ]
    ):
        continue

    if not is_numeric_series(
        merged[col]
    ):
        continue

    feature_columns.append(
        col
    )

feature_columns = list(
    dict.fromkeys(
        feature_columns
    )
)

print(
    "Final candidate feature columns:",
    len(feature_columns)
)


# ============================================================
# VERIFY EXPECTED 232 FEATURES
# ============================================================

missing_expected_features = [
    f
    for f in all_features
    if f not in feature_columns
]

unexpected_features = [
    f
    for f in feature_columns
    if f not in all_features
]

print()
print(
    "Expected Stage 3.7 features:",
    len(all_features)
)

print(
    "Available candidate features:",
    len(feature_columns)
)

print(
    "Expected features missing:",
    len(missing_expected_features)
)

print(
    "Unexpected candidate features:",
    len(unexpected_features)
)

if missing_expected_features:

    print()
    print(
        "Missing expected features:"
    )

    for feature in missing_expected_features:
        print(
            "  - " + feature
        )


# ============================================================
# FEATURE TAXONOMY
# ============================================================

print_header(
    "BUILDING FEATURE TAXONOMY"
)

taxonomy_rows = []

for feature in feature_columns:

    group = classify_feature(
        feature
    )

    series = merged[
        feature
    ]

    non_null_ratio = (
        series.notna().mean()
    )

    unique_count = (
        series.nunique(
            dropna=True
        )
    )

    if unique_count <= 1:

        variance = 0.0

    else:

        variance = float(
            series.var()
        )

    mean_abs = float(
        np.nanmean(
            np.abs(
                series.to_numpy(
                    dtype=float
                )
            )
        )
    )

    taxonomy_rows.append(
        {
            "feature": feature,
            "group": group,
            "priority_score":
                feature_priority(
                    feature,
                    group
                ),
            "non_null_ratio":
                non_null_ratio,
            "unique_count":
                unique_count,
            "variance":
                variance,
            "mean_absolute_value":
                mean_abs,
        }
    )

taxonomy = pd.DataFrame(
    taxonomy_rows
)

taxonomy = taxonomy.sort_values(
    [
        "group",
        "priority_score",
        "feature",
    ],
    ascending=[
        True,
        False,
        True,
    ]
).reset_index(
    drop=True
)

taxonomy.to_csv(
    OUTPUT_TAXONOMY,
    index=False
)

print(
    "Taxonomy saved:"
)

print(
    OUTPUT_TAXONOMY
)

print()
print(
    "Feature group counts:"
)

print(
    taxonomy[
        "group"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# FEATURE QUALITY AUDIT
# ============================================================

print_header(
    "FEATURE QUALITY AUDIT"
)

taxonomy[
    "remove_missingness"
] = (
    taxonomy[
        "non_null_ratio"
    ]
    <
    MIN_NON_NULL_RATIO
)

taxonomy[
    "remove_near_constant"
] = (
    (
        taxonomy[
            "unique_count"
        ]
        <= 1
    )
    |
    (
        taxonomy[
            "variance"
        ]
        <
        NEAR_CONSTANT_THRESHOLD
    )
)

taxonomy[
    "quality_keep"
] = ~(
    taxonomy[
        "remove_missingness"
    ]
    |
    taxonomy[
        "remove_near_constant"
    ]
)

print(
    "Missingness-filtered:",
    int(
        taxonomy[
            "remove_missingness"
        ].sum()
    )
)

print(
    "Near-constant-filtered:",
    int(
        taxonomy[
            "remove_near_constant"
        ].sum()
    )
)

print(
    "Quality-eligible:",
    int(
        taxonomy[
            "quality_keep"
        ].sum()
    )
)


# ============================================================
# CORRELATION REDUNDANCY
# ============================================================

print_header(
    "CORRELATION REDUNDANCY ANALYSIS"
)

quality_features = (
    taxonomy.loc[
        taxonomy[
            "quality_keep"
        ],
        "feature"
    ]
    .tolist()
)

X_corr = merged[
    quality_features
].copy()

for col in X_corr.columns:

    median_value = (
        X_corr[col].median()
    )

    if pd.isna(
        median_value
    ):

        median_value = 0.0

    X_corr[col] = (
        X_corr[col]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(
            median_value
        )
    )

corr_matrix = X_corr.corr(
    method="spearman"
)

priority_lookup = dict(
    zip(
        taxonomy[
            "feature"
        ],
        taxonomy[
            "priority_score"
        ]
    )
)

group_lookup = dict(
    zip(
        taxonomy[
            "feature"
        ],
        taxonomy[
            "group"
        ]
    )
)

redundancy_rows = []

for i in range(
    len(quality_features)
):

    feature_a = (
        quality_features[i]
    )

    for j in range(
        i + 1,
        len(quality_features)
    ):

        feature_b = (
            quality_features[j]
        )

        correlation = (
            corr_matrix.loc[
                feature_a,
                feature_b
            ]
        )

        if pd.isna(
            correlation
        ):
            continue

        absolute_corr = abs(
            float(correlation)
        )

        if (
            absolute_corr
            >= CORRELATION_THRESHOLD
        ):

            score_a = (
                priority_lookup.get(
                    feature_a,
                    0
                )
            )

            score_b = (
                priority_lookup.get(
                    feature_b,
                    0
                )
            )

            if score_a >= score_b:

                preferred = feature_a
                redundant = feature_b

            else:

                preferred = feature_b
                redundant = feature_a

            redundancy_rows.append(
                {
                    "feature_a":
                        feature_a,

                    "feature_b":
                        feature_b,

                    "spearman_correlation":
                        float(
                            correlation
                        ),

                    "absolute_correlation":
                        absolute_corr,

                    "preferred_feature":
                        preferred,

                    "redundant_feature":
                        redundant,

                    "preferred_group":
                        group_lookup.get(
                            preferred,
                            "UNKNOWN"
                        ),

                    "redundant_group":
                        group_lookup.get(
                            redundant,
                            "UNKNOWN"
                        ),
                }
            )

redundancy = pd.DataFrame(
    redundancy_rows
)

if len(
    redundancy
) > 0:

    redundancy = (
        redundancy
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

else:

    redundancy = pd.DataFrame(
        columns=[
            "feature_a",
            "feature_b",
            "spearman_correlation",
            "absolute_correlation",
            "preferred_feature",
            "redundant_feature",
            "preferred_group",
            "redundant_group",
        ]
    )

redundancy.to_csv(
    OUTPUT_REDUNDANCY,
    index=False
)

print(
    "Highly correlated feature pairs:",
    len(redundancy)
)

print(
    "Correlation threshold:",
    CORRELATION_THRESHOLD
)


# ============================================================
# REDUCED FEATURE SET
# ============================================================

print_header(
    "BUILDING REDUCED INTERPRETABLE FEATURE SET"
)

priority_sorted = (
    taxonomy[
        taxonomy[
            "quality_keep"
        ]
    ]
    .sort_values(
        [
            "priority_score",
            "feature",
        ],
        ascending=[
            False,
            True,
        ]
    )
)

redundant_features = set()

if len(
    redundancy
) > 0:

    redundant_features = set(
        redundancy[
            "redundant_feature"
        ]
        .tolist()
    )

selected_features = []

for _, row in (
    priority_sorted.iterrows()
):

    feature = row[
        "feature"
    ]

    if feature in redundant_features:
        continue

    selected_features.append(
        feature
    )


# ============================================================
# FORCE CORE PHYSICAL DAMAGE FEATURES
# ============================================================

core_features = [
    "crack_count",
    "crack_area_ratio",
    "largest_crack_area_ratio",
    "total_crack_area_px",
    "mean_crack_area_px",
    "mean_crack_perimeter_px",
]

for feature in core_features:

    if feature in feature_columns:

        if feature not in selected_features:

            selected_features.append(
                feature
            )


selected_features = list(
    dict.fromkeys(
        selected_features
    )
)

print(
    "Full feature count:",
    len(feature_columns)
)

print(
    "Quality-eligible count:",
    len(quality_features)
)

print(
    "Reduced feature count:",
    len(selected_features)
)

print(
    "Removed feature count:",
    len(feature_columns)
    -
    len(selected_features)
)


# ============================================================
# SAVE REDUCED FEATURE SET
# ============================================================

reduced_rows = []

selected_set = set(
    selected_features
)

for feature in feature_columns:

    taxonomy_row = taxonomy[
        taxonomy[
            "feature"
        ]
        ==
        feature
    ]

    if len(
        taxonomy_row
    ) == 0:
        continue

    row = taxonomy_row.iloc[0]

    if feature in selected_set:

        status = "SELECTED"

        reason = (
            "INTERPRETABLE_AND_NON_REDUNDANT"
        )

    elif not bool(
        row[
            "quality_keep"
        ]
    ):

        status = "REMOVED"

        reason = "QUALITY_FILTER"

    elif feature in redundant_features:

        status = "REMOVED"

        reason = (
            "HIGH_CORRELATION_REDUNDANCY"
        )

    else:

        status = "REMOVED"

        reason = (
            "REDUCED_SET_FILTER"
        )

    reduced_rows.append(
        {
            "feature": feature,
            "group": row["group"],
            "priority_score":
                row["priority_score"],
            "status": status,
            "reason": reason,
        }
    )

reduced_df = pd.DataFrame(
    reduced_rows
)

reduced_df = (
    reduced_df
    .sort_values(
        [
            "status",
            "group",
            "priority_score",
            "feature",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ]
    )
)

reduced_df.to_csv(
    OUTPUT_REDUCED,
    index=False
)

print(
    "Reduced feature set saved:"
)

print(
    OUTPUT_REDUCED
)


# ============================================================
# GET SPLIT INFORMATION
# ============================================================

print_header(
    "RECOVERING TRAIN / VAL / TEST SPLITS"
)

split_column = None

if "split" in temporal.columns:

    split_column = "split"

    split_lookup = (
        temporal[
            JOIN_KEYS
            + ["split"]
        ]
        .drop_duplicates(
            JOIN_KEYS
        )
    )

elif "split" in spatial.columns:

    split_column = "split"

    split_lookup = (
        spatial[
            JOIN_KEYS
            + ["split"]
        ]
        .drop_duplicates(
            JOIN_KEYS
        )
    )

else:

    classification_results = pd.read_csv(
        CLASSIFICATION_RESULTS_PATH
    )

    if "split" not in (
        classification_results.columns
    ):

        raise ValueError(
            "Could not recover split information."
        )

    split_column = "split"

    classification_results[
        "_sequence_key"
    ] = (
        classification_results[
            "sequence"
        ].astype(str)
    )

    classification_results[
        "_component_key"
    ] = pd.to_numeric(
        classification_results[
            "component"
        ],
        errors="coerce"
    )

    classification_frame_column = (
        "current_frame"
        if "current_frame"
        in classification_results.columns
        else "frame_number"
    )

    classification_results[
        "_frame_key"
    ] = pd.to_numeric(
        classification_results[
            classification_frame_column
        ],
        errors="coerce"
    )

    split_lookup = (
        classification_results[
            JOIN_KEYS
            + ["split"]
        ]
        .drop_duplicates(
            JOIN_KEYS
        )
    )

merged = merged.merge(
    split_lookup,
    on=JOIN_KEYS,
    how="left"
)

print(
    "Split column:",
    split_column
)

print()
print(
    merged[
        split_column
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# TARGET VALIDATION
# ============================================================

print_header(
    "TARGET VALIDATION"
)

missing_target = (
    merged[
        TARGET_COLUMN
    ].isna().sum()
)

print(
    "Missing target values:",
    missing_target
)

if missing_target > 0:

    merged = merged[
        merged[
            TARGET_COLUMN
        ].notna()
    ].copy()

print(
    "Model records:",
    len(merged)
)

print()
print(
    "Final target distribution:"
)

print(
    merged[
        TARGET_COLUMN
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# FEATURE GROUPS
# ============================================================

taxonomy_lookup = dict(
    zip(
        taxonomy[
            "feature"
        ],
        taxonomy[
            "group"
        ]
    )
)


def get_group_features(
    groups
):

    return [
        feature
        for feature in feature_columns
        if taxonomy_lookup.get(
            feature,
            "OTHER"
        )
        in groups
        and feature in merged.columns
    ]


current_features = get_group_features(
    {"CURRENT"}
)

temporal_features = get_group_features(
    {
        "TEMPORAL_LAG",
        "TEMPORAL_CHANGE",
        "TEMPORAL_ACCELERATION",
        "TEMPORAL_ROLLING",
        "TEMPORAL_TREND",
        "TEMPORAL_HISTORY",
    }
)

spatial_features = get_group_features(
    {
        "SPATIAL_LAG",
        "SPATIAL_CHANGE",
        "SPATIAL_ACCELERATION",
        "SPATIAL_ROLLING",
        "SPATIAL_TREND",
        "SPATIAL_OTHER",
    }
)

full_features = [
    feature
    for feature in feature_columns
    if feature in merged.columns
]

reduced_features = [
    feature
    for feature in selected_features
    if feature in merged.columns
]


print()
print(
    "Feature set sizes:"
)

print(
    "CURRENT:",
    len(current_features)
)

print(
    "TEMPORAL:",
    len(temporal_features)
)

print(
    "SPATIAL:",
    len(spatial_features)
)

print(
    "FULL:",
    len(full_features)
)

print(
    "REDUCED:",
    len(reduced_features)
)


# ============================================================
# MODEL SPLITS
# ============================================================

print_header(
    "PREPARING MODEL SPLITS"
)

train_mask = (
    merged[
        split_column
    ]
    .astype(str)
    .str.upper()
    ==
    "TRAIN"
)

val_mask = (
    merged[
        split_column
    ]
    .astype(str)
    .str.upper()
    ==
    "VAL"
)

test_mask = (
    merged[
        split_column
    ]
    .astype(str)
    .str.upper()
    ==
    "TEST"
)

print(
    "TRAIN:",
    int(train_mask.sum())
)

print(
    "VAL:",
    int(val_mask.sum())
)

print(
    "TEST:",
    int(test_mask.sum())
)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_feature_set(
    name,
    features
):

    features = [
        f
        for f in features
        if f in merged.columns
    ]

    if len(features) == 0:

        print(
            "Skipping",
            name,
            "- no features."
        )

        return None

    X = merged[
        features
    ].copy()

    y = merged[
        TARGET_COLUMN
    ].astype(str)

    X_train = X.loc[
        train_mask
    ].copy()

    X_val = X.loc[
        val_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ]

    y_val = y.loc[
        val_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    # --------------------------------------------------------
    # TRAIN-ONLY MEDIAN IMPUTATION
    # --------------------------------------------------------

    for col in features:

        X_train[col] = (
            X_train[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        X_val[col] = (
            X_val[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        X_test[col] = (
            X_test[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        median_value = (
            X_train[col].median()
        )

        if pd.isna(
            median_value
        ):

            median_value = 0.0

        X_train[col] = (
            X_train[col]
            .fillna(
                median_value
            )
        )

        X_val[col] = (
            X_val[col]
            .fillna(
                median_value
            )
        )

        X_test[col] = (
            X_test[col]
            .fillna(
                median_value
            )
        )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        **RF_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )

    result = {
        "feature_set": name,

        "feature_count":
            len(features),

        "train_records":
            len(X_train),

        "val_records":
            len(X_val),

        "test_records":
            len(X_test),

        "val_accuracy":
            accuracy_score(
                y_val,
                val_pred
            ),

        "val_balanced_accuracy":
            balanced_accuracy_score(
                y_val,
                val_pred
            ),

        "val_macro_f1":
            f1_score(
                y_val,
                val_pred,
                average="macro",
                zero_division=0
            ),

        "test_accuracy":
            accuracy_score(
                y_test,
                test_pred
            ),

        "test_balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                test_pred
            ),

        "test_macro_f1":
            f1_score(
                y_test,
                test_pred,
                average="macro",
                zero_division=0
            ),
    }

    print()
    print(
        name
    )

    print(
        "  Features:",
        len(features)
    )

    print(
        "  VAL Macro F1:",
        f"{result['val_macro_f1']:.6f}"
    )

    print(
        "  VAL Balanced Accuracy:",
        f"{result['val_balanced_accuracy']:.6f}"
    )

    print(
        "  TEST Macro F1:",
        f"{result['test_macro_f1']:.6f}"
    )

    print(
        "  TEST Balanced Accuracy:",
        f"{result['test_balanced_accuracy']:.6f}"
    )

    print(
        "  TEST Accuracy:",
        f"{result['test_accuracy']:.6f}"
    )

    return result


# ============================================================
# RUN EXPERIMENTS
# ============================================================

print_header(
    "RUNNING FEATURE SET EXPERIMENTS"
)

results = []


# CURRENT
result = evaluate_feature_set(
    "CURRENT_ONLY",
    current_features
)

if result is not None:
    results.append(result)


# CURRENT + TEMPORAL
result = evaluate_feature_set(
    "CURRENT_PLUS_TEMPORAL",
    list(
        dict.fromkeys(
            current_features
            +
            temporal_features
        )
    )
)

if result is not None:
    results.append(result)


# CURRENT + SPATIAL
result = evaluate_feature_set(
    "CURRENT_PLUS_SPATIAL",
    list(
        dict.fromkeys(
            current_features
            +
            spatial_features
        )
    )
)

if result is not None:
    results.append(result)


# FULL
result = evaluate_feature_set(
    "FULL_232_FEATURES",
    full_features
)

if result is not None:
    results.append(result)


# REDUCED
result = evaluate_feature_set(
    "REDUCED_INTERPRETABLE",
    reduced_features
)

if result is not None:
    results.append(result)


# ============================================================
# SAVE COMPARISON
# ============================================================

print_header(
    "FINAL STAGE 3.9 MODEL COMPARISON"
)

comparison = pd.DataFrame(
    results
)

comparison = (
    comparison
    .sort_values(
        "test_macro_f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

print()
print(
    "Model comparison saved:"
)

print(
    OUTPUT_COMPARISON
)


# ============================================================
# REDUCED MODEL RETENTION
# ============================================================

full_macro_f1 = None
reduced_macro_f1 = None
macro_f1_difference = None
macro_f1_retention = None

full_row = comparison[
    comparison[
        "feature_set"
    ]
    ==
    "FULL_232_FEATURES"
]

reduced_row = comparison[
    comparison[
        "feature_set"
    ]
    ==
    "REDUCED_INTERPRETABLE"
]

if (
    len(full_row) == 1
    and len(reduced_row) == 1
):

    full_macro_f1 = float(
        full_row.iloc[0][
            "test_macro_f1"
        ]
    )

    reduced_macro_f1 = float(
        reduced_row.iloc[0][
            "test_macro_f1"
        ]
    )

    macro_f1_difference = (
        reduced_macro_f1
        -
        full_macro_f1
    )

    if full_macro_f1 != 0:

        macro_f1_retention = (
            reduced_macro_f1
            /
            full_macro_f1
        )


print()
print(
    "Full TEST Macro F1:",
    (
        f"{full_macro_f1:.6f}"
        if full_macro_f1 is not None
        else "NA"
    )
)

print(
    "Reduced TEST Macro F1:",
    (
        f"{reduced_macro_f1:.6f}"
        if reduced_macro_f1 is not None
        else "NA"
    )
)

if macro_f1_retention is not None:

    print(
        "Macro F1 retention:",
        f"{macro_f1_retention * 100:.2f}%"
    )

    print(
        "Macro F1 difference:",
        f"{macro_f1_difference:+.6f}"
    )


# ============================================================
# TAXONOMY SUMMARY
# ============================================================

taxonomy_summary = (
    taxonomy
    .groupby("group")
    .agg(
        feature_count=(
            "feature",
            "count"
        ),
        quality_eligible=(
            "quality_keep",
            "sum"
        ),
        mean_priority=(
            "priority_score",
            "mean"
        ),
    )
    .reset_index()
    .sort_values(
        "feature_count",
        ascending=False
    )
)

print()
print(
    "Feature taxonomy summary:"
)

print(
    taxonomy_summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL AUDIT
# ============================================================

print_header(
    "STAGE 3.9 AUDIT"
)

future_named_features = []

for feature in full_features:

    lower = str(
        feature
    ).lower()

    if (
        "future" in lower
        or "target" in lower
        or "next" in lower
    ):

        future_named_features.append(
            feature
        )

duplicate_selected = (
    len(selected_features)
    !=
    len(
        set(
            selected_features
        )
    )
)

missing_reduced_features = [
    feature
    for feature in selected_features
    if feature not in merged.columns
]

checks = {

    "feature_matrix_constructed":
        len(merged) > 0,

    "expected_feature_count_reconstructed":
        len(missing_expected_features) == 0,

    "full_features_available":
        len(full_features) > 0,

    "reduced_features_available":
        len(reduced_features) > 0,

    "no_future_named_features":
        len(
            future_named_features
        ) == 0,

    "no_duplicate_selected_features":
        not duplicate_selected,

    "no_missing_reduced_features":
        len(
            missing_reduced_features
        ) == 0,

    "models_evaluated":
        len(comparison) >= 5,

    "target_column_verified":
        TARGET_COLUMN
        ==
        "persistent_damage_target",
}


all_checks_pass = all(
    checks.values()
)

audit = {

    "stage":
        "3.9",

    "stage_name":
        "Feature Taxonomy and Reduced Feature Set Validation",

    "target_column":
        TARGET_COLUMN,

    "input_target_records":
        int(
            len(targets)
        ),

    "merged_model_records":
        int(
            len(merged)
        ),

    "expected_stage_3_7_features":
        int(
            len(all_features)
        ),

    "reconstructed_feature_count":
        int(
            len(full_features)
        ),

    "missing_expected_features":
        missing_expected_features,

    "unexpected_features":
        unexpected_features,

    "reduced_feature_count":
        int(
            len(reduced_features)
        ),

    "feature_reduction":
        int(
            len(full_features)
            -
            len(reduced_features)
        ),

    "feature_reduction_percent":
        float(
            (
                1
                -
                (
                    len(reduced_features)
                    /
                    len(full_features)
                )
            )
            *
            100
        )
        if len(full_features) > 0
        else None,

    "taxonomy_groups":
        taxonomy[
            "group"
        ]
        .value_counts()
        .to_dict(),

    "quality_filtered_features":
        int(
            (
                ~taxonomy[
                    "quality_keep"
                ]
            ).sum()
        ),

    "high_correlation_pairs":
        int(
            len(
                redundancy
            )
        ),

    "future_named_features_remaining":
        future_named_features,

    "duplicate_selected_features":
        bool(
            duplicate_selected
        ),

    "missing_reduced_features":
        missing_reduced_features,

    "full_test_macro_f1":
        full_macro_f1,

    "reduced_test_macro_f1":
        reduced_macro_f1,

    "reduced_macro_f1_retention":
        macro_f1_retention,

    "reduced_macro_f1_difference":
        macro_f1_difference,

    "checks":
        checks,

    "all_summary_checks_pass":
        bool(
            all_checks_pass
        ),

    "status":
        (
            "PASS"
            if all_checks_pass
            else
            "REVIEW"
        ),
}


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "taxonomy_summary":
        taxonomy_summary.to_dict(
            orient="records"
        ),

    "model_comparison":
        comparison.to_dict(
            orient="records"
        ),

    "audit":
        audit,
}

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

with open(
    OUTPUT_AUDIT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print(
    "STAGE 3.9 COMPLETE"
)
print("=" * 78)

print(
    "Classification target:",
    TARGET_COLUMN
)

print(
    "Full features:",
    len(full_features)
)

print(
    "Reduced features:",
    len(reduced_features)
)

print()
print(
    "Feature taxonomy:"
)

print(
    OUTPUT_TAXONOMY
)

print(
    "Redundancy analysis:"
)

print(
    OUTPUT_REDUNDANCY
)

print(
    "Reduced feature set:"
)

print(
    OUTPUT_REDUCED
)

print(
    "Model comparison:"
)

print(
    OUTPUT_COMPARISON
)

print(
    "Summary:"
)

print(
    OUTPUT_SUMMARY
)

print(
    "Audit:"
)

print(
    OUTPUT_AUDIT
)

print()
print(
    "Final status:",
    audit["status"]
)

print()
print(
    "NEXT: STAGE 3.10 - CLASSIFICATION ERROR ANALYSIS"
)