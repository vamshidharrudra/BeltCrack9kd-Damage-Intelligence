import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

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

FEATURE_SUMMARY_FILE = (
    AUDIT_DIR / "stage_3_7_feature_summary.csv"
)

TARGET_FILE = (
    AUDIT_DIR / "stage_3_6_persistent_damage_targets.csv"
)

CLASSIFICATION_RESULTS_FILE = (
    AUDIT_DIR / "stage_3_7_classification_results.csv"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

TAXONOMY_OUTPUT = (
    AUDIT_DIR / "stage_3_9c_exact_feature_taxonomy.csv"
)

REDUNDANCY_OUTPUT = (
    AUDIT_DIR / "stage_3_9c_exact_feature_redundancy.csv"
)

REDUCED_OUTPUT = (
    AUDIT_DIR / "stage_3_9c_exact_reduced_feature_set.csv"
)

MODEL_COMPARISON_OUTPUT = (
    AUDIT_DIR / "stage_3_9c_exact_model_comparison.csv"
)

AUDIT_JSON_OUTPUT = (
    AUDIT_DIR / "stage_3_9c_exact_feature_audit.json"
)


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_COLUMN = "persistent_damage_target"

TEMPORAL_KEYS = [
    "sequence",
    "component",
    "frame_number",
]

SPATIAL_KEYS = [
    "sequence",
    "component",
    "current_frame",
]

TARGET_KEYS = [
    "sequence",
    "component",
    "current_frame",
]

MERGE_KEYS = [
    "sequence",
    "component",
    "current_frame",
]


# ============================================================================
# PRINT HELPERS
# ============================================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def to_json_safe(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, int):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(value, np.floating):

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(value, np.ndarray):
        return to_json_safe(
            value.tolist()
        )

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, dict):

        return {
            str(key): to_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):

        return [
            to_json_safe(item)
            for item in value
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

def find_existing_file(candidates):

    for filename in candidates:

        path = AUDIT_DIR / filename

        if path.exists():
            return path

    return None


def discover_csv_by_content(
    required_columns,
    preferred_keywords=None,
):

    if preferred_keywords is None:
        preferred_keywords = []

    csv_files = sorted(
        AUDIT_DIR.rglob("*.csv")
    )

    matches = []

    for path in csv_files:

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

        filename = path.name.lower()

        score = 0

        for keyword in preferred_keywords:

            if keyword.lower() in filename:
                score += 10

        matches.append(
            (
                score,
                str(path),
                path,
            )
        )

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return matches[0][2]


def resolve_temporal_file():

    candidates = [
        "stage_2_2_temporal_features.csv",
        "stage_2_2_temporal_feature_engineering.csv",
        "stage_2_2_engineered_temporal_features.csv",
        "stage_2_2_temporal_feature_engineering_audit.csv",
    ]

    found = find_existing_file(
        candidates
    )

    if found is not None:
        return found

    return discover_csv_by_content(
        required_columns=[
            "sequence",
            "component",
        ],
        preferred_keywords=[
            "stage_2_2",
            "temporal",
        ],
    )


def resolve_spatial_file():

    candidates = [
        "stage_3_1_spatial_track_features.csv",
        "stage_3_1_spatial_features.csv",
        "stage_3_1_leakage_safe_spatial_features.csv",
    ]

    found = find_existing_file(
        candidates
    )

    if found is not None:
        return found

    return discover_csv_by_content(
        required_columns=[
            "sequence",
            "component",
        ],
        preferred_keywords=[
            "stage_3_1",
            "spatial",
        ],
    )


# ============================================================================
# COLUMN FINDER
# ============================================================================

def find_column(
    df,
    candidates,
    required=True,
):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    if required:

        raise KeyError(
            "Could not find any of these columns: "
            + ", ".join(candidates)
        )

    return None


# ============================================================================
# KEY STANDARDIZATION
# ============================================================================

def standardize_temporal(df):

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

    df["frame_number"] = pd.to_numeric(
        df[frame_col],
        errors="coerce",
    )

    return df


def standardize_spatial(df):

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


def standardize_targets(df):

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
# FEATURE TAXONOMY
# ============================================================================

def classify_feature(feature):

    f = str(feature)

    current_features = {
        "crack_count",
        "crack_area_ratio",
        "largest_crack_area_px",
        "largest_crack_area_ratio",
        "smallest_crack_area_px",
        "mean_crack_area_px",
        "median_crack_area_px",
        "total_crack_area_px",
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
    }

    if f in current_features:
        return "CURRENT"

    if f.startswith("spatial_delta"):
        return "SPATIAL_CHANGE"

    if f.startswith("spatial_lag"):
        return "SPATIAL_LAG"

    if f.startswith("spatial_rolling"):
        return "SPATIAL_ROLLING"

    if f.startswith("spatial_accel"):
        return "SPATIAL_ACCELERATION"

    if f.startswith("spatial_recent"):
        return "SPATIAL_TREND"

    if f.startswith("spatial_"):
        return "SPATIAL_OTHER"

    if f.startswith("lag"):
        return "TEMPORAL_LAG"

    if f.startswith("delta"):
        return "TEMPORAL_CHANGE"

    if f.startswith("accel"):
        return "TEMPORAL_ACCELERATION"

    if f.startswith("rolling"):
        return "TEMPORAL_ROLLING"

    if (
        f.startswith("recent_")
        or f.endswith("_slope")
        or f.endswith("_growth")
    ):
        return "TEMPORAL_TREND"

    return "OTHER"


# ============================================================================
# QUALITY FILTER
# ============================================================================

def build_quality_filter(
    df,
    features,
):

    eligible = []

    missingness_filtered = []

    near_constant_filtered = []

    for feature in features:

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        missing_fraction = float(
            series.isna().mean()
        )

        if missing_fraction > 0.50:

            missingness_filtered.append(
                feature
            )

            continue

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        if unique_count <= 1:

            near_constant_filtered.append(
                feature
            )

            continue

        eligible.append(
            feature
        )

    return (
        eligible,
        missingness_filtered,
        near_constant_filtered,
    )


# ============================================================================
# REDUNDANCY
# ============================================================================

def calculate_redundancy(
    train_df,
    features,
    threshold=0.98,
):

    if len(features) == 0:

        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "spearman_correlation",
            ]
        )

    numeric = train_df[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation = numeric.corr(
        method="spearman"
    )

    records = []

    for i in range(
        len(features)
    ):

        feature_a = features[i]

        for j in range(
            i + 1,
            len(features),
        ):

            feature_b = features[j]

            value = correlation.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(value):
                continue

            value = float(value)

            if abs(value) >= threshold:

                records.append(
                    {
                        "feature_a":
                            feature_a,

                        "feature_b":
                            feature_b,

                        "spearman_correlation":
                            value,
                    }
                )

    return pd.DataFrame(
        records
    )


# ============================================================================
# REDUCED FEATURE SET
# ============================================================================

def build_reduced_feature_set(
    authoritative_features,
    quality_features,
    redundancy_df,
):

    ordered = [
        feature
        for feature
        in authoritative_features
        if feature
        in quality_features
    ]

    removed = set()

    positions = {
        feature: index
        for index, feature
        in enumerate(ordered)
    }

    for _, row in redundancy_df.iterrows():

        feature_a = row[
            "feature_a"
        ]

        feature_b = row[
            "feature_b"
        ]

        if feature_a in removed:
            continue

        if feature_b in removed:
            continue

        if feature_a not in positions:
            continue

        if feature_b not in positions:
            continue

        if (
            positions[feature_a]
            <
            positions[feature_b]
        ):

            removed.add(
                feature_b
            )

        else:

            removed.add(
                feature_a
            )

    reduced = [
        feature
        for feature
        in ordered
        if feature not in removed
    ]

    return reduced


# ============================================================================
# MATRIX PREPARATION
# ============================================================================

def prepare_matrix(
    df,
    features,
):

    x = df[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return x.fillna(0.0)


# ============================================================================
# RANDOM FOREST
# ============================================================================

def train_random_forest(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    )

    model.fit(
        x_train,
        y_train,
    )

    val_prediction = model.predict(
        x_val
    )

    test_prediction = model.predict(
        x_test
    )

    results = {

        "val_accuracy":
            accuracy_score(
                y_val,
                val_prediction,
            ),

        "val_balanced_accuracy":
            balanced_accuracy_score(
                y_val,
                val_prediction,
            ),

        "val_macro_f1":
            f1_score(
                y_val,
                val_prediction,
                average="macro",
                zero_division=0,
            ),

        "test_accuracy":
            accuracy_score(
                y_test,
                test_prediction,
            ),

        "test_balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                test_prediction,
            ),

        "test_macro_f1":
            f1_score(
                y_test,
                test_prediction,
                average="macro",
                zero_division=0,
            ),
    }

    return model, results


# ============================================================================
# MAIN
# ============================================================================

def main():

    header(
        "STAGE 3.9C - EXACT 232-FEATURE MATRIX CONSTRUCTION"
    )

    print(
        "Project:"
    )

    print(
        PROJECT_ROOT
    )

    # ========================================================================
    # INPUT RESOLUTION
    # ========================================================================

    header(
        "CHECKING INPUT FILES"
    )

    temporal_file = (
        resolve_temporal_file()
    )

    spatial_file = (
        resolve_spatial_file()
    )

    input_files = {

        "feature_summary":
            FEATURE_SUMMARY_FILE,

        "temporal":
            temporal_file,

        "spatial":
            spatial_file,

        "targets":
            TARGET_FILE,

        "classification_results":
            CLASSIFICATION_RESULTS_FILE,
    }

    for name, path in input_files.items():

        if path is None:

            print(
                f"{name:<25}: MISSING"
            )

            raise FileNotFoundError(
                f"Could not locate required {name} "
                f"CSV in {AUDIT_DIR}"
            )

        print(
            f"{name:<25}: FOUND"
        )

        print(
            f"  Path: {path}"
        )

    # ========================================================================
    # LOAD
    # ========================================================================

    header(
        "LOADING DATA"
    )

    feature_summary = pd.read_csv(
        FEATURE_SUMMARY_FILE
    )

    temporal = pd.read_csv(
        temporal_file
    )

    spatial = pd.read_csv(
        spatial_file
    )

    targets = pd.read_csv(
        TARGET_FILE
    )

    classification_results = pd.read_csv(
        CLASSIFICATION_RESULTS_FILE
    )

    print(
        "Stage 3.7 feature summary:",
        len(feature_summary),
    )

    print(
        "Temporal records:",
        len(temporal),
    )

    print(
        "Spatial records:",
        len(spatial),
    )

    print(
        "Target records:",
        len(targets),
    )

    # ========================================================================
    # AUTHORITATIVE FEATURE LIST
    # ========================================================================

    header(
        "LOADING AUTHORITATIVE STAGE 3.7 FEATURE LIST"
    )

    feature_column = find_column(
        feature_summary,
        ["feature"],
    )

    authoritative_features = (
        feature_summary[
            feature_column
        ]
        .astype(str)
        .tolist()
    )

    authoritative_features = list(
        dict.fromkeys(
            authoritative_features
        )
    )

    print(
        "Authoritative feature count:",
        len(authoritative_features),
    )

    if len(authoritative_features) != 232:

        raise RuntimeError(
            "Authoritative Stage 3.7 feature list "
            "does not contain exactly 232 features."
        )

    # ========================================================================
    # STANDARDIZE KEYS
    # ========================================================================

    header(
        "STANDARDIZING KEYS"
    )

    temporal = standardize_temporal(
        temporal
    )

    spatial = standardize_spatial(
        spatial
    )

    targets = standardize_targets(
        targets
    )

    # ========================================================================
    # KEY INTEGRITY
    # ========================================================================

    header(
        "KEY INTEGRITY"
    )

    temporal_duplicate_keys = int(
        temporal[
            TEMPORAL_KEYS
        ]
        .duplicated()
        .sum()
    )

    spatial_duplicate_keys = int(
        spatial[
            SPATIAL_KEYS
        ]
        .duplicated()
        .sum()
    )

    target_duplicate_keys = int(
        targets[
            TARGET_KEYS
        ]
        .duplicated()
        .sum()
    )

    print(
        "Temporal duplicate keys:",
        temporal_duplicate_keys,
    )

    print(
        "Spatial duplicate keys:",
        spatial_duplicate_keys,
    )

    print(
        "Target duplicate keys:",
        target_duplicate_keys,
    )

    if temporal_duplicate_keys != 0:

        raise RuntimeError(
            "Temporal duplicate keys detected."
        )

    if spatial_duplicate_keys != 0:

        raise RuntimeError(
            "Spatial duplicate keys detected."
        )

    if target_duplicate_keys != 0:

        raise RuntimeError(
            "Target duplicate keys detected."
        )

    # ========================================================================
    # EXACT SOURCE MAPPING
    # ========================================================================

    header(
        "EXACT 232 FEATURE SOURCE MAPPING"
    )

    temporal_columns = set(
        temporal.columns
    )

    spatial_columns = set(
        spatial.columns
    )

    feature_sources = {}

    missing_features = []

    ambiguous_features = []

    # ------------------------------------------------------------------------
    # IMPORTANT EXPLICIT MAPPING
    #
    # These two features are known Stage 3.7 aliases.
    #
    # current_crack_count_temporal
    #       -> temporal.crack_count
    #
    # current_crack_count_spatial
    #       -> spatial.current_crack_count
    #
    # Do NOT allow the generic mapper to treat these as ambiguous.
    # ------------------------------------------------------------------------

    explicit_mappings = {

        "current_crack_count_temporal":
            (
                "temporal",
                "crack_count",
            ),

        "current_crack_count_spatial":
            (
                "spatial",
                "current_crack_count",
            ),
    }

    for feature in authoritative_features:

        # ================================================================
        # SPECIAL AUTHORITATIVE ALIASES
        # ================================================================

        if feature in explicit_mappings:

            source_type, source_column = (
                explicit_mappings[
                    feature
                ]
            )

            if source_type == "temporal":

                if source_column not in temporal_columns:

                    raise RuntimeError(
                        "Expected temporal source column "
                        f"'{source_column}' not found for "
                        f"'{feature}'."
                    )

            elif source_type == "spatial":

                if source_column not in spatial_columns:

                    raise RuntimeError(
                        "Expected spatial source column "
                        f"'{source_column}' not found for "
                        f"'{feature}'."
                    )

            feature_sources[
                feature
            ] = (
                source_type,
                source_column,
            )

            continue

        # ================================================================
        # NORMAL MAPPING
        # ================================================================

        temporal_match = (
            feature
            if feature in temporal_columns
            else None
        )

        spatial_match = (
            feature
            if feature in spatial_columns
            else None
        )

        if (
            temporal_match is not None
            and spatial_match is None
        ):

            feature_sources[
                feature
            ] = (
                "temporal",
                temporal_match,
            )

        elif (
            spatial_match is not None
            and temporal_match is None
        ):

            feature_sources[
                feature
            ] = (
                "spatial",
                spatial_match,
            )

        elif (
            temporal_match is not None
            and spatial_match is not None
        ):

            # Spatial-prefixed features belong to spatial source.
            if feature.startswith(
                "spatial_"
            ):

                feature_sources[
                    feature
                ] = (
                    "spatial",
                    spatial_match,
                )

            else:

                # For non-spatial duplicate names,
                # temporal source is authoritative.
                feature_sources[
                    feature
                ] = (
                    "temporal",
                    temporal_match,
                )

        else:

            missing_features.append(
                feature
            )

    print(
        "Mapped features:",
        len(feature_sources),
    )

    print(
        "Missing features:",
        len(missing_features),
    )

    print(
        "Ambiguous features:",
        len(ambiguous_features),
    )

    if missing_features:

        print()
        print(
            "Missing:"
        )

        for feature in missing_features:

            print(
                "  -",
                feature,
            )

    if len(feature_sources) != 232:

        raise RuntimeError(
            "Exact 232-feature source mapping failed. "
            f"Mapped {len(feature_sources)} of 232."
        )

    if missing_features:

        raise RuntimeError(
            "Exact 232-feature source mapping failed."
        )

    print()
    print(
        "Special alias mapping:"
    )

    print(
        "  current_crack_count_temporal"
        " -> temporal.crack_count"
    )

    print(
        "  current_crack_count_spatial"
        " -> spatial.current_crack_count"
    )

    # ========================================================================
    # BASE TABLE
    # ========================================================================

    header(
        "BUILDING EXACT BASE TABLE"
    )

    base = targets[
        TARGET_KEYS
        + [
            "split",
            TARGET_COLUMN,
        ]
    ].copy()

    base = base.drop_duplicates(
        subset=TARGET_KEYS
    ).reset_index(
        drop=True
    )

    print(
        "Base transition records:",
        len(base),
    )

    # ========================================================================
    # ADD FEATURES
    # ========================================================================

    header(
        "ADDING 232 FEATURES INDIVIDUALLY"
    )

    exact_table = base.copy()

    for index, feature in enumerate(
        authoritative_features,
        start=1,
    ):

        source_type, source_column = (
            feature_sources[
                feature
            ]
        )

        if source_type == "temporal":

            source = temporal[
                TEMPORAL_KEYS
                + [source_column]
            ].copy()

            source = source.rename(
                columns={
                    "frame_number":
                        "current_frame",

                    source_column:
                        "__feature_value__",
                }
            )

        elif source_type == "spatial":

            source = spatial[
                SPATIAL_KEYS
                + [source_column]
            ].copy()

            source = source.rename(
                columns={
                    source_column:
                        "__feature_value__",
                }
            )

        else:

            raise RuntimeError(
                f"Unknown source type: {source_type}"
            )

        source = source[
            MERGE_KEYS
            + [
                "__feature_value__"
            ]
        ]

        source = source.drop_duplicates(
            subset=MERGE_KEYS
        )

        exact_table = (
            exact_table.merge(
                source,
                on=MERGE_KEYS,
                how="left",
                validate="one_to_one",
            )
        )

        exact_table[
            feature
        ] = exact_table[
            "__feature_value__"
        ]

        exact_table = (
            exact_table.drop(
                columns=[
                    "__feature_value__"
                ]
            )
        )

        if index % 50 == 0:

            print(
                f"Added: {index} / 232"
            )

    # ========================================================================
    # EXACT MATRIX VALIDATION
    # ========================================================================

    header(
        "EXACT FEATURE MATRIX VALIDATION"
    )

    metadata_columns = (
        MERGE_KEYS
        + [
            "split",
            TARGET_COLUMN,
        ]
    )

    actual_feature_columns = [
        column
        for column
        in exact_table.columns
        if column
        not in metadata_columns
    ]

    expected_set = set(
        authoritative_features
    )

    actual_set = set(
        actual_feature_columns
    )

    missing = sorted(
        expected_set
        - actual_set
    )

    unexpected = sorted(
        actual_set
        - expected_set
    )

    duplicate_names = (
        exact_table.columns[
            exact_table.columns.duplicated()
        ]
        .tolist()
    )

    print(
        "Expected features:",
        len(authoritative_features),
    )

    print(
        "Actual features:",
        len(actual_feature_columns),
    )

    print(
        "Missing:",
        len(missing),
    )

    print(
        "Unexpected:",
        len(unexpected),
    )

    print(
        "Duplicate feature names:",
        len(duplicate_names),
    )

    if missing:

        print()
        print(
            "Missing:"
        )

        for feature in missing:

            print(
                "  -",
                feature,
            )

    if unexpected:

        print()
        print(
            "Unexpected:"
        )

        for feature in unexpected:

            print(
                "  -",
                feature,
            )

    exact_matrix_pass = (
        len(authoritative_features)
        == 232
        and
        len(actual_feature_columns)
        == 232
        and
        len(missing)
        == 0
        and
        len(unexpected)
        == 0
        and
        len(duplicate_names)
        == 0
    )

    if not exact_matrix_pass:

        raise RuntimeError(
            "EXACT 232-FEATURE MATRIX VALIDATION FAILED."
        )

    print()
    print(
        "EXACT 232-FEATURE MATRIX: PASS"
    )

    print(
        "Classification records:",
        len(exact_table),
    )

    # ========================================================================
    # TARGET VALIDATION
    # ========================================================================

    if TARGET_COLUMN not in exact_table.columns:

        raise RuntimeError(
            "Classification target missing."
        )

    missing_target_values = int(
        exact_table[
            TARGET_COLUMN
        ]
        .isna()
        .sum()
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        exact_table[
            TARGET_COLUMN
        ]
        .value_counts()
    )

    if missing_target_values != 0:

        raise RuntimeError(
            "Missing target values detected."
        )

    # ========================================================================
    # TAXONOMY
    # ========================================================================

    header(
        "BUILDING FEATURE TAXONOMY"
    )

    taxonomy_records = []

    for feature in authoritative_features:

        source_type, source_column = (
            feature_sources[
                feature
            ]
        )

        taxonomy_records.append(
            {
                "feature":
                    feature,

                "group":
                    classify_feature(
                        feature
                    ),

                "source":
                    source_type,

                "source_column":
                    source_column,
            }
        )

    taxonomy_df = pd.DataFrame(
        taxonomy_records
    )

    print(
        taxonomy_df[
            "group"
        ].value_counts()
    )

    taxonomy_df.to_csv(
        TAXONOMY_OUTPUT,
        index=False,
    )

    # ========================================================================
    # QUALITY
    # ========================================================================

    header(
        "FEATURE QUALITY AUDIT"
    )

    train_mask = (
        exact_table[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "train"
    )

    train_df = exact_table[
        train_mask
    ].copy()

    (
        quality_features,
        missingness_filtered,
        near_constant_filtered,
    ) = build_quality_filter(
        train_df,
        authoritative_features,
    )

    print(
        "Missingness-filtered:",
        len(missingness_filtered),
    )

    print(
        "Near-constant-filtered:",
        len(near_constant_filtered),
    )

    print(
        "Quality-eligible:",
        len(quality_features),
    )

    # ========================================================================
    # REDUNDANCY
    # ========================================================================

    header(
        "CORRELATION REDUNDANCY ANALYSIS"
    )

    redundancy_df = calculate_redundancy(
        train_df,
        quality_features,
        threshold=0.98,
    )

    print(
        "Highly correlated feature pairs:",
        len(redundancy_df),
    )

    print(
        "Correlation threshold:",
        0.98,
    )

    redundancy_df.to_csv(
        REDUNDANCY_OUTPUT,
        index=False,
    )

    # ========================================================================
    # REDUCED FEATURES
    # ========================================================================

    header(
        "BUILDING REDUCED INTERPRETABLE FEATURE SET"
    )

    reduced_features = (
        build_reduced_feature_set(
            authoritative_features,
            quality_features,
            redundancy_df,
        )
    )

    removed_features = [
        feature
        for feature
        in authoritative_features
        if feature
        not in reduced_features
    ]

    print(
        "Full features:",
        len(authoritative_features),
    )

    print(
        "Quality-eligible:",
        len(quality_features),
    )

    print(
        "Reduced features:",
        len(reduced_features),
    )

    print(
        "Removed:",
        len(removed_features),
    )

    reduced_output = pd.DataFrame(
        {
            "feature":
                reduced_features,

            "group":
                [
                    classify_feature(
                        feature
                    )
                    for feature
                    in reduced_features
                ],
        }
    )

    reduced_output.to_csv(
        REDUCED_OUTPUT,
        index=False,
    )

    # ========================================================================
    # FEATURE GROUPS
    # ========================================================================

    current_features = [
        feature
        for feature
        in authoritative_features
        if classify_feature(
            feature
        )
        == "CURRENT"
    ]

    temporal_groups = {
        "TEMPORAL_LAG",
        "TEMPORAL_CHANGE",
        "TEMPORAL_ACCELERATION",
        "TEMPORAL_ROLLING",
        "TEMPORAL_TREND",
    }

    spatial_groups = {
        "SPATIAL_LAG",
        "SPATIAL_CHANGE",
        "SPATIAL_ROLLING",
        "SPATIAL_ACCELERATION",
        "SPATIAL_TREND",
        "SPATIAL_OTHER",
    }

    temporal_features = [
        feature
        for feature
        in authoritative_features
        if classify_feature(
            feature
        )
        in temporal_groups
    ]

    spatial_features = [
        feature
        for feature
        in authoritative_features
        if classify_feature(
            feature
        )
        in spatial_groups
    ]

    full_features = list(
        authoritative_features
    )

    print()
    print(
        "Feature group sizes:"
    )

    print(
        "CURRENT:",
        len(current_features),
    )

    print(
        "TEMPORAL:",
        len(temporal_features),
    )

    print(
        "SPATIAL:",
        len(spatial_features),
    )

    print(
        "FULL:",
        len(full_features),
    )

    print(
        "REDUCED:",
        len(reduced_features),
    )

    # ========================================================================
    # SPLITS
    # ========================================================================

    header(
        "PREPARING TRAIN / VAL / TEST"
    )

    train_df = exact_table[
        exact_table[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "train"
    ].copy()

    val_df = exact_table[
        exact_table[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "val"
    ].copy()

    test_df = exact_table[
        exact_table[
            "split"
        ]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    print(
        "TRAIN:",
        len(train_df),
    )

    print(
        "VAL:",
        len(val_df),
    )

    print(
        "TEST:",
        len(test_df),
    )

    # ========================================================================
    # TARGET VECTORS
    # ========================================================================

    y_train = (
        train_df[
            TARGET_COLUMN
        ]
        .astype(str)
    )

    y_val = (
        val_df[
            TARGET_COLUMN
        ]
        .astype(str)
    )

    y_test = (
        test_df[
            TARGET_COLUMN
        ]
        .astype(str)
    )

    # ========================================================================
    # EXPERIMENTS
    # ========================================================================

    header(
        "RUNNING EXACT FEATURE EXPERIMENTS"
    )

    experiment_sets = {

        "CURRENT_ONLY":
            current_features,

        "CURRENT_PLUS_TEMPORAL":
            list(
                dict.fromkeys(
                    current_features
                    + temporal_features
                )
            ),

        "CURRENT_PLUS_SPATIAL":
            list(
                dict.fromkeys(
                    current_features
                    + spatial_features
                )
            ),

        "FULL_EXACT_232":
            full_features,

        "REDUCED_INTERPRETABLE":
            reduced_features,
    }

    model_results = []

    for experiment_name, features in (
        experiment_sets.items()
    ):

        print()
        print(
            experiment_name
        )

        print(
            "  Features:",
            len(features),
        )

        x_train = prepare_matrix(
            train_df,
            features,
        )

        x_val = prepare_matrix(
            val_df,
            features,
        )

        x_test = prepare_matrix(
            test_df,
            features,
        )

        _, metrics = (
            train_random_forest(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
            )
        )

        print(
            "  VAL Macro F1:",
            f"{metrics['val_macro_f1']:.6f}",
        )

        print(
            "  VAL Balanced Accuracy:",
            f"{metrics['val_balanced_accuracy']:.6f}",
        )

        print(
            "  TEST Macro F1:",
            f"{metrics['test_macro_f1']:.6f}",
        )

        print(
            "  TEST Balanced Accuracy:",
            f"{metrics['test_balanced_accuracy']:.6f}",
        )

        print(
            "  TEST Accuracy:",
            f"{metrics['test_accuracy']:.6f}",
        )

        model_results.append(
            {
                "feature_set":
                    experiment_name,

                "feature_count":
                    len(features),

                "train_records":
                    len(train_df),

                "val_records":
                    len(val_df),

                "test_records":
                    len(test_df),

                **metrics,
            }
        )

    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================

    header(
        "STAGE 3.9C MODEL COMPARISON"
    )

    model_comparison = pd.DataFrame(
        model_results
    )

    model_comparison = (
        model_comparison
        .sort_values(
            "test_macro_f1",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        model_comparison.to_string(
            index=False
        )
    )

    model_comparison.to_csv(
        MODEL_COMPARISON_OUTPUT,
        index=False,
    )

    full_row = (
        model_comparison[
            model_comparison[
                "feature_set"
            ]
            == "FULL_EXACT_232"
        ]
        .iloc[0]
    )

    reduced_row = (
        model_comparison[
            model_comparison[
                "feature_set"
            ]
            == "REDUCED_INTERPRETABLE"
        ]
        .iloc[0]
    )

    full_macro_f1 = float(
        full_row[
            "test_macro_f1"
        ]
    )

    reduced_macro_f1 = float(
        reduced_row[
            "test_macro_f1"
        ]
    )

    if full_macro_f1 != 0:

        retention = (
            reduced_macro_f1
            / full_macro_f1
            * 100.0
        )

    else:

        retention = 0.0

    macro_difference = (
        reduced_macro_f1
        - full_macro_f1
    )

    print()
    print(
        "FULL EXACT 232 TEST Macro F1:",
        f"{full_macro_f1:.6f}",
    )

    print(
        "REDUCED TEST Macro F1:",
        f"{reduced_macro_f1:.6f}",
    )

    print(
        "Reduced Macro F1 retention:",
        f"{retention:.2f}%",
    )

    print(
        "Macro F1 difference:",
        f"{macro_difference:.6f}",
    )

    # ========================================================================
    # FINAL AUDIT
    # ========================================================================

    header(
        "FINAL STAGE 3.9C AUDIT"
    )

    split_counts = (
        exact_table[
            "split"
        ]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    taxonomy_counts = (
        taxonomy_df[
            "group"
        ]
        .value_counts()
        .to_dict()
    )

    checks = {

        "authoritative_feature_count_is_232":
            len(authoritative_features)
            == 232,

        "actual_feature_count_is_232":
            len(actual_feature_columns)
            == 232,

        "no_missing_features":
            len(missing)
            == 0,

        "no_unexpected_features":
            len(unexpected)
            == 0,

        "no_duplicate_feature_names":
            len(duplicate_names)
            == 0,

        "exact_feature_name_match":
            set(actual_feature_columns)
            == set(
                authoritative_features
            ),

        "no_temporal_duplicate_keys":
            temporal_duplicate_keys
            == 0,

        "no_spatial_duplicate_keys":
            spatial_duplicate_keys
            == 0,

        "no_target_duplicate_keys":
            target_duplicate_keys
            == 0,

        "no_missing_target_values":
            missing_target_values
            == 0,

        "classification_records_are_14029":
            len(exact_table)
            == 14029,

        "train_records_are_9777":
            int(
                split_counts.get(
                    "train",
                    0,
                )
            )
            == 9777,

        "val_records_are_2094":
            int(
                split_counts.get(
                    "val",
                    0,
                )
            )
            == 2094,

        "test_records_are_2158":
            int(
                split_counts.get(
                    "test",
                    0,
                )
            )
            == 2158,

        "reduced_is_subset":
            set(
                reduced_features
            ).issubset(
                set(
                    authoritative_features
                )
            ),

        "reduced_is_quality_eligible":
            set(
                reduced_features
            ).issubset(
                set(
                    quality_features
                )
            ),

        "full_model_present":
            "FULL_EXACT_232"
            in set(
                model_comparison[
                    "feature_set"
                ]
            ),

        "reduced_model_present":
            "REDUCED_INTERPRETABLE"
            in set(
                model_comparison[
                    "feature_set"
                ]
            ),
    }

    all_checks_pass = all(
        bool(value)
        for value in checks.values()
    )

    # ========================================================================
    # AUDIT JSON
    # ========================================================================

    audit = {

        "stage":
            "3.9C",

        "title":
            "Exact Stage 3.7 Feature Taxonomy and Reduced Feature Set",

        "project_root":
            str(PROJECT_ROOT),

        "resolved_input_files": {

            "feature_summary":
                str(
                    FEATURE_SUMMARY_FILE
                ),

            "temporal":
                str(
                    temporal_file
                ),

            "spatial":
                str(
                    spatial_file
                ),

            "targets":
                str(
                    TARGET_FILE
                ),

            "classification_results":
                str(
                    CLASSIFICATION_RESULTS_FILE
                ),
        },

        "authoritative_feature_count":
            len(
                authoritative_features
            ),

        "actual_feature_count":
            len(
                actual_feature_columns
            ),

        "mapped_features":
            len(
                feature_sources
            ),

        "missing_features":
            missing,

        "unexpected_features":
            unexpected,

        "duplicate_feature_names":
            duplicate_names,

        "classification_records":
            len(
                exact_table
            ),

        "split_counts":
            split_counts,

        "target_distribution":
            exact_table[
                TARGET_COLUMN
            ]
            .value_counts()
            .to_dict(),

        "taxonomy_counts":
            taxonomy_counts,

        "missingness_filtered_count":
            len(
                missingness_filtered
            ),

        "near_constant_filtered_count":
            len(
                near_constant_filtered
            ),

        "quality_eligible_count":
            len(
                quality_features
            ),

        "highly_correlated_feature_pairs":
            len(
                redundancy_df
            ),

        "correlation_threshold":
            0.98,

        "full_feature_count":
            len(
                full_features
            ),

        "reduced_feature_count":
            len(
                reduced_features
            ),

        "removed_feature_count":
            len(
                removed_features
            ),

        "reduced_features":
            reduced_features,

        "removed_features":
            removed_features,

        "model_results":
            model_results,

        "full_test_macro_f1":
            full_macro_f1,

        "reduced_test_macro_f1":
            reduced_macro_f1,

        "reduced_macro_f1_retention_percent":
            retention,

        "reduced_minus_full_macro_f1":
            macro_difference,

        "checks":
            checks,

        "final_status":
            (
                "PASS"
                if all_checks_pass
                else "REVIEW"
            ),
    }

    audit_safe = to_json_safe(
        audit
    )

    with open(
        AUDIT_JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit_safe,
            file,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )

    print()
    print(
        "Audit JSON saved:"
    )

    print(
        AUDIT_JSON_OUTPUT
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print(
        "Authoritative features:",
        len(
            authoritative_features
        ),
    )

    print(
        "Actual features:",
        len(
            actual_feature_columns
        ),
    )

    print(
        "Reduced features:",
        len(
            reduced_features
        ),
    )

    print(
        "Full TEST Macro F1:",
        f"{full_macro_f1:.6f}",
    )

    print(
        "Reduced TEST Macro F1:",
        f"{reduced_macro_f1:.6f}",
    )

    print(
        "Macro F1 retention:",
        f"{retention:.2f}%",
    )

    print()

    if all_checks_pass:

        print(
            "FINAL STATUS: PASS"
        )

        print()
        print(
            "NEXT: STAGE 3.10 - CLASSIFICATION ERROR ANALYSIS"
        )

    else:

        print(
            "FINAL STATUS: REVIEW"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()