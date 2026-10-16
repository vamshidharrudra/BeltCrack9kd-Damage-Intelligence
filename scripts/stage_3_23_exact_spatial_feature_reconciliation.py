import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.23
# EXACT SPATIAL FEATURE RECONCILIATION AUDIT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

SPATIAL_FILE = (
    AUDIT_DIR
    / "stage_3_1_spatial_track_features.csv"
)

TEMPORAL_FILE = (
    AUDIT_DIR
    / "stage_3_17_temporal_physical_features.csv"
)

TARGET_FILE = (
    AUDIT_DIR
    / "stage_3_15_physical_regression_targets.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "canonical_dataset"
    / "canonical_manifest.csv"
)

ASSIGNMENT_FILE = (
    AUDIT_DIR
    / "stage_0_9d1_component_assignments.csv"
)

# Authoritative Stage 3.20 feature-importance file.
AUTHORITATIVE_FEATURE_FILE = (
    AUDIT_DIR
    / "stage_3_20_spatial_feature_importance.csv"
)

# Stage 3.21 final ranking can also help verify
# what was previously considered spatial.
STAGE_3_21_RANKING_FILE = (
    AUDIT_DIR
    / "stage_3_21_spatial_feature_ranking.csv"
)

OUT_RECONCILIATION = (
    AUDIT_DIR
    / "stage_3_23_spatial_feature_reconciliation.csv"
)

OUT_AUTHORITATIVE = (
    AUDIT_DIR
    / "stage_3_23_authoritative_spatial_features.csv"
)

OUT_SUMMARY = (
    AUDIT_DIR
    / "stage_3_23_exact_spatial_feature_reconciliation_summary.json"
)

OUT_MATRIX_AUDIT = (
    AUDIT_DIR
    / "stage_3_23_exact_150_feature_matrix_audit.csv"
)


TEMPORAL_FEATURES = [
    "crack_area_ratio",
    "lag_1_damage",
    "lag_2_damage",
    "lag_3_damage",
    "lag_5_damage",
    "delta_1_damage",
    "delta_2_damage",
    "delta_3_damage",
    "acceleration_1_damage",
    "acceleration_2_damage",
    "rolling3_mean_damage",
    "rolling3_std_damage",
    "rolling5_mean_damage",
    "rolling5_std_damage",
    "rolling5_min_damage",
    "rolling5_max_damage",
    "recent3_damage_slope",
    "recent5_damage_slope",
]

TARGET_COLUMN = "future_crack_area_ratio_t1"


# ============================================================
# HELPERS
# ============================================================

def ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def find_column(df, candidates, label):

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find {label}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def check_duplicates(df, keys, name):

    count = int(
        df.duplicated(keys).sum()
    )

    print(
        f"{name} duplicate keys: {count}"
    )

    if count > 0:
        raise RuntimeError(
            f"{name} contains duplicate keys."
        )


def extract_frame_number(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if not text:
        return np.nan

    text = text.replace(
        "\\",
        "/"
    )

    basename = text.split("/")[-1]

    stem = Path(
        basename
    ).stem

    if re.fullmatch(
        r"\d+",
        stem
    ):
        return int(stem)

    tokens = re.findall(
        r"\d+",
        stem
    )

    if tokens:
        return int(
            tokens[-1]
        )

    tokens = re.findall(
        r"\d+",
        text
    )

    if tokens:
        return int(
            tokens[-1]
        )

    return np.nan


def detect_current_script_spatial_features(
    spatial
):

    excluded = {
        "sequence",
        "current_frame",
        "future_frame",
        "component",
        "component_id",
        "split",
        "assigned_split",
        "record_id",
        "source_image",
        "source_json",
    }

    features = []

    for c in spatial.columns:

        if c in excluded:
            continue

        lc = c.lower()

        spatial_like = (
            "spatial" in lc
            or c.startswith("current_")
            or c in {
                "track_continuity_quality",
                "persistent_fraction",
                "future_coverage",
                "mean_match_iou",
                "mean_center_distance",
                "mean_area_similarity",
                "matched_crack_count",
                "matched_fraction_current",
                "mean_match_score",
                "spatial_match_quality",
                "continuity_score",
                "track_continuity_quality",
            }
        )

        if spatial_like:

            if pd.api.types.is_numeric_dtype(
                spatial[c]
            ):
                features.append(c)

    return sorted(
        set(features)
    )


def get_authoritative_feature_names(
    df
):

    possible_columns = [
        "feature",
        "Feature",
        "feature_name",
        "FeatureName",
        "name",
    ]

    feature_column = None

    for c in possible_columns:

        if c in df.columns:
            feature_column = c
            break

    if feature_column is None:

        raise KeyError(
            "Could not find feature-name column "
            "in authoritative Stage 3.20 file.\n"
            f"Columns: {list(df.columns)}"
        )

    names = (
        df[feature_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return sorted(
        set(names)
    ), feature_column


# ============================================================
# START
# ============================================================

print("=" * 80)
print(
    "STAGE 3.23 - EXACT SPATIAL FEATURE RECONCILIATION"
)
print("=" * 80)

print(
    "\nPurpose:"
)

print(
    "Resolve the discrepancy between the authoritative "
    "132 spatial features and Stage 3.22's detected 128."
)

print(
    "\nNO MODEL TRAINING."
)

print(
    "NO FEATURE SELECTION."
)

print(
    "NO TEST-BASED OPTIMIZATION."
)


# ============================================================
# INPUT CHECK
# ============================================================

print(
    "\n[1/9] Checking input files..."
)

for path, label in [

    (
        SPATIAL_FILE,
        "Stage 3.1 spatial features"
    ),

    (
        TEMPORAL_FILE,
        "Stage 3.17 temporal features"
    ),

    (
        TARGET_FILE,
        "Stage 3.15 physical targets"
    ),

    (
        MANIFEST_FILE,
        "Canonical manifest"
    ),

    (
        ASSIGNMENT_FILE,
        "Component assignments"
    ),

    (
        AUTHORITATIVE_FEATURE_FILE,
        "Stage 3.20 authoritative feature file"
    ),
]:

    ensure_exists(
        path,
        label
    )

print(
    "All required files found."
)


# ============================================================
# LOAD
# ============================================================

print(
    "\n[2/9] Loading files..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

target = pd.read_csv(
    TARGET_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

assignments = pd.read_csv(
    ASSIGNMENT_FILE
)

authoritative = pd.read_csv(
    AUTHORITATIVE_FEATURE_FILE
)

print(
    "Spatial records:",
    len(spatial)
)

print(
    "Temporal records:",
    len(temporal)
)

print(
    "Target records:",
    len(target)
)

print(
    "Manifest records:",
    len(manifest)
)

print(
    "Authoritative feature rows:",
    len(authoritative)
)


# ============================================================
# AUTHORITATIVE FEATURE LIST
# ============================================================

print(
    "\n[3/9] Recovering authoritative Stage 3.20 feature list..."
)

authoritative_features, auth_feature_column = (
    get_authoritative_feature_names(
        authoritative
    )
)

print(
    "Authoritative feature column:",
    auth_feature_column
)

print(
    "Authoritative unique feature names:",
    len(authoritative_features)
)

# Only retain features that actually exist
# as columns in the Stage 3.1 spatial dataset.

authoritative_existing = [
    f
    for f in authoritative_features
    if f in spatial.columns
]

authoritative_missing_columns = [
    f
    for f in authoritative_features
    if f not in spatial.columns
]

print(
    "Authoritative features existing "
    "as spatial columns:",
    len(authoritative_existing)
)

print(
    "Authoritative features missing "
    "from spatial CSV:",
    len(authoritative_missing_columns)
)

if authoritative_missing_columns:

    print(
        "\nMISSING FROM SPATIAL CSV:"
    )

    for f in authoritative_missing_columns:
        print(
            "  ",
            f
        )

    raise RuntimeError(
        "Some authoritative Stage 3.20 "
        "features do not exist in Stage 3.1 spatial CSV."
    )


# ============================================================
# CURRENT DETECTION
# ============================================================

print(
    "\n[4/9] Reproducing Stage 3.22 spatial detection..."
)

current_features = (
    detect_current_script_spatial_features(
        spatial
    )
)

print(
    "Stage 3.22-style detected features:",
    len(current_features)
)

missing_from_current = sorted(
    set(authoritative_existing)
    -
    set(current_features)
)

unexpected_current = sorted(
    set(current_features)
    -
    set(authoritative_existing)
)

print(
    "Authoritative but NOT detected:",
    len(missing_from_current)
)

for f in missing_from_current:

    print(
        "  MISSING:",
        f
    )

print(
    "Detected but NOT authoritative:",
    len(unexpected_current)
)

for f in unexpected_current:

    print(
        "  UNEXPECTED:",
        f
    )


# ============================================================
# RECONCILIATION TABLE
# ============================================================

print(
    "\n[5/9] Building feature reconciliation table..."
)

all_features = sorted(
    set(
        authoritative_existing
        +
        current_features
    )
)

rows = []

for feature in all_features:

    in_authoritative = (
        feature
        in authoritative_existing
    )

    in_current = (
        feature
        in current_features
    )

    if (
        in_authoritative
        and in_current
    ):
        status = "MATCH"

    elif (
        in_authoritative
        and not in_current
    ):
        status = "MISSING_FROM_STAGE_3_22_DETECTION"

    elif (
        not in_authoritative
        and in_current
    ):
        status = "UNEXPECTED_STAGE_3_22_FEATURE"

    else:
        status = "UNKNOWN"

    rows.append(
        {
            "feature":
                feature,

            "in_authoritative_stage_3_20":
                in_authoritative,

            "in_stage_3_22_detection":
                in_current,

            "status":
                status,
        }
    )

reconciliation = pd.DataFrame(
    rows
)

reconciliation.to_csv(
    OUT_RECONCILIATION,
    index=False
)


# ============================================================
# EXACT AUTHORITATIVE LIST
# ============================================================

print(
    "\n[6/9] Creating exact authoritative "
    "132-feature spatial list..."
)

authoritative_table = pd.DataFrame(
    {
        "spatial_feature_order":
            np.arange(
                1,
                len(
                    authoritative_existing
                ) + 1
            ),

        "feature":
            authoritative_existing,

        "exists_in_spatial_csv":
            True,

        "detected_by_stage_3_22":
            [
                f in current_features
                for f
                in authoritative_existing
            ],
    }
)

authoritative_table.to_csv(
    OUT_AUTHORITATIVE,
    index=False
)

print(
    "Exact authoritative spatial features:",
    len(authoritative_existing)
)


# ============================================================
# KEY / MERGE VALIDATION
# ============================================================

print(
    "\n[7/9] Verifying exact 150-feature "
    "modeling matrix structure..."
)

target["sequence"] = (
    target["sequence"]
    .astype(str)
    .str.strip()
)

target["current_frame"] = pd.to_numeric(
    target["current_frame"],
    errors="coerce"
)

target = target[
    target["current_frame"].notna()
    &
    target[
        TARGET_COLUMN
    ].notna()
].copy()

target["current_frame"] = (
    target["current_frame"]
    .astype(int)
)

check_duplicates(
    target,
    [
        "sequence",
        "current_frame",
    ],
    "Target"
)


temporal["sequence"] = (
    temporal["sequence"]
    .astype(str)
    .str.strip()
)

temporal["current_frame"] = pd.to_numeric(
    temporal["current_frame"],
    errors="coerce"
)

temporal = temporal[
    temporal["current_frame"].notna()
].copy()

temporal["current_frame"] = (
    temporal["current_frame"]
    .astype(int)
)

check_duplicates(
    temporal,
    [
        "sequence",
        "current_frame",
    ],
    "Temporal"
)


spatial["sequence"] = (
    spatial["sequence"]
    .astype(str)
    .str.strip()
)

spatial["current_frame"] = pd.to_numeric(
    spatial["current_frame"],
    errors="coerce"
)

spatial = spatial[
    spatial["current_frame"].notna()
].copy()

spatial["current_frame"] = (
    spatial["current_frame"]
    .astype(int)
)

check_duplicates(
    spatial,
    [
        "sequence",
        "current_frame",
    ],
    "Spatial"
)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

manifest["sequence"] = (
    manifest["sequence"]
    .astype(str)
    .str.strip()
)

manifest_filename_column = (
    "canonical_image_filename"
)

if manifest_filename_column not in (
    manifest.columns
):

    raise KeyError(
        "canonical_image_filename "
        "not found in manifest."
    )

manifest["current_frame"] = (
    manifest[
        manifest_filename_column
    ]
    .apply(
        extract_frame_number
    )
)

manifest = manifest[
    manifest["current_frame"].notna()
].copy()

manifest["current_frame"] = (
    manifest["current_frame"]
    .astype(int)
)

manifest["component_id"] = (
    manifest["component"]
)

manifest_map = (
    manifest[
        [
            "sequence",
            "current_frame",
            "component_id",
        ]
    ]
    .drop_duplicates()
)

check_duplicates(
    manifest_map,
    [
        "sequence",
        "current_frame",
    ],
    "Manifest"
)


# ------------------------------------------------------------
# Component assignments
# ------------------------------------------------------------

assignments["component_id"] = (
    assignments[
        "component_id"
    ]
)

assignments["assigned_split"] = (
    assignments[
        "assigned_split"
    ]
    .astype(str)
    .str.strip()
)

component_split = (
    assignments[
        [
            "component_id",
            "assigned_split",
        ]
    ]
    .drop_duplicates()
)

component_conflicts = (
    component_split
    .groupby(
        "component_id"
    )[
        "assigned_split"
    ]
    .nunique()
)

if (
    component_conflicts > 1
).any():

    raise RuntimeError(
        "Component split conflict."
    )


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

base = target[
    [
        "sequence",
        "current_frame",
        TARGET_COLUMN,
    ]
].copy()

base = base.merge(
    temporal[
        [
            "sequence",
            "current_frame",
        ]
        +
        TEMPORAL_FEATURES
    ],
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After temporal merge:",
    len(base)
)

base = base.merge(
    spatial[
        [
            "sequence",
            "current_frame",
        ]
        +
        authoritative_existing
    ],
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After exact authoritative spatial merge:",
    len(base)
)

base = base.merge(
    manifest_map,
    on=[
        "sequence",
        "current_frame",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    "After manifest merge:",
    len(base)
)

base = base.merge(
    component_split,
    on="component_id",
    how="left",
    validate="many_to_one",
)

print(
    "After split merge:",
    len(base)
)


# ============================================================
# COMPLETE MATRIX
# ============================================================

all_150_features = (
    TEMPORAL_FEATURES
    +
    authoritative_existing
)

print(
    "\nTotal temporal features:",
    len(TEMPORAL_FEATURES)
)

print(
    "Total authoritative spatial features:",
    len(
        authoritative_existing
    )
)

print(
    "Total expected combined features:",
    len(
        all_150_features
    )
)

if len(
    all_150_features
) != 150:

    raise RuntimeError(
        "Expected exactly 150 features "
        f"but obtained {len(all_150_features)}."
    )


for c in (
    all_150_features
    +
    [TARGET_COLUMN]
):

    base[c] = pd.to_numeric(
        base[c],
        errors="coerce"
    )

before_drop = len(base)

base = base.dropna(
    subset=(
        all_150_features
        +
        [TARGET_COLUMN]
    )
).copy()

print(
    "Complete records:",
    len(base),
    "/",
    before_drop
)


# ============================================================
# SPLIT VALIDATION
# ============================================================

print(
    "\n[8/9] Validating exact split structure..."
)

split_counts = (
    base[
        "assigned_split"
    ]
    .value_counts()
    .to_dict()
)

expected = {
    "train": 8573,
    "val": 1716,
    "test": 1860,
}

split_rows = []

for split_name, expected_count in (
    expected.items()
):

    actual = int(
        split_counts.get(
            split_name,
            0
        )
    )

    split_rows.append(
        {
            "split":
                split_name,

            "expected_records":
                expected_count,

            "actual_records":
                actual,

            "difference":
                actual
                - expected_count,

            "status":
                (
                    "PASS"
                    if actual
                    == expected_count
                    else "FAIL"
                ),
        }
    )

split_audit = pd.DataFrame(
    split_rows
)

print(
    split_audit.to_string(
        index=False
    )
)

if (
    split_audit["status"]
    == "FAIL"
).any():

    raise RuntimeError(
        "Split counts do not match "
        "the authoritative common subset."
    )


# ============================================================
# MATRIX AUDIT
# ============================================================

matrix_rows = []

for split_name in [
    "train",
    "val",
    "test",
]:

    subset = base[
        base[
            "assigned_split"
        ]
        == split_name
    ]

    matrix_rows.append(
        {
            "split":
                split_name,

            "records":
                len(subset),

            "feature_count":
                len(
                    all_150_features
                ),

            "missing_values":
                int(
                    subset[
                        all_150_features
                    ]
                    .isna()
                    .sum()
                    .sum()
                ),

            "sequence_count":
                subset[
                    "sequence"
                ].nunique(),

            "component_count":
                subset[
                    "component_id"
                ].nunique(),
        }
    )

matrix_audit = pd.DataFrame(
    matrix_rows
)

matrix_audit.to_csv(
    OUT_MATRIX_AUDIT,
    index=False
)

print(
    "\nExact matrix audit:"
)

print(
    matrix_audit.to_string(
        index=False
    )
)


# ============================================================
# SEQUENCE / COMPONENT INTEGRITY
# ============================================================

sequence_component_conflicts = (
    base[
        [
            "sequence",
            "assigned_split",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "sequence"
    )[
        "assigned_split"
    ]
    .nunique()
)

sequence_conflicts = int(
    (
        sequence_component_conflicts
        > 1
    ).sum()
)

component_conflicts_final = int(
    (
        component_split
        .groupby(
            "component_id"
        )[
            "assigned_split"
        ]
        .nunique()
        > 1
    ).sum()
)

print(
    "\nSequence split conflicts:",
    sequence_conflicts
)

print(
    "Component split conflicts:",
    component_conflicts_final
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n[9/9] Final reconciliation..."
)

checks = {

    "authoritative_spatial_count_132":
        len(
            authoritative_existing
        )
        == 132,

    "combined_feature_count_150":
        len(
            all_150_features
        )
        == 150,

    "complete_records_12149":
        len(base)
        == 12149,

    "train_8573":
        split_counts.get(
            "train",
            0
        )
        == 8573,

    "val_1716":
        split_counts.get(
            "val",
            0
        )
        == 1716,

    "test_1860":
        split_counts.get(
            "test",
            0
        )
        == 1860,

    "sequence_conflicts_zero":
        sequence_conflicts
        == 0,

    "component_conflicts_zero":
        component_conflicts_final
        == 0,

    "no_missing_values":
        int(
            base[
                all_150_features
            ]
            .isna()
            .sum()
            .sum()
        )
        == 0,
}

print()

for name, status in checks.items():

    print(
        f"{name}: {status}"
    )


final_status = (
    "PASS"
    if all(
        checks.values()
    )
    else "REVIEW"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.23",

    "name":
        "Exact Spatial Feature Reconciliation Audit",

    "authoritative_source":
        str(
            AUTHORITATIVE_FEATURE_FILE
        ),

    "authoritative_spatial_features":
        int(
            len(
                authoritative_existing
            )
        ),

    "stage_3_22_detected_spatial_features":
        int(
            len(
                current_features
            )
        ),

    "missing_from_stage_3_22_detection":
        missing_from_current,

    "unexpected_stage_3_22_features":
        unexpected_current,

    "temporal_features":
        int(
            len(
                TEMPORAL_FEATURES
            )
        ),

    "combined_authoritative_features":
        int(
            len(
                all_150_features
            )
        ),

    "common_records":
        int(
            len(base)
        ),

    "split_counts":
        {
            "train":
                int(
                    split_counts.get(
                        "train",
                        0
                    )
                ),

            "val":
                int(
                    split_counts.get(
                        "val",
                        0
                    )
                ),

            "test":
                int(
                    split_counts.get(
                        "test",
                        0
                    )
                ),
        },

    "sequence_conflicts":
        sequence_conflicts,

    "component_conflicts":
        component_conflicts_final,

    "checks":
        checks,

    "status":
        final_status,

    "outputs":
        {
            "reconciliation":
                str(
                    OUT_RECONCILIATION
                ),

            "authoritative_features":
                str(
                    OUT_AUTHORITATIVE
                ),

            "matrix_audit":
                str(
                    OUT_MATRIX_AUDIT
                ),

            "summary":
                str(
                    OUT_SUMMARY
                ),
        },
}


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STAGE 3.23 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nAuthoritative spatial features:",
    len(
        authoritative_existing
    )
)

print(
    "Stage 3.22 detected:",
    len(
        current_features
    )
)

print(
    "Missing from Stage 3.22:",
    len(
        missing_from_current
    )
)

print(
    "Unexpected Stage 3.22:",
    len(
        unexpected_current
    )
)

print(
    "Exact combined feature count:",
    len(
        all_150_features
    )
)

print(
    "Common records:",
    len(base)
)

print(
    "Train:",
    split_counts.get(
        "train",
        0
    )
)

print(
    "Validation:",
    split_counts.get(
        "val",
        0
    )
)

print(
    "Test:",
    split_counts.get(
        "test",
        0
    )
)

print(
    "\nFINAL STATUS:",
    final_status
)

print(
    "\nOutputs:"
)

print(
    OUT_RECONCILIATION
)

print(
    OUT_AUTHORITATIVE
)

print(
    OUT_MATRIX_AUDIT
)

print(
    OUT_SUMMARY
)

print(
    "=" * 80
)