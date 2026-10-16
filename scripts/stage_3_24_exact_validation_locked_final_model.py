import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.24
# EXACT 150-FEATURE VALIDATION-LOCKED FINAL MODEL
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

TARGET_FILE = (
    AUDIT_DIR
    / "stage_3_15_physical_regression_targets.csv"
)

TEMPORAL_FILE = (
    AUDIT_DIR
    / "stage_3_17_temporal_physical_features.csv"
)

SPATIAL_FILE = (
    AUDIT_DIR
    / "stage_3_1_spatial_track_features.csv"
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

AUTHORITATIVE_SPATIAL_FILE = (
    AUDIT_DIR
    / "stage_3_23_authoritative_spatial_features.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUT_RANKING = (
    AUDIT_DIR
    / "stage_3_24_exact_validation_spatial_ranking.csv"
)

OUT_COMPARISON = (
    AUDIT_DIR
    / "stage_3_24_exact_validation_feature_set_comparison.csv"
)

OUT_SELECTED = (
    AUDIT_DIR
    / "stage_3_24_exact_locked_spatial_features.csv"
)

OUT_FINAL_FEATURES = (
    AUDIT_DIR
    / "stage_3_24_exact_final_feature_set.csv"
)

OUT_RESULTS = (
    AUDIT_DIR
    / "stage_3_24_exact_final_model_results.csv"
)

OUT_PREDICTIONS = (
    AUDIT_DIR
    / "stage_3_24_exact_final_test_predictions.csv"
)

OUT_SUMMARY = (
    AUDIT_DIR
    / "stage_3_24_exact_validation_locked_final_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
}

TOP_K_VALUES = [
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    80,
    100,
    132,
]

SELECTION_TOLERANCE = 0.01

TARGET_COLUMN = (
    "future_crack_area_ratio_t1"
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


# ============================================================
# HELPERS
# ============================================================

def ensure_exists(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def find_column(
    df,
    candidates,
    label
):

    for c in candidates:

        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find {label}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def check_duplicates(
    df,
    keys,
    name
):

    count = int(
        df.duplicated(keys).sum()
    )

    print(
        f"{name} duplicate keys: {count}"
    )

    if count > 0:

        examples = (
            df.loc[
                df.duplicated(
                    keys,
                    keep=False
                ),
                keys
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{name} contains duplicate keys.\n"
            f"{examples}"
        )


def make_model():

    return GradientBoostingRegressor(
        **GB_PARAMS
    )


def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                )
            ),

        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred
                )
            ),
    }


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


# ============================================================
# START
# ============================================================

print("=" * 80)
print(
    "STAGE 3.24 - EXACT 150-FEATURE "
    "VALIDATION-LOCKED FINAL MODEL"
)
print("=" * 80)

print(
    "\nThis stage uses the authoritative "
    "Stage 3.23 feature taxonomy."
)

print(
    "\nIMPORTANT:"
)

print(
    "TEST SET IS NOT USED FOR FEATURE RANKING."
)

print(
    "TEST SET IS NOT USED FOR FEATURE SELECTION."
)

print(
    "TEST SET IS USED ONLY FOR FINAL EVALUATION."
)


# ============================================================
# INPUT CHECK
# ============================================================

print(
    "\n[1/11] Checking inputs..."
)

for path, label in [

    (
        TARGET_FILE,
        "Physical regression targets"
    ),

    (
        TEMPORAL_FILE,
        "Temporal physical features"
    ),

    (
        SPATIAL_FILE,
        "Spatial track features"
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
        AUTHORITATIVE_SPATIAL_FILE,
        "Authoritative 132 spatial feature list"
    ),
]:

    ensure_exists(
        path,
        label
    )

print(
    "All inputs found."
)


# ============================================================
# LOAD
# ============================================================

print(
    "\n[2/11] Loading datasets..."
)

target = pd.read_csv(
    TARGET_FILE
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

assignments = pd.read_csv(
    ASSIGNMENT_FILE
)

authoritative = pd.read_csv(
    AUTHORITATIVE_SPATIAL_FILE
)

print(
    "Target:",
    len(target)
)

print(
    "Temporal:",
    len(temporal)
)

print(
    "Spatial:",
    len(spatial)
)

print(
    "Manifest:",
    len(manifest)
)

print(
    "Authoritative spatial list:",
    len(authoritative)
)


# ============================================================
# AUTHORITATIVE SPATIAL FEATURES
# ============================================================

print(
    "\n[3/11] Loading exact authoritative "
    "132 spatial features..."
)

feature_column = find_column(
    authoritative,
    ["feature"],
    "authoritative feature"
)

authoritative_spatial = (
    authoritative[
        feature_column
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

# Preserve order while removing duplicates.

authoritative_spatial = list(
    dict.fromkeys(
        authoritative_spatial
    )
)

print(
    "Authoritative spatial features:",
    len(authoritative_spatial)
)

if len(
    authoritative_spatial
) != 132:

    raise RuntimeError(
        "Authoritative spatial feature "
        f"count is {len(authoritative_spatial)}, "
        "expected exactly 132."
    )

missing_spatial_columns = [
    f
    for f in authoritative_spatial
    if f not in spatial.columns
]

if missing_spatial_columns:

    print(
        "\nMissing spatial columns:"
    )

    for f in missing_spatial_columns:
        print(
            "  ",
            f
        )

    raise RuntimeError(
        "Authoritative spatial features "
        "are missing from spatial dataset."
    )

print(
    "All 132 authoritative spatial "
    "features exist in Stage 3.1."
)


# ============================================================
# TARGET
# ============================================================

print(
    "\n[4/11] Preparing target..."
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
        "current_frame"
    ],
    "Target"
)


# ============================================================
# TEMPORAL
# ============================================================

print(
    "\n[5/11] Preparing temporal features..."
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

missing_temporal = [
    f
    for f in TEMPORAL_FEATURES
    if f not in temporal.columns
]

if missing_temporal:

    raise RuntimeError(
        "Missing temporal features:\n"
        + "\n".join(
            missing_temporal
        )
    )

check_duplicates(
    temporal,
    [
        "sequence",
        "current_frame"
    ],
    "Temporal"
)


# ============================================================
# SPATIAL
# ============================================================

print(
    "\n[6/11] Preparing spatial features..."
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
        "current_frame"
    ],
    "Spatial"
)


# ============================================================
# MANIFEST / COMPONENT
# ============================================================

print(
    "\n[7/11] Preparing manifest and "
    "leakage-safe component splits..."
)

manifest["sequence"] = (
    manifest["sequence"]
    .astype(str)
    .str.strip()
)

manifest["current_frame"] = (
    manifest[
        "canonical_image_filename"
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
        "current_frame"
    ],
    "Manifest"
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
        "Component split conflicts detected."
    )


# ============================================================
# MERGE EXACT 150 FEATURE DATASET
# ============================================================

print(
    "\n[8/11] Building exact 150-feature "
    "common modeling dataset..."
)

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
        "current_frame"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    "After temporal:",
    len(base)
)

base = base.merge(
    spatial[
        [
            "sequence",
            "current_frame",
        ]
        +
        authoritative_spatial
    ],
    on=[
        "sequence",
        "current_frame"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    "After exact spatial:",
    len(base)
)

base = base.merge(
    manifest_map,
    on=[
        "sequence",
        "current_frame"
    ],
    how="inner",
    validate="one_to_one"
)

print(
    "After manifest:",
    len(base)
)

base = base.merge(
    component_split,
    on="component_id",
    how="left",
    validate="many_to_one"
)

print(
    "After split assignment:",
    len(base)
)

if base.empty:

    raise RuntimeError(
        "Final merged dataset is empty."
    )

if base[
    "assigned_split"
].isna().any():

    raise RuntimeError(
        "Some records have no assigned split."
    )


# ============================================================
# EXACT FEATURE MATRIX
# ============================================================

all_features = (
    TEMPORAL_FEATURES
    +
    authoritative_spatial
)

if len(
    all_features
) != 150:

    raise RuntimeError(
        f"Expected 150 features, "
        f"got {len(all_features)}."
    )

if len(
    set(all_features)
) != 150:

    raise RuntimeError(
        "Duplicate feature names detected."
    )

for c in (
    all_features
    +
    [TARGET_COLUMN]
):

    base[c] = pd.to_numeric(
        base[c],
        errors="coerce"
    )

base = base.dropna(
    subset=(
        all_features
        +
        [TARGET_COLUMN]
    )
).copy()

print(
    "Complete common records:",
    len(base)
)

if len(base) != 12149:

    raise RuntimeError(
        "Expected 12,149 complete common "
        f"records, got {len(base)}."
    )


# ============================================================
# SPLIT CHECK
# ============================================================

train_df = base[
    base["assigned_split"]
    == "train"
].copy()

val_df = base[
    base["assigned_split"]
    == "val"
].copy()

test_df = base[
    base["assigned_split"]
    == "test"
].copy()

print(
    "\nSplit sizes:"
)

print(
    "TRAIN:",
    len(train_df)
)

print(
    "VAL:",
    len(val_df)
)

print(
    "TEST:",
    len(test_df)
)

if len(train_df) != 8573:
    raise RuntimeError(
        "TRAIN size mismatch."
    )

if len(val_df) != 1716:
    raise RuntimeError(
        "VAL size mismatch."
    )

if len(test_df) != 1860:
    raise RuntimeError(
        "TEST size mismatch."
    )


# ============================================================
# NUMPY MATRICES
# ============================================================

X_train = (
    train_df[
        all_features
    ]
    .to_numpy(
        dtype=float
    )
)

X_val = (
    val_df[
        all_features
    ]
    .to_numpy(
        dtype=float
    )
)

X_test = (
    test_df[
        all_features
    ]
    .to_numpy(
        dtype=float
    )
)

y_train = (
    train_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)

y_val = (
    val_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)

y_test = (
    test_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)

feature_index = {
    feature: i
    for i, feature
    in enumerate(
        all_features
    )
}


# ============================================================
# FULL 150 MODEL
# ============================================================

print(
    "\n[9/11] Training exact 150-feature "
    "model on TRAIN only..."
)

full_model = make_model()

full_model.fit(
    X_train,
    y_train
)

val_full_pred = (
    full_model.predict(
        X_val
    )
)

val_full_metrics = calculate_metrics(
    y_val,
    val_full_pred
)

print(
    "\nEXACT 150 FEATURE VALIDATION:"
)

print(
    "MAE :",
    f"{val_full_metrics['mae']:.8f}"
)

print(
    "RMSE:",
    f"{val_full_metrics['rmse']:.8f}"
)

print(
    "R2  :",
    f"{val_full_metrics['r2']:.8f}"
)


# ============================================================
# VALIDATION PERMUTATION IMPORTANCE
# ============================================================

print(
    "\n[10/11] Ranking spatial features "
    "using VALIDATION ONLY..."
)

permutation = permutation_importance(
    full_model,
    X_val,
    y_val,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

ranking = pd.DataFrame(
    {
        "feature":
            all_features,

        "permutation_importance_mae":
            permutation.importances_mean,

        "permutation_importance_std":
            permutation.importances_std,
    }
)

ranking["feature_group"] = np.where(
    ranking["feature"].isin(
        authoritative_spatial
    ),
    "SPATIAL",
    "TEMPORAL"
)

ranking["absolute_importance"] = (
    ranking[
        "permutation_importance_mae"
    ].abs()
)

ranking = ranking.sort_values(
    "permutation_importance_mae",
    ascending=False
).reset_index(
    drop=True
)

ranking["validation_rank"] = (
    np.arange(
        len(ranking)
    )
    + 1
)

spatial_ranking = (
    ranking[
        ranking[
            "feature_group"
        ]
        == "SPATIAL"
    ]
    .copy()
    .sort_values(
        "permutation_importance_mae",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

spatial_ranking[
    "spatial_rank"
] = (
    np.arange(
        len(
            spatial_ranking
        )
    )
    + 1
)

ranking.to_csv(
    OUT_RANKING,
    index=False
)

print(
    "\nTOP 30 EXACT SPATIAL FEATURES "
    "- VALIDATION:"
)

print(
    spatial_ranking[
        [
            "spatial_rank",
            "feature",
            "permutation_importance_mae",
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# VALIDATION FEATURE COUNT EXPERIMENT
# ============================================================

print(
    "\n[11/11] Validation-locked "
    "feature-count selection..."
)

spatial_ranked_names = (
    spatial_ranking[
        "feature"
    ]
    .tolist()
)

candidate_sets = [
    {
        "candidate":
            "TEMPORAL_ONLY",

        "spatial_k":
            0,

        "features":
            TEMPORAL_FEATURES.copy(),
    }
]

for k in TOP_K_VALUES:

    selected_spatial = (
        spatial_ranked_names[
            :k
        ]
    )

    if len(
        selected_spatial
    ) != k:

        continue

    candidate_sets.append(
        {
            "candidate":
                f"TEMPORAL_PLUS_TOP_{k}_SPATIAL",

            "spatial_k":
                k,

            "features":
                (
                    TEMPORAL_FEATURES
                    +
                    selected_spatial
                ),
        }
    )

candidate_results = []

for candidate in candidate_sets:

    features = candidate[
        "features"
    ]

    indices = [
        feature_index[f]
        for f in features
    ]

    Xtr = (
        X_train[
            :,
            indices
        ]
    )

    Xv = (
        X_val[
            :,
            indices
        ]
    )

    model = make_model()

    model.fit(
        Xtr,
        y_train
    )

    prediction = (
        model.predict(
            Xv
        )
    )

    metrics = calculate_metrics(
        y_val,
        prediction
    )

    candidate_results.append(
        {
            "candidate":
                candidate[
                    "candidate"
                ],

            "spatial_features":
                candidate[
                    "spatial_k"
                ],

            "total_features":
                len(features),

            "validation_mae":
                metrics["mae"],

            "validation_rmse":
                metrics["rmse"],

            "validation_r2":
                metrics["r2"],
        }
    )

    print(
        f"{candidate['candidate']:40s} | "
        f"features={len(features):3d} | "
        f"VAL MAE={metrics['mae']:.8f} | "
        f"VAL R2={metrics['r2']:.6f}"
    )


comparison = pd.DataFrame(
    candidate_results
)

best_validation_mae = float(
    comparison[
        "validation_mae"
    ].min()
)

comparison[
    "within_1pct_of_best"
] = (
    comparison[
        "validation_mae"
    ]
    <=
    best_validation_mae
    *
    (
        1.0
        +
        SELECTION_TOLERANCE
    )
)

eligible = (
    comparison[
        comparison[
            "within_1pct_of_best"
        ]
    ]
    .copy()
)

if eligible.empty:

    raise RuntimeError(
        "No candidate is within "
        "1% of best validation MAE."
    )

# Smallest feature set first.
# Validation MAE is the tie-breaker.

eligible = eligible.sort_values(
    [
        "total_features",
        "validation_mae",
    ],
    ascending=[
        True,
        True,
    ]
)

selected_row = (
    eligible.iloc[0]
)

selected_candidate = (
    selected_row[
        "candidate"
    ]
)

selected_spatial_k = int(
    selected_row[
        "spatial_features"
    ]
)

selected_total_features = int(
    selected_row[
        "total_features"
    ]
)

selected_features = next(
    c["features"]
    for c in candidate_sets
    if c["candidate"]
    == selected_candidate
)

comparison[
    "selected_final_candidate"
] = (
    comparison[
        "candidate"
    ]
    == selected_candidate
)

comparison[
    "best_validation_candidate"
] = (
    comparison[
        "validation_mae"
    ]
    ==
    best_validation_mae
)

comparison.to_csv(
    OUT_COMPARISON,
    index=False
)


# ============================================================
# SAVE LOCKED FEATURE LIST
# ============================================================

selected_spatial_features = (
    spatial_ranking[
        spatial_ranking[
            "feature"
        ].isin(
            selected_features
        )
    ]
    .copy()
)

selected_spatial_features = (
    selected_spatial_features[
        selected_spatial_features[
            "feature"
        ].isin(
            authoritative_spatial
        )
    ]
)

selected_spatial_features.to_csv(
    OUT_SELECTED,
    index=False
)

final_feature_table = pd.DataFrame(
    {
        "feature_order":
            np.arange(
                1,
                len(
                    selected_features
                )
                + 1
            ),

        "feature":
            selected_features,

        "feature_group":
            [
                (
                    "TEMPORAL"
                    if f
                    in TEMPORAL_FEATURES
                    else
                    "SPATIAL"
                )
                for f
                in selected_features
            ],
    }
)

final_feature_table.to_csv(
    OUT_FINAL_FEATURES,
    index=False
)


# ============================================================
# LOCKED FINAL MODEL
# TRAIN + VAL -> TEST
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "LOCKED FINAL MODEL"
)

print(
    "=" * 80
)

print(
    "\nSelected candidate:",
    selected_candidate
)

print(
    "Selected spatial features:",
    selected_spatial_k
)

print(
    "Selected total features:",
    selected_total_features
)

print(
    "Best validation MAE:",
    f"{best_validation_mae:.8f}"
)

print(
    "Selected validation MAE:",
    f"{selected_row['validation_mae']:.8f}"
)


selected_indices = [
    feature_index[f]
    for f in selected_features
]

train_val_df = pd.concat(
    [
        train_df,
        val_df,
    ],
    ignore_index=True
)

X_train_val = (
    train_val_df[
        all_features
    ]
    .to_numpy(
        dtype=float
    )
)

y_train_val = (
    train_val_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=float
    )
)

X_train_val_selected = (
    X_train_val[
        :,
        selected_indices
    ]
)

X_test_selected = (
    X_test[
        :,
        selected_indices
    ]
)

locked_model = make_model()

locked_model.fit(
    X_train_val_selected,
    y_train_val
)

locked_test_pred = (
    locked_model.predict(
        X_test_selected
    )
)

locked_metrics = calculate_metrics(
    y_test,
    locked_test_pred
)


# ============================================================
# PERSISTENCE
# ============================================================

persistence_index = feature_index[
    "crack_area_ratio"
]

persistence_pred = (
    X_test[
        :,
        persistence_index
    ]
)

persistence_metrics = calculate_metrics(
    y_test,
    persistence_pred
)


# ============================================================
# TEMPORAL ONLY
# ============================================================

temporal_indices = [
    feature_index[f]
    for f in TEMPORAL_FEATURES
]

temporal_model = make_model()

temporal_model.fit(
    X_train_val[
        :,
        temporal_indices
    ],
    y_train_val
)

temporal_test_pred = (
    temporal_model.predict(
        X_test[
            :,
            temporal_indices
        ]
    )
)

temporal_metrics = calculate_metrics(
    y_test,
    temporal_test_pred
)


# ============================================================
# FULL 150 BENCHMARK
# ============================================================

full_final_model = make_model()

full_final_model.fit(
    X_train_val,
    y_train_val
)

full_test_pred = (
    full_final_model.predict(
        X_test
    )
)

full_metrics = calculate_metrics(
    y_test,
    full_test_pred
)


# ============================================================
# TEST IMPROVEMENTS
# ============================================================

locked_vs_persistence = (
    1.0
    -
    (
        locked_metrics["mae"]
        /
        persistence_metrics["mae"]
    )
) * 100.0

locked_vs_temporal = (
    1.0
    -
    (
        locked_metrics["mae"]
        /
        temporal_metrics["mae"]
    )
) * 100.0

locked_vs_full = (
    1.0
    -
    (
        locked_metrics["mae"]
        /
        full_metrics["mae"]
    )
) * 100.0


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "model":
                "PERSISTENCE",

            "feature_count":
                1,

            "spatial_features":
                0,

            "test_mae":
                persistence_metrics[
                    "mae"
                ],

            "test_rmse":
                persistence_metrics[
                    "rmse"
                ],

            "test_r2":
                persistence_metrics[
                    "r2"
                ],
        },

        {
            "model":
                "TEMPORAL_ONLY_18",

            "feature_count":
                18,

            "spatial_features":
                0,

            "test_mae":
                temporal_metrics[
                    "mae"
                ],

            "test_rmse":
                temporal_metrics[
                    "rmse"
                ],

            "test_r2":
                temporal_metrics[
                    "r2"
                ],
        },

        {
            "model":
                "LOCKED_FINAL",

            "feature_count":
                selected_total_features,

            "spatial_features":
                selected_spatial_k,

            "test_mae":
                locked_metrics[
                    "mae"
                ],

            "test_rmse":
                locked_metrics[
                    "rmse"
                ],

            "test_r2":
                locked_metrics[
                    "r2"
                ],
        },

        {
            "model":
                "FULL_150_BENCHMARK",

            "feature_count":
                150,

            "spatial_features":
                132,

            "test_mae":
                full_metrics[
                    "mae"
                ],

            "test_rmse":
                full_metrics[
                    "rmse"
                ],

            "test_r2":
                full_metrics[
                    "r2"
                ],
        },
    ]
)

results.to_csv(
    OUT_RESULTS,
    index=False
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions = test_df[
    [
        "sequence",
        "current_frame",
        "component_id",
        "assigned_split",
        TARGET_COLUMN,
    ]
].copy()

test_predictions[
    "prediction_locked_final"
] = locked_test_pred

test_predictions[
    "prediction_full_150"
] = full_test_pred

test_predictions[
    "prediction_temporal_only"
] = temporal_test_pred

test_predictions[
    "prediction_persistence"
] = persistence_pred

test_predictions[
    "locked_absolute_error"
] = (
    np.abs(
        test_predictions[
            TARGET_COLUMN
        ]
        -
        test_predictions[
            "prediction_locked_final"
        ]
    )
)

test_predictions[
    "full_150_absolute_error"
] = (
    np.abs(
        test_predictions[
            TARGET_COLUMN
        ]
        -
        test_predictions[
            "prediction_full_150"
        ]
    )
)

test_predictions.to_csv(
    OUT_PREDICTIONS,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.24",

    "name":
        "Exact 150-Feature "
        "Validation-Locked Final Model",

    "authoritative_feature_source":
        str(
            AUTHORITATIVE_SPATIAL_FILE
        ),

    "methodology":
        {
            "total_features":
                150,

            "temporal_features":
                18,

            "spatial_features":
                132,

            "feature_ranking":
                "VALIDATION_ONLY",

            "feature_selection":
                "VALIDATION_ONLY",

            "test_used_for_selection":
                False,

            "test_used_for_ranking":
                False,

            "final_training":
                "TRAIN_PLUS_VALIDATION",

            "final_evaluation":
                "HELD_OUT_TEST",
        },

    "dataset":
        {
            "common_records":
                int(len(base)),

            "train":
                int(len(train_df)),

            "validation":
                int(len(val_df)),

            "test":
                int(len(test_df)),
        },

    "validation_selection":
        {
            "best_validation_mae":
                float(
                    best_validation_mae
                ),

            "selected_candidate":
                selected_candidate,

            "selected_spatial_features":
                selected_spatial_k,

            "selected_total_features":
                selected_total_features,

            "selected_validation_mae":
                float(
                    selected_row[
                        "validation_mae"
                    ]
                ),

            "selected_validation_rmse":
                float(
                    selected_row[
                        "validation_rmse"
                    ]
                ),

            "selected_validation_r2":
                float(
                    selected_row[
                        "validation_r2"
                    ]
                ),
        },

    "final_test":
        {
            "persistence_mae":
                persistence_metrics[
                    "mae"
                ],

            "persistence_rmse":
                persistence_metrics[
                    "rmse"
                ],

            "persistence_r2":
                persistence_metrics[
                    "r2"
                ],

            "temporal_mae":
                temporal_metrics[
                    "mae"
                ],

            "temporal_rmse":
                temporal_metrics[
                    "rmse"
                ],

            "temporal_r2":
                temporal_metrics[
                    "r2"
                ],

            "locked_final_mae":
                locked_metrics[
                    "mae"
                ],

            "locked_final_rmse":
                locked_metrics[
                    "rmse"
                ],

            "locked_final_r2":
                locked_metrics[
                    "r2"
                ],

            "full_150_mae":
                full_metrics[
                    "mae"
                ],

            "full_150_rmse":
                full_metrics[
                    "rmse"
                ],

            "full_150_r2":
                full_metrics[
                    "r2"
                ],
        },

    "improvements_percent":
        {
            "locked_vs_persistence":
                float(
                    locked_vs_persistence
                ),

            "locked_vs_temporal":
                float(
                    locked_vs_temporal
                ),

            "locked_vs_full_150":
                float(
                    locked_vs_full
                ),
        },

    "outputs":
        {
            "ranking":
                str(
                    OUT_RANKING
                ),

            "comparison":
                str(
                    OUT_COMPARISON
                ),

            "selected_spatial":
                str(
                    OUT_SELECTED
                ),

            "final_features":
                str(
                    OUT_FINAL_FEATURES
                ),

            "results":
                str(
                    OUT_RESULTS
                ),

            "predictions":
                str(
                    OUT_PREDICTIONS
                ),

            "summary":
                str(
                    OUT_SUMMARY
                ),
        },

    "status":
        "PASS",
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
# FINAL PRINT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STAGE 3.24 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nEXACT FEATURE MATRIX"
)

print(
    "Temporal features:",
    18
)

print(
    "Spatial features:",
    132
)

print(
    "Total features:",
    150
)

print(
    "Common records:",
    len(base)
)

print(
    "\nVALIDATION-LOCKED SELECTION"
)

print(
    "Best validation MAE:",
    f"{best_validation_mae:.8f}"
)

print(
    "Selected candidate:",
    selected_candidate
)

print(
    "Selected spatial features:",
    selected_spatial_k
)

print(
    "Selected total features:",
    selected_total_features
)

print(
    "Selected validation MAE:",
    f"{selected_row['validation_mae']:.8f}"
)

print(
    "\nFINAL HELD-OUT TEST"
)

print(
    "Persistence MAE:",
    f"{persistence_metrics['mae']:.8f}"
)

print(
    "Temporal-only MAE:",
    f"{temporal_metrics['mae']:.8f}"
)

print(
    "Locked final MAE:",
    f"{locked_metrics['mae']:.8f}"
)

print(
    "Locked final RMSE:",
    f"{locked_metrics['rmse']:.8f}"
)

print(
    "Locked final R2:",
    f"{locked_metrics['r2']:.8f}"
)

print(
    "Full 150 MAE:",
    f"{full_metrics['mae']:.8f}"
)

print(
    "Full 150 R2:",
    f"{full_metrics['r2']:.8f}"
)

print(
    "\nIMPROVEMENT"
)

print(
    "Locked vs persistence:",
    f"{locked_vs_persistence:.4f}%"
)

print(
    "Locked vs temporal:",
    f"{locked_vs_temporal:.4f}%"
)

print(
    "Locked vs full 150:",
    f"{locked_vs_full:.4f}%"
)

print(
    "\nOutputs:"
)

for path in [
    OUT_RANKING,
    OUT_COMPARISON,
    OUT_SELECTED,
    OUT_FINAL_FEATURES,
    OUT_RESULTS,
    OUT_PREDICTIONS,
    OUT_SUMMARY,
]:

    print(
        path
    )

print(
    "\nFinal status: PASS"
)

print(
    "=" * 80
)