import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.8
# CLASSIFICATION FEATURE IMPORTANCE & ABLATION
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

TARGET_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_6_persistent_damage_targets.csv"
)

TEMPORAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_2_2_temporal_features.csv"
)

SPATIAL_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_1_spatial_track_features.csv"
)

OUTPUT_IMPORTANCE = os.path.join(
    AUDIT_DIR,
    "stage_3_8_classification_feature_importance.csv"
)

OUTPUT_ABLATION = os.path.join(
    AUDIT_DIR,
    "stage_3_8_classification_ablation_results.csv"
)

OUTPUT_GROUP_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_8_feature_group_summary.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_8_classification_feature_importance_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path, description):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def require_columns(df, columns, name):

    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing)
        )


def make_feature_groups(features):

    groups = {

        "CURRENT": [],

        "TEMPORAL_LAG": [],
        "TEMPORAL_CHANGE": [],
        "TEMPORAL_ACCELERATION": [],
        "TEMPORAL_ROLLING": [],
        "TEMPORAL_TREND": [],

        "SPATIAL_LAG": [],
        "SPATIAL_CHANGE": [],
        "SPATIAL_ACCELERATION": [],
        "SPATIAL_ROLLING": [],
        "SPATIAL_TREND": [],

        "OTHER": []

    }

    for feature in features:

        lower = feature.lower()

        # ----------------------------------------------------
        # SPATIAL FEATURES
        # ----------------------------------------------------

        if lower.startswith(
            "spatial_acceleration"
        ):

            groups[
                "SPATIAL_ACCELERATION"
            ].append(feature)

        elif lower.startswith(
            "spatial_rolling"
        ):

            groups[
                "SPATIAL_ROLLING"
            ].append(feature)

        elif lower.startswith(
            "spatial_trend"
        ):

            groups[
                "SPATIAL_TREND"
            ].append(feature)

        elif lower.startswith(
            "spatial_lag"
        ):

            groups[
                "SPATIAL_LAG"
            ].append(feature)

        elif (
            lower.startswith(
                "spatial_delta"
            )
            or
            lower.startswith(
                "spatial_recent"
            )
        ):

            groups[
                "SPATIAL_CHANGE"
            ].append(feature)

        # ----------------------------------------------------
        # TEMPORAL FEATURES
        # ----------------------------------------------------

        elif lower.startswith(
            "lag"
        ):

            groups[
                "TEMPORAL_LAG"
            ].append(feature)

        elif lower.startswith(
            "delta"
        ):

            groups[
                "TEMPORAL_CHANGE"
            ].append(feature)

        elif lower.startswith(
            "accel"
        ):

            groups[
                "TEMPORAL_ACCELERATION"
            ].append(feature)

        elif lower.startswith(
            "rolling"
        ):

            groups[
                "TEMPORAL_ROLLING"
            ].append(feature)

        elif (
            lower.startswith(
                "recent_"
            )
            or
            "slope" in lower
            or
            "growth" in lower
        ):

            groups[
                "TEMPORAL_TREND"
            ].append(feature)

        # ----------------------------------------------------
        # CURRENT FEATURES
        # ----------------------------------------------------

        elif (
            lower.startswith(
                "current_"
            )
            or
            lower in [

                "crack_area_ratio",

                "crack_count",

                "largest_crack_area_ratio",

                "largest_crack_area_px",

                "total_crack_area_px",

                "mean_crack_area_px",

                "total_crack_perimeter_px",

                "mean_crack_perimeter_px",

                "annotation_shape_count",

                "image_area_px"

            ]
        ):

            groups[
                "CURRENT"
            ].append(feature)

        # ----------------------------------------------------
        # OTHER
        # ----------------------------------------------------

        else:

            groups[
                "OTHER"
            ].append(feature)

    return groups


def get_feature_groups(features):

    groups = make_feature_groups(
        features
    )

    return {
        key: value
        for key, value in groups.items()
        if len(value) > 0
    }


def train_rf(
    X_train,
    y_train,
    X_test
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_leaf=2,

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1

    )

    model.fit(
        X_train_imp,
        y_train
    )

    predictions = model.predict(
        X_test_imp
    )

    return (
        model,
        imputer,
        predictions
    )


def evaluate_predictions(
    y_true,
    predictions
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            )

    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.8 - CLASSIFICATION FEATURE IMPORTANCE & ABLATION"
)
print("=" * 70)


# ============================================================
# FILE CHECK
# ============================================================

print(
    "\nChecking required files..."
)

require_file(
    TARGET_FILE,
    "Stage 3.6 target file"
)

require_file(
    TEMPORAL_FILE,
    "Stage 2.2 temporal feature file"
)

require_file(
    SPATIAL_FILE,
    "Stage 3.1 spatial feature file"
)

print(
    "All required files: FOUND"
)


# ============================================================
# LOAD
# ============================================================

print(
    "\nLoading target..."
)

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


print(
    "\nLoading temporal features..."
)

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
)


print(
    "\nLoading spatial features..."
)

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial records:",
    len(spatial)
)


# ============================================================
# KEY NORMALIZATION
# ============================================================

key_columns = [

    "sequence",

    "component",

    "split",

    "current_frame",

    "future_frame"

]


require_columns(
    targets,
    key_columns,
    "Stage 3.6 target"
)

require_columns(
    targets,
    [
        "persistent_damage_target"
    ],
    "Stage 3.6 target"
)


# ============================================================
# NORMALIZE STAGE 2.2 FRAME COLUMNS
# ============================================================

require_columns(

    temporal,

    [

        "sequence",

        "component",

        "split",

        "frame_number",

        "target_current_frame",

        "target_future_frame"

    ],

    "Stage 2.2 temporal features"

)


temporal[
    "frame_number"
] = pd.to_numeric(

    temporal[
        "frame_number"
    ],

    errors="coerce"

)


temporal[
    "target_current_frame"
] = pd.to_numeric(

    temporal[
        "target_current_frame"
    ],

    errors="coerce"

)


temporal[
    "target_future_frame"
] = pd.to_numeric(

    temporal[
        "target_future_frame"
    ],

    errors="coerce"

)


current_alignment_errors = int(

    (

        temporal[
            "frame_number"
        ]

        !=

        temporal[
            "target_current_frame"
        ]

    )

    .fillna(False)

    .sum()

)


print(
    "\nStage 2.2 current-frame alignment errors:",
    current_alignment_errors
)


if current_alignment_errors != 0:

    raise RuntimeError(
        "Stage 2.2 frame alignment failed."
    )


temporal[
    "current_frame"
] = temporal[
    "frame_number"
]


temporal[
    "future_frame"
] = temporal[
    "target_future_frame"
]


# ============================================================
# SPATIAL SCHEMA
# ============================================================

require_columns(
    spatial,
    key_columns,
    "Stage 3.1 spatial features"
)


# ============================================================
# DUPLICATE KEY AUDIT
# ============================================================

temporal_duplicate_keys = int(

    temporal.duplicated(
        subset=key_columns
    ).sum()

)


spatial_duplicate_keys = int(

    spatial.duplicated(
        subset=key_columns
    ).sum()

)


target_duplicate_keys = int(

    targets.duplicated(
        subset=key_columns
    ).sum()

)


print(
    "\nDuplicate transition keys:"
)

print(
    "  Temporal:",
    temporal_duplicate_keys
)

print(
    "  Spatial:",
    spatial_duplicate_keys
)

print(
    "  Target:",
    target_duplicate_keys
)


if (
    temporal_duplicate_keys
    or spatial_duplicate_keys
    or target_duplicate_keys
):

    raise RuntimeError(
        "Duplicate transition keys detected."
    )


# ============================================================
# MERGE TEMPORAL + SPATIAL
# ============================================================

print(
    "\nMerging temporal + spatial features..."
)


merged = temporal.merge(

    spatial,

    on=key_columns,

    how="inner",

    suffixes=(
        "_temporal",
        "_spatial"
    ),

    validate="one_to_one"

)


print(
    "Temporal + spatial records:",
    len(merged)
)


# ============================================================
# MERGE TARGET
# ============================================================

target_subset = targets[
    key_columns
    +
    [
        "persistent_damage_target"
    ]
].copy()


merged = merged.merge(

    target_subset,

    on=key_columns,

    how="inner",

    validate="one_to_one"

)


print(
    "Final merged records:",
    len(merged)
)


# ============================================================
# LEAKAGE-SAFE FEATURES
# ============================================================

future_tokens = [

    "target",

    "future",

    "damage_future",

    "damage_change",

    "absolute_damage",

    "relative_damage",

    "persistent_damage",

    "rapid_persistent",

    "continuity_class",

    "continuity"

]


metadata_columns = set(

    key_columns

    +

    [

        "record_id",

        "frame_number",

        "target_current_frame",

        "target_future_frame",

        "original_image_filename",

        "original_json_filename",

        "canonical_image_filename",

        "canonical_json_filename",

        "source_image",

        "source_json",

        "persistent_damage_target"

    ]

)


features = []


for column in merged.columns:

    if column in metadata_columns:

        continue

    if not pd.api.types.is_numeric_dtype(
        merged[column]
    ):

        continue

    lower = column.lower()

    if any(
        token in lower
        for token in future_tokens
    ):

        continue

    features.append(
        column
    )


for blocked in [

    "current_frame",

    "future_frame",

    "component"

]:

    if blocked in features:

        features.remove(
            blocked
        )


print(
    "Leakage-safe features:",
    len(features)
)


# ============================================================
# FEATURE GROUPS
# ============================================================

feature_groups = get_feature_groups(
    features
)


print(
    "\nFeature groups:"
)


for group, group_features in feature_groups.items():

    print(
        f"  {group}: {len(group_features)}"
    )


# ============================================================
# MODEL DATA
# ============================================================

model_df = merged[
    key_columns
    +
    [
        "persistent_damage_target"
    ]
    +
    features
].copy()


train_df = model_df[
    model_df[
        "split"
    ]
    ==
    "train"
].copy()


test_df = model_df[
    model_df[
        "split"
    ]
    ==
    "test"
].copy()


train_df = train_df[
    train_df[
        "persistent_damage_target"
    ].notna()
].copy()


test_df = test_df[
    test_df[
        "persistent_damage_target"
    ].notna()
].copy()


y_train = train_df[
    "persistent_damage_target"
]


y_test = test_df[
    "persistent_damage_target"
]


print(
    "\nCommon test records:",
    len(test_df)
)

print(
    "Common training records:",
    len(train_df)
)


# ============================================================
# FULL RANDOM FOREST
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FULL RANDOM FOREST"
)

print(
    "=" * 70
)


full_model, full_imputer, full_predictions = train_rf(

    train_df[
        features
    ],

    y_train,

    test_df[
        features
    ]

)


full_metrics = evaluate_predictions(

    y_test,

    full_predictions

)


print(
    "Test Accuracy:",
    f"{full_metrics['accuracy']:.6f}"
)

print(
    "Test Balanced Accuracy:",
    f"{full_metrics['balanced_accuracy']:.6f}"
)

print(
    "Test Macro F1:",
    f"{full_metrics['macro_f1']:.6f}"
)

print(
    "Test Weighted F1:",
    f"{full_metrics['weighted_f1']:.6f}"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print(
    "\nCalculating permutation importance..."
)


X_test_imp = full_imputer.transform(

    test_df[
        features
    ]

)


permutation = permutation_importance(

    full_model,

    X_test_imp,

    y_test,

    scoring="f1_macro",

    n_repeats=5,

    random_state=42,

    n_jobs=-1

)


# Correct RandomForest attribute access:
#
# full_model.feature_importances_
#
# NOT:
#
# full_model["feature_importances_"]


rf_importances = (
    full_model.feature_importances_
)


importance_rows = []


for index, feature in enumerate(
    features
):

    importance_rows.append({

        "feature":
            feature,

        "permutation_importance_mean":
            float(
                permutation[
                    "importances_mean"
                ][index]
            ),

        "permutation_importance_std":
            float(
                permutation[
                    "importances_std"
                ][index]
            ),

        "random_forest_importance":
            float(
                rf_importances[index]
            )

    })


importance_df = (

    pd.DataFrame(
        importance_rows
    )

    .sort_values(

        "permutation_importance_mean",

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


importance_df[
    "importance_rank"
] = np.arange(

    1,

    len(
        importance_df
    )
    +
    1

)


importance_df.to_csv(

    OUTPUT_IMPORTANCE,

    index=False

)


print(
    "\nTop 25 features:"
)


print(

    importance_df[

        [

            "importance_rank",

            "feature",

            "permutation_importance_mean",

            "permutation_importance_std",

            "random_forest_importance"

        ]

    ]

    .head(25)

    .to_string(
        index=False
    )

)


# ============================================================
# GROUP IMPORTANCE
# ============================================================

group_lookup = {}


for group, group_features in feature_groups.items():

    for feature in group_features:

        group_lookup[
            feature
        ] = group


importance_df[
    "feature_group"
] = importance_df[
    "feature"
].map(
    group_lookup
)


group_summary = (

    importance_df

    .groupby(
        "feature_group",
        dropna=False
    )

    .agg(

        feature_count=(
            "feature",
            "count"
        ),

        total_permutation_importance=(
            "permutation_importance_mean",
            "sum"
        ),

        mean_permutation_importance=(
            "permutation_importance_mean",
            "mean"
        ),

        max_permutation_importance=(
            "permutation_importance_mean",
            "max"
        )

    )

    .reset_index()

    .sort_values(

        "total_permutation_importance",

        ascending=False

    )

)


group_summary.to_csv(

    OUTPUT_GROUP_SUMMARY,

    index=False

)


print(
    "\nFeature-group importance:"
)


print(

    group_summary.to_string(
        index=False
    )

)


# ============================================================
# ABLATION DEFINITIONS
# ============================================================

ablation_definitions = {

    "FULL":
        features,

    "CURRENT_ONLY":
        feature_groups.get(
            "CURRENT",
            []
        ),

    "CURRENT_PLUS_TEMPORAL":
        (

            feature_groups.get(
                "CURRENT",
                []
            )

            +

            feature_groups.get(
                "TEMPORAL_LAG",
                []
            )

            +

            feature_groups.get(
                "TEMPORAL_CHANGE",
                []
            )

            +

            feature_groups.get(
                "TEMPORAL_ACCELERATION",
                []
            )

            +

            feature_groups.get(
                "TEMPORAL_ROLLING",
                []
            )

            +

            feature_groups.get(
                "TEMPORAL_TREND",
                []
            )

        ),

    "CURRENT_PLUS_SPATIAL":
        (

            feature_groups.get(
                "CURRENT",
                []
            )

            +

            feature_groups.get(
                "SPATIAL_LAG",
                []
            )

            +

            feature_groups.get(
                "SPATIAL_CHANGE",
                []
            )

            +

            feature_groups.get(
                "SPATIAL_ACCELERATION",
                []
            )

            +

            feature_groups.get(
                "SPATIAL_ROLLING",
                []
            )

            +

            feature_groups.get(
                "SPATIAL_TREND",
                []
            )

        ),

    "CURRENT_PLUS_TEMPORAL_PLUS_SPATIAL":
        features

}


# ============================================================
# GROUP-REMOVAL ABLATIONS
# ============================================================

for group in [

    "TEMPORAL_LAG",

    "TEMPORAL_CHANGE",

    "TEMPORAL_ACCELERATION",

    "TEMPORAL_ROLLING",

    "TEMPORAL_TREND",

    "SPATIAL_LAG",

    "SPATIAL_CHANGE",

    "SPATIAL_ACCELERATION",

    "SPATIAL_ROLLING",

    "SPATIAL_TREND"

]:

    group_features = feature_groups.get(
        group,
        []
    )


    remaining = [

        feature

        for feature in features

        if feature not in group_features

    ]


    ablation_definitions[
        "WITHOUT_" + group
    ] = remaining


# ============================================================
# ABLATION EXPERIMENT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ABLATION EXPERIMENT"
)

print(
    "=" * 70
)


ablation_results = []


for model_name, model_features in (
    ablation_definitions.items()
):

    model_features = list(
        dict.fromkeys(
            model_features
        )
    )


    if len(model_features) == 0:

        print(
            "\nSkipping empty model:",
            model_name
        )

        continue


    print(
        "\nRunning:",
        model_name
    )


    print(
        "Features:",
        len(model_features)
    )


    model, imputer, predictions = train_rf(

        train_df[
            model_features
        ],

        y_train,

        test_df[
            model_features
        ]

    )


    metrics = evaluate_predictions(

        y_test,

        predictions

    )


    f1_change = (

        metrics[
            "macro_f1"
        ]

        -

        full_metrics[
            "macro_f1"
        ]

    )


    ablation_results.append({

        "configuration":
            model_name,

        "feature_count":
            len(model_features),

        "test_accuracy":
            metrics[
                "accuracy"
            ],

        "test_balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "test_macro_f1":
            metrics[
                "macro_f1"
            ],

        "test_weighted_f1":
            metrics[
                "weighted_f1"
            ],

        "macro_f1_change_vs_full":
            f1_change,

        "macro_f1_drop_vs_full":
            (
                full_metrics[
                    "macro_f1"
                ]

                -

                metrics[
                    "macro_f1"
                ]
            )

    })


    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.6f}"
    )


    print(
        "Balanced Accuracy:",
        f"{metrics['balanced_accuracy']:.6f}"
    )


    print(
        "Accuracy:",
        f"{metrics['accuracy']:.6f}"
    )


ablation_df = pd.DataFrame(
    ablation_results
)


ablation_df = (

    ablation_df

    .sort_values(

        "test_macro_f1",

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


ablation_df.to_csv(

    OUTPUT_ABLATION,

    index=False

)


# ============================================================
# ABLATION RESULTS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "GROUP ABLATION RESULTS"
)

print(
    "=" * 70
)


print(

    ablation_df[

        [

            "configuration",

            "feature_count",

            "test_macro_f1",

            "test_balanced_accuracy",

            "test_accuracy",

            "macro_f1_drop_vs_full"

        ]

    ]

    .to_string(
        index=False
    )

)


# ============================================================
# TEMPORAL / SPATIAL CONTRIBUTION
# ============================================================

lookup = {

    row[
        "configuration"
    ]: row

    for _, row
    in ablation_df.iterrows()

}


full_f1 = full_metrics[
    "macro_f1"
]


current_f1 = lookup.get(
    "CURRENT_ONLY",
    {}
).get(
    "test_macro_f1",
    np.nan
)


current_temporal_f1 = lookup.get(
    "CURRENT_PLUS_TEMPORAL",
    {}
).get(
    "test_macro_f1",
    np.nan
)


current_spatial_f1 = lookup.get(
    "CURRENT_PLUS_SPATIAL",
    {}
).get(
    "test_macro_f1",
    np.nan
)


if np.isfinite(
    current_temporal_f1
) and np.isfinite(
    current_f1
):

    temporal_gain = (

        current_temporal_f1
        -
        current_f1

    )

else:

    temporal_gain = np.nan


if np.isfinite(
    current_spatial_f1
) and np.isfinite(
    current_f1
):

    spatial_gain = (

        current_spatial_f1
        -
        current_f1

    )

else:

    spatial_gain = np.nan


if np.isfinite(
    full_f1
) and np.isfinite(
    current_temporal_f1
):

    additional_spatial_gain = (

        full_f1
        -
        current_temporal_f1

    )

else:

    additional_spatial_gain = np.nan


# ============================================================
# BEST ABLATION
# ============================================================

best_ablation = (

    ablation_df

    .iloc[0]

)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.8",

    "status":
        "PASS",

    "records":
        int(
            len(model_df)
        ),

    "train_records":
        int(
            len(train_df)
        ),

    "test_records":
        int(
            len(test_df)
        ),

    "feature_count":
        int(
            len(features)
        ),

    "current_frame_alignment_errors":
        current_alignment_errors,

    "duplicate_keys": {

        "temporal":
            temporal_duplicate_keys,

        "spatial":
            spatial_duplicate_keys,

        "target":
            target_duplicate_keys

    },

    "full_model": {

        "test_accuracy":
            full_metrics[
                "accuracy"
            ],

        "test_balanced_accuracy":
            full_metrics[
                "balanced_accuracy"
            ],

        "test_macro_f1":
            full_metrics[
                "macro_f1"
            ],

        "test_weighted_f1":
            full_metrics[
                "weighted_f1"
            ]

    },

    "best_ablation": {

        "configuration":
            best_ablation[
                "configuration"
            ],

        "feature_count":
            int(
                best_ablation[
                    "feature_count"
                ]
            ),

        "test_macro_f1":
            float(
                best_ablation[
                    "test_macro_f1"
                ]
            ),

        "test_balanced_accuracy":
            float(
                best_ablation[
                    "test_balanced_accuracy"
                ]
            )

    },

    "feature_group_counts": {

        group:
            len(group_features)

        for group, group_features
        in feature_groups.items()

    },

    "temporal_gain_over_current":
        (
            None
            if not np.isfinite(
                temporal_gain
            )
            else float(
                temporal_gain
            )
        ),

    "spatial_gain_over_current":
        (
            None
            if not np.isfinite(
                spatial_gain
            )
            else float(
                spatial_gain
            )
        ),

    "additional_spatial_gain_over_temporal":
        (
            None
            if not np.isfinite(
                additional_spatial_gain
            )
            else float(
                additional_spatial_gain
            )
        ),

    "output_files": {

        "feature_importance":
            OUTPUT_IMPORTANCE,

        "ablation":
            OUTPUT_ABLATION,

        "group_summary":
            OUTPUT_GROUP_SUMMARY,

        "summary":
            OUTPUT_SUMMARY

    },

    "methodological_notes": [

        "All ablation configurations use the exact same train and test records.",

        "Missing feature values are median-imputed within each configuration.",

        "Stage 3.6 future-derived target information is excluded from predictors.",

        "Stage 3.0 current-to-future continuity values are excluded as current predictors.",

        "Permutation importance is calculated using test-set Macro F1.",

        "Random Forest is used consistently across ablations.",

        "Random Forest impurity importance is reported separately from permutation importance.",

        "Hyperparameter tuning is intentionally deferred."

    ]

}


# ============================================================
# SAVE GROUP SUMMARY AGAIN WITH RANK
# ============================================================

group_summary = group_summary.copy()

group_summary[
    "group_rank"
] = np.arange(

    1,

    len(group_summary)
    +
    1

)


group_summary.to_csv(

    OUTPUT_GROUP_SUMMARY,

    index=False

)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(

    OUTPUT_SUMMARY,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        summary,

        file,

        indent=2

    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "STAGE 3.8 SUMMARY"
)

print(
    "=" * 70
)


print(
    "Total records:",
    len(model_df)
)


print(
    "Training records:",
    len(train_df)
)


print(
    "Test records:",
    len(test_df)
)


print(
    "Leakage-safe features:",
    len(features)
)


print(
    "\nFull Random Forest:"
)


print(
    "  Test Macro F1:",
    f"{full_metrics['macro_f1']:.6f}"
)


print(
    "  Test Balanced Accuracy:",
    f"{full_metrics['balanced_accuracy']:.6f}"
)


print(
    "  Test Accuracy:",
    f"{full_metrics['accuracy']:.6f}"
)


print(
    "\nTemporal gain over current:",
    (
        "N/A"
        if not np.isfinite(
            temporal_gain
        )
        else f"{temporal_gain:.6f}"
    )
)


print(
    "Spatial gain over current:",
    (
        "N/A"
        if not np.isfinite(
            spatial_gain
        )
        else f"{spatial_gain:.6f}"
    )
)


print(
    "Additional spatial gain over temporal:",
    (
        "N/A"
        if not np.isfinite(
            additional_spatial_gain
        )
        else f"{additional_spatial_gain:.6f}"
    )
)


print(
    "\nBest ablation:"
)


print(
    "  Configuration:",
    best_ablation[
        "configuration"
    ]
)


print(
    "  Features:",
    int(
        best_ablation[
            "feature_count"
        ]
    )
)


print(
    "  Macro F1:",
    f"{best_ablation['test_macro_f1']:.6f}"
)


print(
    "  Balanced Accuracy:",
    f"{best_ablation['test_balanced_accuracy']:.6f}"
)


print(
    "\nOutput files:"
)


print(
    OUTPUT_IMPORTANCE
)

print(
    OUTPUT_ABLATION
)

print(
    OUTPUT_GROUP_SUMMARY
)

print(
    OUTPUT_SUMMARY
)


print(
    "\nFINAL STATUS: PASS"
)