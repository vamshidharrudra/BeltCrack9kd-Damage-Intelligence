import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# STAGE 3.7
# PERSISTENT DAMAGE CLASSIFICATION
# TARGET AUDIT + LEAKAGE-SAFE BASELINE ML
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

OUTPUT_RESULTS = os.path.join(
    AUDIT_DIR,
    "stage_3_7_classification_results.csv"
)

OUTPUT_CLASS_REPORT = os.path.join(
    AUDIT_DIR,
    "stage_3_7_classification_report.csv"
)

OUTPUT_CONFUSION = os.path.join(
    AUDIT_DIR,
    "stage_3_7_confusion_matrices.csv"
)

OUTPUT_DISTRIBUTION = os.path.join(
    AUDIT_DIR,
    "stage_3_7_target_distribution.csv"
)

OUTPUT_FEATURE_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_7_feature_summary.csv"
)

OUTPUT_ALIGNMENT = os.path.join(
    AUDIT_DIR,
    "stage_3_7_alignment_audit.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_7_classification_summary.json"
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
        c for c in columns
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 3.7 - PERSISTENT DAMAGE CLASSIFICATION")
print("TARGET AUDIT + LEAKAGE-SAFE BASELINE ML")
print("=" * 70)


# ============================================================
# FILE CHECK
# ============================================================

print("\nChecking required files...")

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

print("All required files: FOUND")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Stage 3.6 targets...")

targets = pd.read_csv(
    TARGET_FILE
)

print(
    "Target records:",
    len(targets)
)


print("\nLoading Stage 2.2 temporal features...")

temporal = pd.read_csv(
    TEMPORAL_FILE
)

print(
    "Temporal records:",
    len(temporal)
)


print("\nLoading Stage 3.1 spatial features...")

spatial = pd.read_csv(
    SPATIAL_FILE
)

print(
    "Spatial records:",
    len(spatial)
)


# ============================================================
# TARGET SCHEMA
# ============================================================

target_key_columns = [

    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"

]


require_columns(
    targets,
    target_key_columns,
    "Stage 3.6 targets"
)

require_columns(
    targets,
    [
        "persistent_damage_target"
    ],
    "Stage 3.6 targets"
)


# ============================================================
# TEMPORAL SCHEMA
# ============================================================
#
# IMPORTANT:
#
# Stage 2.2 does NOT contain current_frame/future_frame.
#
# Actual Stage 2.2 fields:
#
#   frame_number
#   target_current_frame
#   target_future_frame
#
# Normalize them to:
#
#   current_frame
#   future_frame
#
# only after verifying their relationship.
# ============================================================

temporal_required = [

    "sequence",
    "component",
    "split",
    "frame_number",
    "target_current_frame",
    "target_future_frame"

]


require_columns(
    temporal,
    temporal_required,
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


# Verify target_current_frame corresponds to frame_number.

current_frame_alignment_errors = int(

    (

        temporal[
            "target_current_frame"
        ]

        !=

        temporal[
            "frame_number"
        ]

    )

    .fillna(False)

    .sum()

)


print(
    "\nStage 2.2 current-frame alignment errors:",
    current_frame_alignment_errors
)


if current_frame_alignment_errors != 0:

    raise RuntimeError(
        "Stage 2.2 target_current_frame does not "
        "consistently match frame_number."
    )


# Rename normalized frame fields.

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

spatial_required = [

    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"

]


require_columns(
    spatial,
    spatial_required,
    "Stage 3.1 spatial features"
)


# ============================================================
# TEMPORAL KEY CHECK
# ============================================================

temporal_key_columns = [

    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame"

]


temporal_duplicate_keys = int(

    temporal.duplicated(
        subset=temporal_key_columns
    ).sum()

)


spatial_duplicate_keys = int(

    spatial.duplicated(
        subset=spatial_required
    ).sum()

)


target_duplicate_keys = int(

    targets.duplicated(
        subset=target_key_columns
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


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)


target_counts = (

    targets[
        "persistent_damage_target"
    ]

    .value_counts()

    .sort_index()

)


distribution_rows = []


for target_class, count in target_counts.items():

    distribution_rows.append({

        "scope":
            "ALL",

        "split":
            "ALL",

        "target_class":
            target_class,

        "record_count":
            int(count),

        "percentage":
            float(
                count
                /
                len(targets)
                *
                100.0
            )

    })


for split, group in targets.groupby(
    "split"
):

    counts = (

        group[
            "persistent_damage_target"
        ]

        .value_counts()

        .sort_index()

    )


    for target_class, count in counts.items():

        distribution_rows.append({

            "scope":
                "SPLIT",

            "split":
                split,

            "target_class":
                target_class,

            "record_count":
                int(count),

            "percentage":
                float(
                    count
                    /
                    len(group)
                    *
                    100.0
                )

        })


target_distribution = pd.DataFrame(
    distribution_rows
)


target_distribution.to_csv(
    OUTPUT_DISTRIBUTION,
    index=False
)


print(
    target_counts.to_string()
)


print("\nClass percentages:")


for target_class, count in target_counts.items():

    print(
        f"  {target_class}: "
        f"{count} "
        f"({count / len(targets) * 100:.2f}%)"
    )


# ============================================================
# MERGE TEMPORAL + SPATIAL
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE CONSTRUCTION")
print("=" * 70)


print(
    "\nMerging temporal and spatial features..."
)


merged = temporal.merge(

    spatial,

    on=temporal_key_columns,

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

print(
    "\nMerging Stage 3.6 classification target..."
)


target_subset = targets[
    target_key_columns
    +
    [
        "persistent_damage_target"
    ]
].copy()


merged = merged.merge(

    target_subset,

    on=target_key_columns,

    how="inner",

    validate="one_to_one"

)


print(
    "Records after target merge:",
    len(merged)
)


# ============================================================
# ALIGNMENT AUDIT
# ============================================================

alignment_rows = []


alignment_rows.append({

    "check":
        "Stage 2.2 temporal records",

    "value":
        len(temporal)

})


alignment_rows.append({

    "check":
        "Stage 3.1 spatial records",

    "value":
        len(spatial)

})


alignment_rows.append({

    "check":
        "Stage 3.6 target records",

    "value":
        len(targets)

})


alignment_rows.append({

    "check":
        "Temporal spatial merged records",

    "value":
        len(
            temporal.merge(
                spatial,
                on=temporal_key_columns,
                how="inner"
            )
        )

})


alignment_rows.append({

    "check":
        "Final target-feature merged records",

    "value":
        len(merged)

})


alignment_rows.append({

    "check":
        "Temporal current-frame alignment errors",

    "value":
        current_frame_alignment_errors

})


alignment_rows.append({

    "check":
        "Temporal duplicate keys",

    "value":
        temporal_duplicate_keys

})


alignment_rows.append({

    "check":
        "Spatial duplicate keys",

    "value":
        spatial_duplicate_keys

})


alignment_rows.append({

    "check":
        "Target duplicate keys",

    "value":
        target_duplicate_keys

})


alignment_df = pd.DataFrame(
    alignment_rows
)


alignment_df.to_csv(
    OUTPUT_ALIGNMENT,
    index=False
)


# ============================================================
# TARGET MISSING CHECK
# ============================================================

target_missing = int(

    merged[
        "persistent_damage_target"
    ]
    .isna()
    .sum()

)


print(
    "Missing target values:",
    target_missing
)


# ============================================================
# LEAKAGE-SAFE FEATURE CONSTRUCTION
# ============================================================

print(
    "\nBuilding leakage-safe feature set..."
)


# Future-derived / target-derived tokens.

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

    target_key_columns

    +

    [

        "record_id",

        "original_image_filename",

        "original_json_filename",

        "canonical_image_filename",

        "canonical_json_filename",

        "source_image",

        "source_json",

        "persistent_damage_target",

        "frame_number",

        "target_current_frame",

        "target_future_frame"

    ]

)


candidate_features = []


for col in merged.columns:

    if col in metadata_columns:
        continue

    if not pd.api.types.is_numeric_dtype(
        merged[col]
    ):
        continue

    lower = col.lower()

    blocked = False

    for token in future_tokens:

        if token in lower:

            blocked = True

            break


    if blocked:
        continue


    candidate_features.append(
        col
    )


# Remove numeric identifiers.

for blocked_col in [

    "current_frame",
    "future_frame",
    "component"

]:

    if blocked_col in candidate_features:

        candidate_features.remove(
            blocked_col
        )


print(
    "Candidate leakage-safe features:",
    len(candidate_features)
)


# ============================================================
# EXPLICIT LEAKAGE AUDIT
# ============================================================

leakage_candidates = []


for col in candidate_features:

    lower = col.lower()

    if any(
        token in lower
        for token in future_tokens
    ):

        leakage_candidates.append(
            col
        )


print(
    "Potential future-derived features remaining:",
    len(leakage_candidates)
)


if leakage_candidates:

    print(
        "\nWARNING - potential leakage features:"
    )

    for col in leakage_candidates:

        print(
            " ",
            col
        )

    raise RuntimeError(
        "Leakage audit failed."
    )


# ============================================================
# FEATURE SUMMARY
# ============================================================

feature_summary_rows = []


for col in candidate_features:

    numeric_series = pd.to_numeric(
        merged[col],
        errors="coerce"
    )


    feature_summary_rows.append({

        "feature":
            col,

        "dtype":
            str(
                merged[col].dtype
            ),

        "missing_count":
            int(
                numeric_series.isna().sum()
            ),

        "missing_percentage":
            float(
                numeric_series.isna().mean()
                *
                100.0
            ),

        "unique_values":
            int(
                numeric_series.nunique()
            ),

        "minimum":
            float(
                numeric_series.min()
            ),

        "maximum":
            float(
                numeric_series.max()
            )

    })


feature_summary = pd.DataFrame(
    feature_summary_rows
)


feature_summary.to_csv(
    OUTPUT_FEATURE_SUMMARY,
    index=False
)


# ============================================================
# MODEL DATA
# ============================================================

model_df = merged[
    target_key_columns
    +
    [
        "persistent_damage_target"
    ]
    +
    candidate_features
].copy()


# ============================================================
# SPLITS
# ============================================================

train_df = model_df[
    model_df["split"] == "train"
].copy()


val_df = model_df[
    model_df["split"] == "val"
].copy()


test_df = model_df[
    model_df["split"] == "test"
].copy()


print("\nML split sizes:")

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


# ============================================================
# SEQUENCE / COMPONENT SPLIT AUDIT
# ============================================================

sequence_split_conflicts = int(

    model_df

    .groupby(
        "sequence"
    )[
        "split"
    ]

    .nunique()

    .gt(1)

    .sum()

)


component_split_conflicts = int(

    model_df

    .groupby(
        "component"
    )[
        "split"
    ]

    .nunique()

    .gt(1)

    .sum()

)


print(
    "\nSequence split conflicts:",
    sequence_split_conflicts
)

print(
    "Component split conflicts:",
    component_split_conflicts
)


# ============================================================
# TARGET VALIDATION
# ============================================================

train_df = train_df[
    train_df[
        "persistent_damage_target"
    ].notna()
].copy()


val_df = val_df[
    val_df[
        "persistent_damage_target"
    ].notna()
].copy()


test_df = test_df[
    test_df[
        "persistent_damage_target"
    ].notna()
].copy()


X_train = train_df[
    candidate_features
]

y_train = train_df[
    "persistent_damage_target"
]


X_val = val_df[
    candidate_features
]

y_val = val_df[
    "persistent_damage_target"
]


X_test = test_df[
    candidate_features
]

y_test = test_df[
    "persistent_damage_target"
]


# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_class = (

    y_train

    .value_counts()

    .idxmax()

)


majority_predictions = np.full(

    len(y_test),

    majority_class

)


majority_accuracy = accuracy_score(

    y_test,

    majority_predictions

)


majority_balanced_accuracy = (

    balanced_accuracy_score(

        y_test,

        majority_predictions

    )

)


majority_macro_f1 = f1_score(

    y_test,

    majority_predictions,

    average="macro",

    zero_division=0

)


print("\n")
print("=" * 70)
print("MAJORITY CLASS BASELINE")
print("=" * 70)


print(
    "Majority class:",
    majority_class
)


print(
    "Test accuracy:",
    f"{majority_accuracy:.6f}"
)


print(
    "Test balanced accuracy:",
    f"{majority_balanced_accuracy:.6f}"
)


print(
    "Test macro F1:",
    f"{majority_macro_f1:.6f}"
)


# ============================================================
# MODELS
# ============================================================

models = {

    "LogisticRegression":

        Pipeline(

            steps=[

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "scaler",

                    StandardScaler()
                ),

                (
                    "model",

                    LogisticRegression(

                        max_iter=2000,

                        class_weight="balanced",

                        random_state=42

                    )

                )

            ]

        ),


    "RandomForest":

        Pipeline(

            steps=[

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "model",

                    RandomForestClassifier(

                        n_estimators=300,

                        max_depth=None,

                        min_samples_leaf=2,

                        class_weight=
                        "balanced_subsample",

                        random_state=42,

                        n_jobs=-1

                    )

                )

            ]

        ),


    "GradientBoosting":

        Pipeline(

            steps=[

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "model",

                    GradientBoostingClassifier(

                        n_estimators=200,

                        learning_rate=0.05,

                        max_depth=3,

                        random_state=42

                    )

                )

            ]

        )

}


# ============================================================
# EVALUATION
# ============================================================

results = []

reports = []

confusion_rows = []


def evaluate_model(

    model_name,

    model,

    X_train,

    y_train,

    X_val,

    y_val,

    X_test,

    y_test

):

    print(
        "\nTraining:",
        model_name
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


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )


    val_balanced = balanced_accuracy_score(
        y_val,
        val_pred
    )


    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    val_weighted_f1 = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )


    val_macro_precision = precision_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    val_macro_recall = recall_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )


    test_balanced = balanced_accuracy_score(
        y_test,
        test_pred
    )


    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    test_weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )


    test_macro_precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    test_macro_recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )


    results.append({

        "model":
            model_name,

        "feature_count":
            len(candidate_features),

        "train_records":
            len(y_train),

        "val_records":
            len(y_val),

        "test_records":
            len(y_test),

        "val_accuracy":
            val_accuracy,

        "val_balanced_accuracy":
            val_balanced,

        "val_macro_f1":
            val_macro_f1,

        "val_weighted_f1":
            val_weighted_f1,

        "val_macro_precision":
            val_macro_precision,

        "val_macro_recall":
            val_macro_recall,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced,

        "test_macro_f1":
            test_macro_f1,

        "test_weighted_f1":
            test_weighted_f1,

        "test_macro_precision":
            test_macro_precision,

        "test_macro_recall":
            test_macro_recall

    })


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    labels = sorted(

        set(
            y_test.unique()
        )

        |

        set(
            test_pred
        )

    )


    report = classification_report(

        y_test,

        test_pred,

        labels=labels,

        output_dict=True,

        zero_division=0

    )


    for label in labels:

        row = report.get(
            label,
            {}
        )


        reports.append({

            "model":
                model_name,

            "class":
                label,

            "precision":
                row.get(
                    "precision",
                    0.0
                ),

            "recall":
                row.get(
                    "recall",
                    0.0
                ),

            "f1":
                row.get(
                    "f1-score",
                    0.0
                ),

            "support":
                row.get(
                    "support",
                    0
                )

        })


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_test,

        test_pred,

        labels=labels

    )


    for i, actual_class in enumerate(
        labels
    ):

        for j, predicted_class in enumerate(
            labels
        ):

            confusion_rows.append({

                "model":
                    model_name,

                "actual_class":
                    actual_class,

                "predicted_class":
                    predicted_class,

                "count":
                    int(
                        cm[i, j]
                    )

            })


    print(
        "Validation Macro F1:",
        f"{val_macro_f1:.6f}"
    )


    print(
        "Validation Balanced Accuracy:",
        f"{val_balanced:.6f}"
    )


    print(
        "Test Macro F1:",
        f"{test_macro_f1:.6f}"
    )


    print(
        "Test Balanced Accuracy:",
        f"{test_balanced:.6f}"
    )


    print(
        "Test Accuracy:",
        f"{test_accuracy:.6f}"
    )


    print(
        "\nTest classification report:"
    )


    print(

        classification_report(

            y_test,

            test_pred,

            zero_division=0

        )

    )


# ============================================================
# TRAIN
# ============================================================

for model_name, model in models.items():

    evaluate_model(

        model_name,

        model,

        X_train,

        y_train,

        X_val,

        y_val,

        X_test,

        y_test

    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

reports_df = pd.DataFrame(
    reports
)

confusion_df = pd.DataFrame(
    confusion_rows
)


results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


reports_df.to_csv(
    OUTPUT_CLASS_REPORT,
    index=False
)


confusion_df.to_csv(
    OUTPUT_CONFUSION,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (

    results_df

    .sort_values(

        [
            "val_macro_f1",
            "val_balanced_accuracy"

        ],

        ascending=False

    )

    .iloc[0][
        "model"
    ]

)


best_row = (

    results_df[
        results_df[
            "model"
        ]
        ==
        best_model_name
    ]

    .iloc[0]

)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.7",

    "status":
        "PASS",

    "task":
        "Persistent damage multi-class classification",

    "target":
        "persistent_damage_target",

    "target_records":
        int(
            len(targets)
        ),

    "candidate_feature_count":
        int(
            len(candidate_features)
        ),

    "train_records":
        int(
            len(train_df)
        ),

    "val_records":
        int(
            len(val_df)
        ),

    "test_records":
        int(
            len(test_df)
        ),

    "temporal_records":
        int(
            len(temporal)
        ),

    "spatial_records":
        int(
            len(spatial)
        ),

    "target_feature_merged_records":
        int(
            len(merged)
        ),

    "temporal_current_frame_alignment_errors":
        current_frame_alignment_errors,

    "sequence_split_conflicts":
        sequence_split_conflicts,

    "component_split_conflicts":
        component_split_conflicts,

    "potential_future_features":
        leakage_candidates,

    "majority_class":
        majority_class,

    "majority_test_accuracy":
        float(
            majority_accuracy
        ),

    "majority_test_balanced_accuracy":
        float(
            majority_balanced_accuracy
        ),

    "majority_test_macro_f1":
        float(
            majority_macro_f1
        ),

    "best_validation_model":
        best_model_name,

    "best_validation_macro_f1":
        float(
            best_row[
                "val_macro_f1"
            ]
        ),

    "best_validation_balanced_accuracy":
        float(
            best_row[
                "val_balanced_accuracy"
            ]
        ),

    "best_test_macro_f1":
        float(
            best_row[
                "test_macro_f1"
            ]
        ),

    "best_test_balanced_accuracy":
        float(
            best_row[
                "test_balanced_accuracy"
            ]
        ),

    "best_test_accuracy":
        float(
            best_row[
                "test_accuracy"
            ]
        ),

    "target_distribution": {

        str(key): int(value)

        for key, value

        in target_counts.items()

    },

    "output_files": {

        "results":
            OUTPUT_RESULTS,

        "classification_report":
            OUTPUT_CLASS_REPORT,

        "confusion_matrices":
            OUTPUT_CONFUSION,

        "target_distribution":
            OUTPUT_DISTRIBUTION,

        "feature_summary":
            OUTPUT_FEATURE_SUMMARY,

        "alignment_audit":
            OUTPUT_ALIGNMENT,

        "summary":
            OUTPUT_SUMMARY

    }

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


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3.7 SUMMARY")
print("=" * 70)


print(
    "Target records:",
    len(targets)
)


print(
    "Temporal records:",
    len(temporal)
)


print(
    "Spatial records:",
    len(spatial)
)


print(
    "Final merged records:",
    len(merged)
)


print(
    "Candidate leakage-safe features:",
    len(candidate_features)
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


print(
    "Temporal current-frame alignment errors:",
    current_frame_alignment_errors
)


print(
    "Sequence split conflicts:",
    sequence_split_conflicts
)


print(
    "Component split conflicts:",
    component_split_conflicts
)


print(
    "Potential future features:",
    len(leakage_candidates)
)


print(
    "\nMajority baseline:"
)


print(
    "  Class:",
    majority_class
)


print(
    "  Test accuracy:",
    f"{majority_accuracy:.6f}"
)


print(
    "  Test balanced accuracy:",
    f"{majority_balanced_accuracy:.6f}"
)


print(
    "  Test macro F1:",
    f"{majority_macro_f1:.6f}"
)


print(
    "\nModel results:"
)


print(

    results_df[

        [
            "model",

            "val_macro_f1",

            "val_balanced_accuracy",

            "test_macro_f1",

            "test_balanced_accuracy",

            "test_accuracy"

        ]

    ]

    .sort_values(

        "val_macro_f1",

        ascending=False

    )

    .to_string(
        index=False
    )

)


print(
    "\nBest validation model:",
    best_model_name
)


print(
    "Best validation Macro F1:",
    f"{best_row['val_macro_f1']:.6f}"
)


print(
    "Best test Macro F1:",
    f"{best_row['test_macro_f1']:.6f}"
)


print(
    "Best test Balanced Accuracy:",
    f"{best_row['test_balanced_accuracy']:.6f}"
)


print(
    "\nOutput:"
)


print(
    OUTPUT_RESULTS
)


print(
    OUTPUT_CLASS_REPORT
)


print(
    OUTPUT_CONFUSION
)


print(
    OUTPUT_DISTRIBUTION
)


print(
    OUTPUT_FEATURE_SUMMARY
)


print(
    OUTPUT_ALIGNMENT
)


print(
    OUTPUT_SUMMARY
)


print(
    "\nFINAL STATUS: PASS"
)