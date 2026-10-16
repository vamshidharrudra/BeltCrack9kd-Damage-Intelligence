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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_DIR = PROJECT_ROOT / "audit_outputs"

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
# INPUTS
# ============================================================================

FEATURE_SUMMARY_FILE = (
    AUDIT_DIR / "stage_3_7_feature_summary.csv"
)

TARGET_FILE = (
    AUDIT_DIR / "stage_3_6_persistent_damage_targets.csv"
)

TEMPORAL_FILE = (
    AUDIT_DIR / "stage_2_2_temporal_features.csv"
)

SPATIAL_FILE = (
    AUDIT_DIR / "stage_3_1_spatial_track_features.csv"
)


# ============================================================================
# OUTPUTS
# ============================================================================

PER_CLASS_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_per_class_error_analysis.csv"
)

CONFUSION_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_confusion_matrix.csv"
)

PREDICTIONS_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_test_predictions.csv"
)

SEQUENCE_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_sequence_error_analysis.csv"
)

CLASS_PAIR_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_confused_class_pairs.csv"
)

ERROR_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_largest_errors.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR /
    "stage_3_10_error_analysis_summary.json"
)


# ============================================================================
# JSON SAFE
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

    if isinstance(value, (list, tuple, set)):

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


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def resolve_file(
    exact_path,
    required_columns,
    keywords,
):

    if exact_path.exists():
        return exact_path

    candidates = []

    for path in AUDIT_DIR.rglob("*.csv"):

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

        score = 0

        filename = path.name.lower()

        for keyword in keywords:

            if keyword.lower() in filename:
                score += 10

        candidates.append(
            (
                score,
                str(path),
                path,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    return candidates[0][2]


# ============================================================================
# STANDARDIZATION
# ============================================================================

def find_column(
    df,
    candidates,
):

    for column in candidates:

        if column in df.columns:
            return column

    raise KeyError(
        "Could not find required column. "
        + str(candidates)
    )


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
# BUILD EXACT 232 FEATURES
# ============================================================================

def build_exact_feature_matrix(
    feature_summary,
    temporal,
    spatial,
    targets,
):

    feature_column = find_column(
        feature_summary,
        ["feature"],
    )

    features = (
        feature_summary[
            feature_column
        ]
        .astype(str)
        .tolist()
    )

    features = list(
        dict.fromkeys(features)
    )

    if len(features) != 232:

        raise RuntimeError(
            "Stage 3.7 authoritative feature list "
            f"contains {len(features)} features, not 232."
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

    # ------------------------------------------------------------------------
    # Key validation
    # ------------------------------------------------------------------------

    if (
        temporal[TEMPORAL_KEYS]
        .duplicated()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Temporal duplicate keys detected."
        )

    if (
        spatial[SPATIAL_KEYS]
        .duplicated()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Spatial duplicate keys detected."
        )

    if (
        targets[TARGET_KEYS]
        .duplicated()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Target duplicate keys detected."
        )

    # ------------------------------------------------------------------------
    # Explicit Stage 3.7 aliases
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

    temporal_columns = set(
        temporal.columns
    )

    spatial_columns = set(
        spatial.columns
    )

    feature_sources = {}

    for feature in features:

        if feature in explicit_mappings:

            source_type, source_column = (
                explicit_mappings[
                    feature
                ]
            )

            feature_sources[
                feature
            ] = (
                source_type,
                source_column,
            )

            continue

        in_temporal = (
            feature in temporal_columns
        )

        in_spatial = (
            feature in spatial_columns
        )

        if in_temporal and not in_spatial:

            feature_sources[
                feature
            ] = (
                "temporal",
                feature,
            )

        elif in_spatial and not in_temporal:

            feature_sources[
                feature
            ] = (
                "spatial",
                feature,
            )

        elif (
            in_temporal
            and in_spatial
        ):

            if feature.startswith(
                "spatial_"
            ):

                feature_sources[
                    feature
                ] = (
                    "spatial",
                    feature,
                )

            else:

                feature_sources[
                    feature
                ] = (
                    "temporal",
                    feature,
                )

        else:

            raise RuntimeError(
                "Could not map feature: "
                + feature
            )

    # ------------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------------

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

    exact = base.copy()

    # ------------------------------------------------------------------------
    # Add one feature at a time
    # ------------------------------------------------------------------------

    for index, feature in enumerate(
        features,
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
                        "__value__",
                }
            )

        else:

            source = spatial[
                SPATIAL_KEYS
                + [source_column]
            ].copy()

            source = source.rename(
                columns={
                    source_column:
                        "__value__",
                }
            )

        source = source[
            MERGE_KEYS
            + ["__value__"]
        ]

        source = source.drop_duplicates(
            subset=MERGE_KEYS
        )

        exact = exact.merge(
            source,
            on=MERGE_KEYS,
            how="left",
            validate="one_to_one",
        )

        exact[feature] = exact[
            "__value__"
        ]

        exact = exact.drop(
            columns=[
                "__value__"
            ]
        )

    actual_features = [
        column
        for column in exact.columns
        if column not in (
            MERGE_KEYS
            + [
                "split",
                TARGET_COLUMN,
            ]
        )
    ]

    if set(actual_features) != set(
        features
    ):

        missing = sorted(
            set(features)
            - set(actual_features)
        )

        unexpected = sorted(
            set(actual_features)
            - set(features)
        )

        raise RuntimeError(
            "Exact 232-feature reconstruction failed. "
            f"Missing={missing}, "
            f"Unexpected={unexpected}"
        )

    return exact, features


# ============================================================================
# MATRIX
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
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "STAGE 3.10 - CLASSIFICATION ERROR ANALYSIS"
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
    # INPUTS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CHECKING INPUT FILES"
    )
    print("=" * 78)

    temporal_file = resolve_file(
        TEMPORAL_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_2_2",
            "temporal",
        ],
    )

    spatial_file = resolve_file(
        SPATIAL_FILE,
        [
            "sequence",
            "component",
        ],
        [
            "stage_3_1",
            "spatial",
        ],
    )

    inputs = {
        "feature_summary":
            FEATURE_SUMMARY_FILE,

        "temporal":
            temporal_file,

        "spatial":
            spatial_file,

        "targets":
            TARGET_FILE,
    }

    for name, path in inputs.items():

        if path is None:

            raise FileNotFoundError(
                "Could not locate input: "
                + name
            )

        print(
            f"{name:<20}: FOUND"
        )

        print(
            "  ",
            path,
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

    print(
        "Feature summary:",
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
    # BUILD EXACT MATRIX
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "RECONSTRUCTING EXACT STAGE 3.7 MATRIX"
    )
    print("=" * 78)

    data, features = (
        build_exact_feature_matrix(
            feature_summary,
            temporal,
            spatial,
            targets,
        )
    )

    print(
        "Exact features:",
        len(features),
    )

    print(
        "Records:",
        len(data),
    )

    if len(features) != 232:

        raise RuntimeError(
            "Exact feature count is not 232."
        )

    if len(data) != 14029:

        raise RuntimeError(
            "Unexpected classification record count."
        )

    # ========================================================================
    # SPLITS
    # ========================================================================

    train_df = data[
        data["split"]
        .astype(str)
        .str.lower()
        == "train"
    ].copy()

    val_df = data[
        data["split"]
        .astype(str)
        .str.lower()
        == "val"
    ].copy()

    test_df = data[
        data["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    print()
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
    # TARGET
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
    # TRAIN FULL 232 RF
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TRAINING FULL EXACT 232-FEATURE RANDOM FOREST"
    )
    print("=" * 78)

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

    test_probability = model.predict_proba(
        x_test
    )

    classes = list(
        model.classes_
    )

    # ========================================================================
    # OVERALL PERFORMANCE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "OVERALL TEST PERFORMANCE"
    )
    print("=" * 78)

    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_prediction,
        )
    )

    test_macro_f1 = f1_score(
        y_test,
        test_prediction,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_prediction,
        average="weighted",
        zero_division=0,
    )

    print(
        "Accuracy:",
        f"{test_accuracy:.6f}",
    )

    print(
        "Balanced Accuracy:",
        f"{test_balanced_accuracy:.6f}",
    )

    print(
        "Macro F1:",
        f"{test_macro_f1:.6f}",
    )

    print(
        "Weighted F1:",
        f"{test_weighted_f1:.6f}",
    )

    # ========================================================================
    # PER CLASS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "PER-CLASS ERROR ANALYSIS"
    )
    print("=" * 78)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_test,
            test_prediction,
            labels=classes,
            zero_division=0,
        )
    )

    per_class_records = []

    for index, class_name in enumerate(
        classes
    ):

        true_count = int(
            support[index]
        )

        predicted_count = int(
            np.sum(
                test_prediction
                == class_name
            )
        )

        correct_count = int(
            np.sum(
                (
                    y_test.to_numpy()
                    == class_name
                )
                &
                (
                    test_prediction
                    == class_name
                )
            )
        )

        false_negative = (
            true_count
            - correct_count
        )

        false_positive = (
            predicted_count
            - correct_count
        )

        per_class_records.append(
            {
                "class":
                    class_name,

                "support":
                    true_count,

                "predicted_count":
                    predicted_count,

                "correct":
                    correct_count,

                "false_negative":
                    false_negative,

                "false_positive":
                    false_positive,

                "precision":
                    float(
                        precision[index]
                    ),

                "recall":
                    float(
                        recall[index]
                    ),

                "f1":
                    float(
                        f1[index]
                    ),
            }
        )

        print()
        print(
            class_name
        )

        print(
            "  Support:",
            true_count,
        )

        print(
            "  Precision:",
            f"{precision[index]:.6f}",
        )

        print(
            "  Recall:",
            f"{recall[index]:.6f}",
        )

        print(
            "  F1:",
            f"{f1[index]:.6f}",
        )

        print(
            "  False positives:",
            false_positive,
        )

        print(
            "  False negatives:",
            false_negative,
        )

    per_class_df = pd.DataFrame(
        per_class_records
    )

    per_class_df.to_csv(
        PER_CLASS_OUTPUT,
        index=False,
    )

    # ========================================================================
    # CONFUSION MATRIX
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CONFUSION MATRIX"
    )
    print("=" * 78)

    cm = confusion_matrix(
        y_test,
        test_prediction,
        labels=classes,
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "TRUE_" + str(c)
            for c in classes
        ],
        columns=[
            "PRED_" + str(c)
            for c in classes
        ],
    )

    print(
        confusion_df.to_string()
    )

    confusion_df.to_csv(
        CONFUSION_OUTPUT
    )

    # ========================================================================
    # CLASS PAIRS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "MOST CONFUSED CLASS PAIRS"
    )
    print("=" * 78)

    pair_records = []

    for i, true_class in enumerate(
        classes
    ):

        for j, predicted_class in enumerate(
            classes
        ):

            if i == j:
                continue

            count = int(
                cm[i, j]
            )

            if count == 0:
                continue

            pair_records.append(
                {
                    "true_class":
                        true_class,

                    "predicted_class":
                        predicted_class,

                    "count":
                        count,
                }
            )

    pair_df = pd.DataFrame(
        pair_records
    )

    if len(pair_df) > 0:

        pair_df = (
            pair_df
            .sort_values(
                "count",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print(
            pair_df.head(
                15
            ).to_string(
                index=False
            )
        )

    pair_df.to_csv(
        CLASS_PAIR_OUTPUT,
        index=False,
    )

    # ========================================================================
    # TEST PREDICTIONS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "BUILDING TEST PREDICTION TABLE"
    )
    print("=" * 78)

    prediction_data = test_df[
        [
            "sequence",
            "component",
            "current_frame",
            "split",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_data[
        "predicted_class"
    ] = test_prediction

    prediction_data[
        "correct"
    ] = (
        prediction_data[
            TARGET_COLUMN
        ]
        ==
        prediction_data[
            "predicted_class"
        ]
    )

    max_probability = (
        np.max(
            test_probability,
            axis=1,
        )
    )

    prediction_data[
        "prediction_confidence"
    ] = max_probability

    prediction_data[
        "error"
    ] = (
        ~prediction_data[
            "correct"
        ]
    )

    # Actual probability for true class.
    true_class_probability = []

    for row_index, true_class in enumerate(
        y_test
    ):

        class_index = classes.index(
            true_class
        )

        true_class_probability.append(
            float(
                test_probability[
                    row_index,
                    class_index,
                ]
            )
        )

    prediction_data[
        "true_class_probability"
    ] = true_class_probability

    prediction_data[
        "confidence_margin"
    ] = (
        prediction_data[
            "prediction_confidence"
        ]
        -
        prediction_data[
            "true_class_probability"
        ]
    )

    prediction_data.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    # ========================================================================
    # CONFIDENCE ANALYSIS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "CONFIDENCE / ERROR ANALYSIS"
    )
    print("=" * 78)

    correct_rows = (
        prediction_data[
            "correct"
        ]
    )

    wrong_rows = (
        prediction_data[
            "correct"
        ]
        == False
    )

    mean_correct_confidence = float(
        prediction_data.loc[
            correct_rows,
            "prediction_confidence",
        ].mean()
    )

    mean_wrong_confidence = float(
        prediction_data.loc[
            wrong_rows,
            "prediction_confidence",
        ].mean()
    )

    high_conf_wrong = int(
        (
            wrong_rows
            &
            (
                prediction_data[
                    "prediction_confidence"
                ]
                >= 0.80
            )
        ).sum()
    )

    very_high_conf_wrong = int(
        (
            wrong_rows
            &
            (
                prediction_data[
                    "prediction_confidence"
                ]
                >= 0.90
            )
        ).sum()
    )

    print(
        "Mean confidence - correct:",
        f"{mean_correct_confidence:.6f}",
    )

    print(
        "Mean confidence - wrong:",
        f"{mean_wrong_confidence:.6f}",
    )

    print(
        "Wrong predictions with confidence >= 0.80:",
        high_conf_wrong,
    )

    print(
        "Wrong predictions with confidence >= 0.90:",
        very_high_conf_wrong,
    )

    # ========================================================================
    # SEQUENCE ANALYSIS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "SEQUENCE-LEVEL ERROR ANALYSIS"
    )
    print("=" * 78)

    sequence_records = []

    for (
        sequence,
        group,
    ) in prediction_data.groupby(
        "sequence"
    ):

        n = len(group)

        errors = int(
            (
                group[
                    "correct"
                ]
                == False
            ).sum()
        )

        accuracy = (
            1.0
            - errors / n
        )

        macro = f1_score(
            group[
                TARGET_COLUMN
            ],
            group[
                "predicted_class"
            ],
            average="macro",
            zero_division=0,
        )

        balanced = (
            balanced_accuracy_score(
                group[
                    TARGET_COLUMN
                ],
                group[
                    "predicted_class"
                ],
            )
        )

        mean_confidence = float(
            group[
                "prediction_confidence"
            ].mean()
        )

        wrong_confidence = float(
            group.loc[
                group["correct"]
                == False,
                "prediction_confidence",
            ].mean()
        )

        sequence_records.append(
            {
                "sequence":
                    sequence,

                "records":
                    n,

                "errors":
                    errors,

                "accuracy":
                    accuracy,

                "balanced_accuracy":
                    balanced,

                "macro_f1":
                    macro,

                "mean_confidence":
                    mean_confidence,

                "mean_wrong_confidence":
                    wrong_confidence,
            }
        )

    sequence_df = pd.DataFrame(
        sequence_records
    )

    sequence_df = (
        sequence_df
        .sort_values(
            "macro_f1",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        sequence_df.to_string(
            index=False
        )
    )

    sequence_df.to_csv(
        SEQUENCE_OUTPUT,
        index=False,
    )

    # ========================================================================
    # LARGEST / MOST CONFIDENT ERRORS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "MOST CONFIDENT WRONG PREDICTIONS"
    )
    print("=" * 78)

    wrong_prediction_df = (
        prediction_data[
            prediction_data[
                "correct"
            ]
            == False
        ]
        .copy()
    )

    wrong_prediction_df = (
        wrong_prediction_df
        .sort_values(
            [
                "prediction_confidence",
                "confidence_margin",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        wrong_prediction_df.head(
            30
        ).to_string(
            index=False
        )
    )

    wrong_prediction_df.head(
        100
    ).to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    # ========================================================================
    # ERROR BY TRUE CLASS
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "ERROR RATE BY TRUE CLASS"
    )
    print("=" * 78)

    class_error_records = []

    for class_name in classes:

        class_rows = (
            prediction_data[
                prediction_data[
                    TARGET_COLUMN
                ]
                == class_name
            ]
        )

        n = len(
            class_rows
        )

        errors = int(
            (
                class_rows[
                    "correct"
                ]
                == False
            ).sum()
        )

        error_rate = (
            errors / n
            if n > 0
            else 0.0
        )

        class_error_records.append(
            {
                "class":
                    class_name,

                "support":
                    n,

                "errors":
                    errors,

                "error_rate":
                    error_rate,
            }
        )

    class_error_df = pd.DataFrame(
        class_error_records
    )

    print(
        class_error_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # TARGET IMBALANCE
    # ========================================================================

    print()
    print("=" * 78)
    print(
        "TEST CLASS DISTRIBUTION"
    )
    print("=" * 78)

    test_distribution = (
        y_test.value_counts()
        .reindex(classes)
        .fillna(0)
    )

    for class_name in classes:

        count = int(
            test_distribution[
                class_name
            ]
        )

        fraction = (
            count
            / len(y_test)
        )

        print(
            f"{class_name:<32}"
            f"{count:>6}"
            f"  {fraction:.4f}"
        )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    summary = {

        "stage":
            "3.10",

        "title":
            "Classification Error Analysis",

        "model":
            "RandomForestClassifier",

        "feature_count":
            len(features),

        "test_records":
            len(test_df),

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "test_macro_f1":
            test_macro_f1,

        "test_weighted_f1":
            test_weighted_f1,

        "classes":
            classes,

        "class_count":
            len(classes),

        "per_class":
            per_class_records,

        "most_confused_pairs":
            pair_records[:20]
            if len(pair_records) > 20
            else pair_records,

        "mean_correct_confidence":
            mean_correct_confidence,

        "mean_wrong_confidence":
            mean_wrong_confidence,

        "high_confidence_wrong_count":
            high_conf_wrong,

        "very_high_confidence_wrong_count":
            very_high_conf_wrong,

        "sequence_count":
            len(sequence_df),

        "worst_sequences":
            sequence_df.head(
                10
            ).to_dict(
                orient="records"
            ),

        "best_sequences":
            sequence_df.tail(
                10
            )
            .sort_values(
                "macro_f1",
                ascending=False,
            )
            .to_dict(
                orient="records"
            ),

        "test_distribution":
            test_distribution.to_dict(),

        "error_analysis_outputs": {

            "per_class":
                str(
                    PER_CLASS_OUTPUT
                ),

            "confusion_matrix":
                str(
                    CONFUSION_OUTPUT
                ),

            "predictions":
                str(
                    PREDICTIONS_OUTPUT
                ),

            "sequence_analysis":
                str(
                    SEQUENCE_OUTPUT
                ),

            "confused_class_pairs":
                str(
                    CLASS_PAIR_OUTPUT
                ),

            "largest_errors":
                str(
                    ERROR_OUTPUT
                ),
        },
    }

    summary = to_json_safe(
        summary
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
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
        "STAGE 3.10 COMPLETE"
    )
    print("=" * 78)

    print(
        "Full 232-feature model analyzed."
    )

    print(
        "Test Macro F1:",
        f"{test_macro_f1:.6f}",
    )

    print(
        "Test Balanced Accuracy:",
        f"{test_balanced_accuracy:.6f}",
    )

    print(
        "Test Accuracy:",
        f"{test_accuracy:.6f}",
    )

    print()
    print(
        "Outputs saved to:"
    )

    print(
        AUDIT_DIR
    )

    print()
    print(
        "FINAL STATUS: PASS"
    )


if __name__ == "__main__":
    main()