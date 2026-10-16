import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ============================================================
# STAGE 4.0
# OFFLINE BELT DAMAGE INTELLIGENCE APPLICATION
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

DATASET_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_27_final_inference_predictions.csv"
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "beltcrack9kd_final_gbr_33_features.pkl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_4_0_offline_application"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_0_application_report.json"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "selected_sequence_analysis.csv"
)

PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "selected_sequence_dashboard.png"
)

CRACK_VISUAL_PATH = os.path.join(
    OUTPUT_DIR,
    "selected_frame_crack_visualization.png"
)


# ============================================================
# USER CONFIGURATION
# ============================================================

SELECTED_SEQUENCE = "test/data24"

# Choose a representative frame.
# Set to None to automatically select the middle frame.
SELECTED_FRAME = None


# ============================================================
# HELPERS
# ============================================================

def header(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def extract_frame(value):

    value = str(value)

    stem = os.path.splitext(
        os.path.basename(value)
    )[0]

    if stem.isdigit():
        return int(stem)

    numbers = re.findall(
        r"\d+",
        stem
    )

    if not numbers:
        return np.nan

    return int(numbers[-1])


def resolve_image_path(row):

    candidates = []

    for column in [
        "source_image",
        "canonical_image_path"
    ]:

        if column in row.index:

            value = str(
                row[column]
            )

            if value not in [
                "",
                "nan",
                "None"
            ]:

                candidates.append(
                    value
                )


    for candidate in candidates:

        if os.path.exists(candidate):

            return candidate


    filename = str(
        row[
            "original_image_filename"
        ]
    )


    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )


    return None


def resolve_json_path(row):

    source_json = str(
        row[
            "source_json"
        ]
    )

    if os.path.exists(
        source_json
    ):

        return source_json


    filename = str(
        row[
            "original_json_filename"
        ]
    )


    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )


    return None


def load_crack_polygons(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    polygons = []

    for shape in data.get(
        "shapes",
        []
    ):

        label = str(
            shape.get(
                "label",
                ""
            )
        ).lower().strip()


        if label != "crack":
            continue


        points = shape.get(
            "points",
            []
        )


        if len(points) < 3:
            continue


        polygons.append(
            np.asarray(
                points,
                dtype=float
            )
        )


    return polygons


def damage_level(value):

    if value < 0.002:
        return "VERY LOW"

    if value < 0.005:
        return "LOW"

    if value < 0.008:
        return "MEDIUM"

    if value < 0.012:
        return "HIGH"

    if value < 0.016:
        return "VERY HIGH"

    return "EXTREME"


# ============================================================
# START
# ============================================================

header(
    "STAGE 4.0 - OFFLINE BELT DAMAGE INTELLIGENCE"
)

print(
    "Project:",
    PROJECT_ROOT
)

print(
    "Dataset:",
    DATASET_ROOT
)

print(
    "Selected sequence:",
    SELECTED_SEQUENCE
)


# ============================================================
# CHECK FILES
# ============================================================

header(
    "CHECKING REQUIRED FILES"
)

required_files = [

    PREDICTION_PATH,

    MANIFEST_PATH,

    MODEL_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            "Missing:\n" + path
        )

    print(
        "FOUND:",
        os.path.basename(path)
    )


# ============================================================
# CREATE OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

header(
    "LOADING FINAL MODEL"
)

import joblib

artifact = joblib.load(
    MODEL_PATH
)

print(
    "Artifact type:",
    type(artifact).__name__
)


if isinstance(
    artifact,
    dict
):

    print(
        "Artifact keys:",
        list(
            artifact.keys()
        )
    )

    model = artifact[
        "model"
    ]

    feature_names = artifact[
        "feature_names"
    ]

else:

    model = artifact

    feature_names = []


print(
    "Predictor:",
    type(model).__name__
)

print(
    "Feature count:",
    len(feature_names)
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

header(
    "LOADING INFERENCE RESULTS"
)

predictions = pd.read_csv(
    PREDICTION_PATH
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

print(
    "Prediction records:",
    len(predictions)
)

print(
    "Manifest records:",
    len(manifest)
)


# ============================================================
# SELECT SEQUENCE
# ============================================================

header(
    "SELECTING SEQUENCE"
)

sequence_df = predictions[
    predictions[
        "sequence_key"
    ].astype(str)
    ==
    SELECTED_SEQUENCE
].copy()


if len(sequence_df) == 0:

    raise ValueError(
        "Sequence not found: "
        + SELECTED_SEQUENCE
    )


sequence_df = sequence_df.sort_values(
    "frame_key"
).reset_index(
    drop=True
)


print(
    "Sequence records:",
    len(sequence_df)
)

print(
    "First frame:",
    sequence_df[
        "frame_key"
    ].iloc[0]
)

print(
    "Last frame:",
    sequence_df[
        "frame_key"
    ].iloc[-1]
)


# ============================================================
# SELECT FRAME
# ============================================================

if SELECTED_FRAME is None:

    selected_index = (
        len(sequence_df)
        // 2
    )

    selected_row = (
        sequence_df.iloc[
            selected_index
        ]
    )

else:

    matches = sequence_df[
        sequence_df[
            "frame_key"
        ]
        ==
        SELECTED_FRAME
    ]

    if len(matches) == 0:

        raise ValueError(
            "Frame not found: "
            + str(
                SELECTED_FRAME
            )
        )

    selected_row = matches.iloc[0]


selected_frame = int(
    selected_row[
        "frame_key"
    ]
)


print(
    "Selected frame:",
    selected_frame
)


# ============================================================
# SEQUENCE STATISTICS
# ============================================================

header(
    "SEQUENCE DAMAGE SUMMARY"
)

current_damage = sequence_df[
    "current_damage"
].to_numpy(
    dtype=float
)

predicted_damage = sequence_df[
    "predicted_future_crack_area_ratio"
].to_numpy(
    dtype=float
)

actual_damage = sequence_df[
    "future_crack_area_ratio_t1"
].to_numpy(
    dtype=float
)

absolute_error = sequence_df[
    "absolute_error"
].to_numpy(
    dtype=float
)


sequence_mae = float(
    np.mean(
        np.abs(
            predicted_damage
            -
            actual_damage
        )
    )
)


sequence_rmse = float(
    np.sqrt(
        np.mean(
            (
                predicted_damage
                -
                actual_damage
            ) ** 2
        )
    )
)


print(
    "Sequence MAE:",
    f"{sequence_mae:.8f}"
)

print(
    "Sequence RMSE:",
    f"{sequence_rmse:.8f}"
)

print(
    "Mean current damage:",
    f"{np.mean(current_damage):.8f}"
)

print(
    "Mean predicted damage:",
    f"{np.mean(predicted_damage):.8f}"
)

print(
    "Mean actual next damage:",
    f"{np.mean(actual_damage):.8f}"
)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

print(
    "\nPredicted status distribution:"
)

print(
    sequence_df[
        "predicted_damage_status"
    ].value_counts()
)


print(
    "\nPredicted severity distribution:"
)

print(
    sequence_df[
        "predicted_damage_level"
    ].value_counts()
)


# ============================================================
# MANIFEST LOOKUP
# ============================================================

header(
    "RESOLVING SELECTED BELT FRAME"
)

manifest[
    "sequence_key"
] = manifest[
    "sequence"
].astype(str)


manifest[
    "frame_key"
] = manifest[
    "canonical_image_filename"
].apply(
    extract_frame
)


frame_matches = manifest[
    (
        manifest[
            "sequence_key"
        ]
        ==
        SELECTED_SEQUENCE
    )
    &
    (
        manifest[
            "frame_key"
        ]
        ==
        selected_frame
    )
]


if len(frame_matches) != 1:

    raise ValueError(
        "Expected exactly one manifest "
        "record, found "
        + str(
            len(frame_matches)
        )
    )


manifest_row = (
    frame_matches.iloc[0]
)


image_path = resolve_image_path(
    manifest_row
)

json_path = resolve_json_path(
    manifest_row
)


if image_path is None:

    raise FileNotFoundError(
        "Could not resolve image."
    )


if json_path is None:

    raise FileNotFoundError(
        "Could not resolve JSON."
    )


print(
    "Image:",
    image_path
)

print(
    "JSON:",
    json_path
)


# ============================================================
# SELECTED PREDICTION
# ============================================================

selected_prediction = (
    sequence_df[
        sequence_df[
            "frame_key"
        ]
        ==
        selected_frame
    ].iloc[0]
)


selected_current = float(
    selected_prediction[
        "current_damage"
    ]
)

selected_predicted = float(
    selected_prediction[
        "predicted_future_crack_area_ratio"
    ]
)

selected_actual = float(
    selected_prediction[
        "future_crack_area_ratio_t1"
    ]
)

selected_error = float(
    selected_prediction[
        "absolute_error"
    ]
)

selected_status = str(
    selected_prediction[
        "predicted_damage_status"
    ]
)

selected_severity = str(
    selected_prediction[
        "predicted_damage_level"
    ]
)


print(
    "\nSelected frame intelligence:"
)

print(
    "Current damage:",
    f"{selected_current:.6f}"
)

print(
    "Predicted next damage:",
    f"{selected_predicted:.6f}"
)

print(
    "Actual next damage:",
    f"{selected_actual:.6f}"
)

print(
    "Prediction error:",
    f"{selected_error:.6f}"
)

print(
    "Status:",
    selected_status
)

print(
    "Severity:",
    selected_severity
)


# ============================================================
# LOAD IMAGE
# ============================================================

header(
    "CREATING CRACK VISUALIZATION"
)

image = plt.imread(
    image_path
)

polygons = load_crack_polygons(
    json_path
)


print(
    "Image dimensions:",
    image.shape
)

print(
    "Crack polygons:",
    len(polygons)
)


# ============================================================
# CRACK VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)

ax.imshow(
    image
)


for polygon in polygons:

    patch = Polygon(
        polygon,
        closed=True,
        fill=False,
        linewidth=2
    )

    ax.add_patch(
        patch
    )


ax.set_title(
    "BELTCRACK9KD - DAMAGE INTELLIGENCE",
    fontsize=18,
    fontweight="bold"
)

ax.text(
    0.5,
    0.96,
    (
        f"Sequence: {SELECTED_SEQUENCE}    "
        f"Frame: {selected_frame}"
    ),
    transform=ax.transAxes,
    ha="center",
    fontsize=12
)


information = (
    f"Cracks: {len(polygons)}\n"
    f"Current damage: {selected_current:.6f}\n"
    f"Predicted next: {selected_predicted:.6f}\n"
    f"Actual next: {selected_actual:.6f}\n"
    f"Error: {selected_error:.6f}\n"
    f"Status: {selected_status}\n"
    f"Severity: {selected_severity}"
)


ax.text(
    0.02,
    0.03,
    information,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="bottom",
    bbox=dict(
        alpha=0.88
    )
)

ax.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    CRACK_VISUAL_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    CRACK_VISUAL_PATH
)


# ============================================================
# DAMAGE PROGRESSION DASHBOARD
# ============================================================

header(
    "CREATING DAMAGE PROGRESSION DASHBOARD"
)

frames = sequence_df[
    "frame_key"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(15, 7)
)

ax.plot(
    frames,
    current_damage,
    label="Current damage",
    linewidth=2
)

ax.plot(
    frames,
    predicted_damage,
    label="Predicted next damage",
    linewidth=2
)

ax.plot(
    frames,
    actual_damage,
    label="Actual next damage",
    linewidth=2
)

ax.axvline(
    selected_frame,
    linestyle="--",
    linewidth=1.5
)

ax.set_title(
    "Sequential Belt Damage Intelligence",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Frame"
)

ax.set_ylabel(
    "Crack area ratio"
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    PLOT_PATH
)


# ============================================================
# EXPORT SEQUENCE DATA
# ============================================================

sequence_df.to_csv(
    CSV_PATH,
    index=False
)

print(
    "Saved:",
    CSV_PATH
)


# ============================================================
# BUILD APPLICATION REPORT
# ============================================================

report = {

    "stage":
        "4.0",

    "status":
        "PASS",

    "application":
        "Offline Belt Damage Intelligence",

    "sequence":
        SELECTED_SEQUENCE,

    "selected_frame":
        int(selected_frame),

    "sequence_records":
        int(len(sequence_df)),

    "sequence_mae":
        sequence_mae,

    "sequence_rmse":
        sequence_rmse,

    "mean_current_damage":
        float(
            np.mean(
                current_damage
            )
        ),

    "mean_predicted_damage":
        float(
            np.mean(
                predicted_damage
            )
        ),

    "mean_actual_next_damage":
        float(
            np.mean(
                actual_damage
            )
        ),

    "selected_frame":

        {
            "frame":
                int(selected_frame),

            "current_damage":
                selected_current,

            "predicted_next_damage":
                selected_predicted,

            "actual_next_damage":
                selected_actual,

            "prediction_error":
                selected_error,

            "status":
                selected_status,

            "severity":
                selected_severity,

            "crack_count":
                int(len(polygons))
        },

    "outputs":
        {
            "sequence_csv":
                CSV_PATH,

            "dashboard":
                PLOT_PATH,

            "crack_visualization":
                CRACK_VISUAL_PATH,

            "report":
                REPORT_PATH
        }
}


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL
# ============================================================

header(
    "STAGE 4.0 FINAL SUMMARY"
)

print(
    "Application status: PASS"
)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Selected frame:",
    selected_frame
)

print(
    "Sequence MAE:",
    f"{sequence_mae:.8f}"
)

print(
    "Sequence RMSE:",
    f"{sequence_rmse:.8f}"
)

print(
    "\nOutput directory:",
    OUTPUT_DIR
)

print(
    "\nReport:",
    REPORT_PATH
)

print(
    "Sequence CSV:",
    CSV_PATH
)

print(
    "Dashboard:",
    PLOT_PATH
)

print(
    "Crack visualization:",
    CRACK_VISUAL_PATH
)

print(
    "\n" + "=" * 80
)

print(
    "STAGE 4.0 STATUS: PASS"
)

print(
    "=" * 80
)