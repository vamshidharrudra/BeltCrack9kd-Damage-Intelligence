import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 3.28 - DAMAGE INTELLIGENCE VISUALIZATION
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_28_visualizations"
)

SEQUENCE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_28_sequence_damage_progression.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_28_visualization_summary.json"
)

PLOT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_28_damage_progression.png"
)

STATUS_PLOT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_28_damage_status.png"
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# START
# ============================================================

print_header(
    "STAGE 3.28 - DAMAGE INTELLIGENCE VISUALIZATION"
)

print("\nProject:")
print(PROJECT_ROOT)

print("\nPurpose:")
print(" - Visualize final model predictions")
print(" - Show physical damage progression")
print(" - Show predicted future damage")
print(" - Show prediction error")
print(" - Show damage status")
print(" - Show damage severity")
print(" - Do NOT retrain the model")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

print_header(
    "CHECKING REQUIRED FILES"
)

if not os.path.exists(PREDICTION_PATH):

    raise FileNotFoundError(
        PREDICTION_PATH
    )

if not os.path.exists(MANIFEST_PATH):

    raise FileNotFoundError(
        MANIFEST_PATH
    )

print(
    "Prediction file: FOUND"
)

print(
    "Manifest file: FOUND"
)


# ============================================================
# LOAD DATA
# ============================================================

print_header(
    "LOADING DATA"
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
# REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "sequence_key",
    "frame_key",
    "assigned_split",
    "future_crack_area_ratio_t1",
    "predicted_future_crack_area_ratio",
    "prediction_error",
    "absolute_error",
    "current_damage",
    "predicted_damage_change",
    "predicted_change_percent",
    "predicted_damage_status",
    "predicted_damage_level"
]

for column in required_prediction_columns:

    if column not in predictions.columns:

        raise ValueError(
            f"Missing prediction column: {column}"
        )


# ============================================================
# TEST DATA ONLY
# ============================================================

print_header(
    "SELECTING HELD-OUT TEST SEQUENCE"
)

test_data = predictions[
    predictions["assigned_split"] == "TEST"
].copy()

print(
    "TEST records:",
    len(test_data)
)

print(
    "\nTEST sequences:"
)

for sequence in sorted(
    test_data["sequence_key"].unique()
):

    count = (
        test_data[
            "sequence_key"
        ] == sequence
    ).sum()

    print(
        " -",
        sequence,
        ":",
        count,
        "records"
    )


# ============================================================
# SELECT DEMONSTRATION SEQUENCE
# ============================================================

# We use the first TEST sequence so the result
# is deterministic and reproducible.

test_sequences = sorted(
    test_data[
        "sequence_key"
    ].unique()
)

if len(test_sequences) == 0:

    raise ValueError(
        "No TEST sequences found."
    )


SELECTED_SEQUENCE = test_sequences[0]

print(
    "\nSelected sequence:",
    SELECTED_SEQUENCE
)


sequence_data = test_data[
    test_data["sequence_key"]
    == SELECTED_SEQUENCE
].copy()


sequence_data = sequence_data.sort_values(
    "frame_key"
).reset_index(
    drop=True
)


print(
    "Sequence records:",
    len(sequence_data)
)


# ============================================================
# SEQUENCE STATISTICS
# ============================================================

print_header(
    "SEQUENCE DAMAGE STATISTICS"
)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Frames:",
    len(sequence_data)
)

print(
    "First frame:",
    int(
        sequence_data[
            "frame_key"
        ].min()
    )
)

print(
    "Last frame:",
    int(
        sequence_data[
            "frame_key"
        ].max()
    )
)

print(
    "Mean current damage:",
    f"{sequence_data['current_damage'].mean():.8f}"
)

print(
    "Mean predicted damage:",
    f"{sequence_data['predicted_future_crack_area_ratio'].mean():.8f}"
)

print(
    "Mean absolute error:",
    f"{sequence_data['absolute_error'].mean():.8f}"
)

print(
    "Maximum absolute error:",
    f"{sequence_data['absolute_error'].max():.8f}"
)


# ============================================================
# STATUS COUNTS
# ============================================================

print_header(
    "DAMAGE STATUS DISTRIBUTION"
)

status_counts = (
    sequence_data[
        "predicted_damage_status"
    ]
    .value_counts()
)

print(
    status_counts
)


# ============================================================
# DAMAGE LEVEL COUNTS
# ============================================================

print_header(
    "DAMAGE LEVEL DISTRIBUTION"
)

level_counts = (
    sequence_data[
        "predicted_damage_level"
    ]
    .value_counts()
)

print(
    level_counts
)


# ============================================================
# SAVE SEQUENCE DATA
# ============================================================

sequence_data.to_csv(
    SEQUENCE_OUTPUT,
    index=False
)

print(
    "\nSequence data saved:"
)

print(
    SEQUENCE_OUTPUT
)


# ============================================================
# PLOT 1
# ACTUAL VS PREDICTED DAMAGE
# ============================================================

print_header(
    "CREATING DAMAGE PROGRESSION PLOT"
)

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    sequence_data["frame_key"],
    sequence_data["current_damage"],
    label="Current damage",
    linewidth=2
)

plt.plot(
    sequence_data["frame_key"],
    sequence_data[
        "predicted_future_crack_area_ratio"
    ],
    label="Predicted next-frame damage",
    linewidth=2
)

plt.plot(
    sequence_data["frame_key"],
    sequence_data[
        "future_crack_area_ratio_t1"
    ],
    label="Actual next-frame damage",
    linewidth=1
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Crack area ratio"
)

plt.title(
    "Belt Damage Progression - "
    + SELECTED_SEQUENCE
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    PLOT_OUTPUT
)


# ============================================================
# PLOT 2
# DAMAGE STATUS
# ============================================================

print_header(
    "CREATING DAMAGE STATUS PLOT"
)

status_numeric = {
    "DECREASING": -1,
    "STABLE": 0,
    "INCREASING": 1
}

sequence_data[
    "status_numeric"
] = sequence_data[
    "predicted_damage_status"
].map(
    status_numeric
)


plt.figure(
    figsize=(14, 4)
)

plt.step(
    sequence_data["frame_key"],
    sequence_data["status_numeric"],
    where="mid"
)

plt.yticks(
    [-1, 0, 1],
    [
        "Decreasing",
        "Stable",
        "Increasing"
    ]
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Predicted status"
)

plt.title(
    "Predicted Damage Direction - "
    + SELECTED_SEQUENCE
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    STATUS_PLOT_OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    STATUS_PLOT_OUTPUT
)


# ============================================================
# TOP ERROR FRAMES
# ============================================================

print_header(
    "LARGEST PREDICTION ERRORS"
)

top_errors = sequence_data.sort_values(
    "absolute_error",
    ascending=False
).head(10)


for _, row in top_errors.iterrows():

    print(
        "\nFrame:",
        int(row["frame_key"])
    )

    print(
        "Current:",
        f"{row['current_damage']:.6f}"
    )

    print(
        "Predicted:",
        f"{row['predicted_future_crack_area_ratio']:.6f}"
    )

    print(
        "Actual:",
        f"{row['future_crack_area_ratio_t1']:.6f}"
    )

    print(
        "Absolute error:",
        f"{row['absolute_error']:.6f}"
    )


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

summary = {

    "stage": "3.28",

    "status": "PASS",

    "selected_sequence": SELECTED_SEQUENCE,

    "split": "TEST",

    "records": int(
        len(sequence_data)
    ),

    "first_frame": int(
        sequence_data[
            "frame_key"
        ].min()
    ),

    "last_frame": int(
        sequence_data[
            "frame_key"
        ].max()
    ),

    "mean_current_damage": float(
        sequence_data[
            "current_damage"
        ].mean()
    ),

    "mean_predicted_damage": float(
        sequence_data[
            "predicted_future_crack_area_ratio"
        ].mean()
    ),

    "mean_absolute_error": float(
        sequence_data[
            "absolute_error"
        ].mean()
    ),

    "maximum_absolute_error": float(
        sequence_data[
            "absolute_error"
        ].max()
    ),

    "status_distribution": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "damage_level_distribution": {
        str(k): int(v)
        for k, v in level_counts.items()
    },

    "visualizations": [
        PLOT_OUTPUT,
        STATUS_PLOT_OUTPUT
    ],

    "prediction_data": SEQUENCE_OUTPUT
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header(
    "STAGE 3.28 FINAL SUMMARY"
)

print(
    "Selected sequence:",
    SELECTED_SEQUENCE
)

print(
    "Records:",
    len(sequence_data)
)

print(
    "Mean absolute error:",
    f"{sequence_data['absolute_error'].mean():.8f}"
)

print(
    "\nDamage progression plot:"
)

print(
    PLOT_OUTPUT
)

print(
    "\nDamage status plot:"
)

print(
    STATUS_PLOT_OUTPUT
)

print(
    "\nSequence CSV:"
)

print(
    SEQUENCE_OUTPUT
)

print(
    "\nSummary:"
)

print(
    SUMMARY_OUTPUT
)

print(
    "\n" + "=" * 78
)

print(
    "STAGE 3.28 STATUS: PASS"
)

print(
    "=" * 78
)