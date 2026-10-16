import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


# ============================================================
# STAGE 3.29 - ACTUAL BELT + CRACK SPATIAL VISUALIZATION
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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_3_29_spatial_visualization"
)

FRAME_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_29_spatial_visualization_summary.json"
)

FRAME_CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_29_visualized_frames.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Same held-out sequence used in Stage 3.28
SELECTED_SEQUENCE = "test/data24"

# Select a small number of representative frames.
# These are deterministic and spread across the sequence.
NUMBER_OF_FRAMES = 6


# ============================================================
# HELPERS
# ============================================================

def print_header(title):

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


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


def resolve_image_path(
    manifest_row
):

    source_image = str(
        manifest_row["source_image"]
    )

    if os.path.exists(source_image):
        return source_image

    canonical_image_path = str(
        manifest_row["canonical_image_path"]
    )

    if os.path.exists(
        canonical_image_path
    ):
        return canonical_image_path

    original_image_filename = str(
        manifest_row[
            "original_image_filename"
        ]
    )

    candidates = []

    # Search inside dataset root.
    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if original_image_filename in files:

            candidates.append(
                os.path.join(
                    root,
                    original_image_filename
                )
            )

            if len(candidates) >= 5:
                break

    if candidates:

        return candidates[0]

    return None


def resolve_json_path(
    manifest_row
):

    source_json = str(
        manifest_row["source_json"]
    )

    if os.path.exists(source_json):
        return source_json

    canonical_json_path = str(
        manifest_row["canonical_json_filename"]
    )

    if os.path.isabs(
        canonical_json_path
    ) and os.path.exists(
        canonical_json_path
    ):
        return canonical_json_path

    original_json_filename = str(
        manifest_row[
            "original_json_filename"
        ]
    )

    for root, _, files in os.walk(
        DATASET_ROOT
    ):

        if original_json_filename in files:

            return os.path.join(
                root,
                original_json_filename
            )

    return None


def polygon_area(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.dot(
            x,
            np.roll(y, -1)
        )
        -
        np.dot(
            y,
            np.roll(x, -1)
        )
    )


def polygon_bbox(points):

    points = np.asarray(
        points,
        dtype=float
    )

    x_min = np.min(
        points[:, 0]
    )

    x_max = np.max(
        points[:, 0]
    )

    y_min = np.min(
        points[:, 1]
    )

    y_max = np.max(
        points[:, 1]
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


def load_crack_polygons(
    json_path
):

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

        points = np.asarray(
            points,
            dtype=float
        )

        area = polygon_area(
            points
        )

        bbox = polygon_bbox(
            points
        )

        polygons.append(
            {
                "points": points,
                "area": area,
                "bbox": bbox
            }
        )

    return polygons


def damage_level(value):

    if value < 0.002:
        return "VERY_LOW"

    elif value < 0.005:
        return "LOW"

    elif value < 0.008:
        return "MEDIUM"

    elif value < 0.012:
        return "HIGH"

    elif value < 0.016:
        return "VERY_HIGH"

    else:
        return "EXTREME"


# ============================================================
# START
# ============================================================

print_header(
    "STAGE 3.29 - ACTUAL BELT + CRACK SPATIAL VISUALIZATION"
)

print("\nProject:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATASET_ROOT)

print("\nSelected sequence:")
print(SELECTED_SEQUENCE)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FRAME_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK INPUTS
# ============================================================

print_header(
    "CHECKING REQUIRED FILES"
)

for path in [
    PREDICTION_PATH,
    MANIFEST_PATH
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    print(
        os.path.basename(path),
        "FOUND"
    )


# ============================================================
# LOAD DATA
# ============================================================

print_header(
    "LOADING PREDICTIONS AND MANIFEST"
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
# SELECT TEST SEQUENCE
# ============================================================

sequence_data = predictions[
    (
        predictions[
            "assigned_split"
        ] == "TEST"
    )
    &
    (
        predictions[
            "sequence_key"
        ] == SELECTED_SEQUENCE
    )
].copy()


if len(sequence_data) == 0:

    raise ValueError(
        "Selected TEST sequence was not found."
    )


sequence_data = sequence_data.sort_values(
    "frame_key"
).reset_index(
    drop=True
)


print(
    "\nSequence records:",
    len(sequence_data)
)


# ============================================================
# SELECT REPRESENTATIVE FRAMES
# ============================================================

print_header(
    "SELECTING REPRESENTATIVE FRAMES"
)

n = len(sequence_data)

if n < NUMBER_OF_FRAMES:

    selected_indices = list(
        range(n)
    )

else:

    selected_indices = np.linspace(
        0,
        n - 1,
        NUMBER_OF_FRAMES,
        dtype=int
    )

selected_frames = (
    sequence_data.iloc[
        selected_indices
    ].copy()
)


print(
    "Selected frame numbers:"
)

for frame in selected_frames[
    "frame_key"
]:

    print(
        " -",
        int(frame)
    )


# ============================================================
# PREPARE MANIFEST
# ============================================================

print_header(
    "BUILDING MANIFEST LOOKUP"
)

manifest["frame_key"] = manifest[
    "canonical_image_filename"
].apply(
    extract_frame
)

manifest["frame_key"] = pd.to_numeric(
    manifest["frame_key"],
    errors="coerce"
)

manifest["sequence_key"] = manifest[
    "sequence"
].astype(str)


manifest_lookup = manifest[
    (
        manifest[
            "sequence_key"
        ]
        ==
        SELECTED_SEQUENCE
    )
].copy()


print(
    "Manifest records for sequence:",
    len(manifest_lookup)
)


# ============================================================
# VISUALIZE EACH FRAME
# ============================================================

print_header(
    "GENERATING CRACK VISUALIZATIONS"
)

visualized_records = []


for _, prediction_row in selected_frames.iterrows():

    frame_number = int(
        prediction_row[
            "frame_key"
        ]
    )

    print(
        "\nProcessing frame:",
        frame_number
    )


    # --------------------------------------------------------
    # FIND MANIFEST RECORD
    # --------------------------------------------------------

    matches = manifest_lookup[
        manifest_lookup[
            "frame_key"
        ]
        ==
        frame_number
    ]


    if len(matches) != 1:

        print(
            "WARNING: manifest match count =",
            len(matches)
        )

        continue


    manifest_row = matches.iloc[0]


    # --------------------------------------------------------
    # FIND IMAGE
    # --------------------------------------------------------

    image_path = resolve_image_path(
        manifest_row
    )


    if image_path is None:

        print(
            "WARNING: image not found."
        )

        continue


    print(
        "Image:",
        image_path
    )


    # --------------------------------------------------------
    # FIND JSON
    # --------------------------------------------------------

    json_path = resolve_json_path(
        manifest_row
    )


    if json_path is None:

        print(
            "WARNING: annotation JSON not found."
        )

        continue


    print(
        "JSON:",
        json_path
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = plt.imread(
            image_path
        )

    except Exception as e:

        print(
            "WARNING: could not read image:",
            e
        )

        continue


    # --------------------------------------------------------
    # LOAD CRACK POLYGONS
    # --------------------------------------------------------

    polygons = load_crack_polygons(
        json_path
    )


    crack_count = len(
        polygons
    )


    image_height = image.shape[0]
    image_width = image.shape[1]

    image_area = (
        image_width
        *
        image_height
    )


    total_crack_area = sum(
        polygon["area"]
        for polygon in polygons
    )


    crack_area_ratio = (
        total_crack_area
        /
        image_area
        if image_area > 0
        else 0.0
    )


    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )


    ax.imshow(
        image
    )


    # --------------------------------------------------------
    # DRAW CRACK POLYGONS
    # --------------------------------------------------------

    for index, polygon_data in enumerate(
        polygons,
        start=1
    ):

        points = polygon_data[
            "points"
        ]

        bbox = polygon_data[
            "bbox"
        ]

        polygon_patch = Polygon(
            points,
            closed=True,
            fill=False,
            linewidth=2
        )

        ax.add_patch(
            polygon_patch
        )


        x_min, y_min, x_max, y_max = (
            bbox
        )

        width = (
            x_max
            -
            x_min
        )

        height = (
            y_max
            -
            y_min
        )


        rectangle = Rectangle(
            (
                x_min,
                y_min
            ),
            width,
            height,
            fill=False,
            linewidth=1
        )

        ax.add_patch(
            rectangle
        )


        ax.text(
            x_min,
            max(
                0,
                y_min - 5
            ),
            f"Crack {index}",
            fontsize=9,
            bbox=dict(
                alpha=0.7
            )
        )


    # --------------------------------------------------------
    # DAMAGE INFORMATION
    # --------------------------------------------------------

    current_damage = float(
        prediction_row[
            "current_damage"
        ]
    )

    predicted_damage = float(
        prediction_row[
            "predicted_future_crack_area_ratio"
        ]
    )

    actual_damage = float(
        prediction_row[
            "future_crack_area_ratio_t1"
        ]
    )

    predicted_change = float(
        prediction_row[
            "predicted_damage_change"
        ]
    )

    predicted_change_percent = float(
        prediction_row[
            "predicted_change_percent"
        ]
    )

    status = str(
        prediction_row[
            "predicted_damage_status"
        ]
    )

    predicted_level = str(
        prediction_row[
            "predicted_damage_level"
        ]
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        "BeltCrack9kd Damage Intelligence\n"
        f"Sequence: {SELECTED_SEQUENCE} | "
        f"Frame: {frame_number}",
        fontsize=16
    )


    ax.axis(
        "off"
    )


    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    information = (
        f"CRACK COUNT: {crack_count}\n"
        f"CURRENT CRACK AREA RATIO: "
        f"{current_damage:.6f}\n"
        f"PREDICTED NEXT DAMAGE: "
        f"{predicted_damage:.6f}\n"
        f"ACTUAL NEXT DAMAGE: "
        f"{actual_damage:.6f}\n"
        f"PREDICTED CHANGE: "
        f"{predicted_change:+.6f}\n"
        f"PREDICTED CHANGE: "
        f"{predicted_change_percent:+.2f}%\n"
        f"STATUS: {status}\n"
        f"SEVERITY: {predicted_level}"
    )


    ax.text(
        0.015,
        0.015,
        information,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            alpha=0.85
        )
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_filename = (
        f"{SELECTED_SEQUENCE.replace('/', '_')}"
        f"_frame_{frame_number:04d}.png"
    )


    output_path = os.path.join(
        FRAME_OUTPUT_DIR,
        output_filename
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        output_path
    )


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    visualized_records.append(
        {
            "sequence": SELECTED_SEQUENCE,
            "frame": frame_number,
            "image_path": image_path,
            "json_path": json_path,
            "crack_count": crack_count,
            "computed_crack_area_ratio": crack_area_ratio,
            "model_current_damage": current_damage,
            "predicted_next_damage": predicted_damage,
            "actual_next_damage": actual_damage,
            "predicted_change": predicted_change,
            "predicted_change_percent": predicted_change_percent,
            "predicted_status": status,
            "predicted_severity": predicted_level,
            "absolute_prediction_error": float(
                prediction_row[
                    "absolute_error"
                ]
            ),
            "visualization_path": output_path
        }
    )


# ============================================================
# SAVE FRAME TABLE
# ============================================================

print_header(
    "SAVING VISUALIZATION RESULTS"
)

visualized_df = pd.DataFrame(
    visualized_records
)

visualized_df.to_csv(
    FRAME_CSV_OUTPUT,
    index=False
)


print(
    "Visualization records:",
    len(visualized_df)
)

print(
    "Saved:",
    FRAME_CSV_OUTPUT
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage": "3.29",

    "status": "PASS",

    "sequence": SELECTED_SEQUENCE,

    "requested_frames": NUMBER_OF_FRAMES,

    "visualized_frames": int(
        len(visualized_df)
    ),

    "frame_numbers": [
        int(x)
        for x in visualized_df[
            "frame"
        ].tolist()
    ],

    "mean_crack_count": (
        float(
            visualized_df[
                "crack_count"
            ].mean()
        )
        if len(visualized_df)
        else None
    ),

    "mean_computed_crack_area_ratio": (
        float(
            visualized_df[
                "computed_crack_area_ratio"
            ].mean()
        )
        if len(visualized_df)
        else None
    ),

    "mean_prediction_error": (
        float(
            visualized_df[
                "absolute_prediction_error"
            ].mean()
        )
        if len(visualized_df)
        else None
    ),

    "visualization_output_directory": FRAME_OUTPUT_DIR,

    "frame_csv": FRAME_CSV_OUTPUT
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
# FINAL
# ============================================================

print_header(
    "STAGE 3.29 FINAL SUMMARY"
)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Requested frames:",
    NUMBER_OF_FRAMES
)

print(
    "Successfully visualized:",
    len(visualized_df)
)

print(
    "\nVisualization directory:"
)

print(
    FRAME_OUTPUT_DIR
)

print(
    "\nFrame CSV:"
)

print(
    FRAME_CSV_OUTPUT
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
    "STAGE 3.29 STATUS: PASS"
)

print(
    "=" * 78
)