import os
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


# ============================================================
# STAGE 3.30 - SEQUENTIAL DAMAGE INTELLIGENCE DEMO
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
    "stage_3_30_sequential_demo"
)

FRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sequential_damage_data.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_3_30_summary.json"
)

VIDEO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "belt_damage_intelligence.mp4"
)

SELECTED_SEQUENCE = "test/data24"


# ============================================================
# SETTINGS
# ============================================================

# Set this to True if you want to regenerate all frames.
# Keep True for the first run.
REBUILD_FRAMES = True

# Video FPS
VIDEO_FPS = 10

# Maximum number of frames to visualize.
# 0 = all available frames.
MAX_FRAMES = 0


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


def resolve_image_path(row):

    source_image = str(
        row["source_image"]
    )

    if os.path.exists(source_image):
        return source_image

    canonical_path = str(
        row["canonical_image_path"]
    )

    if os.path.exists(canonical_path):
        return canonical_path

    filename = str(
        row["original_image_filename"]
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
        row["source_json"]
    )

    if os.path.exists(source_json):
        return source_json

    filename = str(
        row["original_json_filename"]
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

        points = np.asarray(
            points,
            dtype=float
        )

        polygons.append(
            points
        )

    return polygons


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


def damage_level(value):

    if value < 0.002:
        return "VERY LOW"

    elif value < 0.005:
        return "LOW"

    elif value < 0.008:
        return "MEDIUM"

    elif value < 0.012:
        return "HIGH"

    elif value < 0.016:
        return "VERY HIGH"

    return "EXTREME"


# ============================================================
# START
# ============================================================

print_header(
    "STAGE 3.30 - SEQUENTIAL DAMAGE INTELLIGENCE DEMO"
)

print(
    "\nProject:",
    PROJECT_ROOT
)

print(
    "Dataset:",
    DATASET_ROOT
)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Video FPS:",
    VIDEO_FPS
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
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if REBUILD_FRAMES and os.path.exists(
    FRAME_DIR
):

    print(
        "\nRemoving previous frame directory..."
    )

    shutil.rmtree(
        FRAME_DIR
    )

os.makedirs(
    FRAME_DIR,
    exist_ok=True
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
# SELECT SEQUENCE
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
        "Selected TEST sequence not found."
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
# LIMIT FRAMES IF REQUESTED
# ============================================================

if MAX_FRAMES > 0:

    sequence_data = sequence_data.head(
        MAX_FRAMES
    ).copy()

    print(
        "Using first",
        MAX_FRAMES,
        "frames."
    )

else:

    print(
        "Using all available frames."
    )


# ============================================================
# BUILD MANIFEST LOOKUP
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

manifest["sequence_key"] = (
    manifest[
        "sequence"
    ].astype(str)
)


manifest_lookup = manifest[
    manifest[
        "sequence_key"
    ] == SELECTED_SEQUENCE
].copy()


print(
    "Manifest sequence records:",
    len(manifest_lookup)
)


# ============================================================
# PROCESS FRAMES
# ============================================================

print_header(
    "GENERATING SEQUENTIAL DAMAGE FRAMES"
)

records = []

total = len(
    sequence_data
)

failed = 0


for position, (_, row) in enumerate(
    sequence_data.iterrows(),
    start=1
):

    frame_number = int(
        row["frame_key"]
    )

    print(
        f"[{position}/{total}] Frame {frame_number}"
    )


    # --------------------------------------------------------
    # MANIFEST MATCH
    # --------------------------------------------------------

    matches = manifest_lookup[
        manifest_lookup[
            "frame_key"
        ] == frame_number
    ]


    if len(matches) != 1:

        print(
            "  WARNING: manifest match:",
            len(matches)
        )

        failed += 1

        continue


    manifest_row = matches.iloc[0]


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = resolve_image_path(
        manifest_row
    )

    if image_path is None:

        print(
            "  WARNING: image not found."
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = resolve_json_path(
        manifest_row
    )

    if json_path is None:

        print(
            "  WARNING: JSON not found."
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = plt.imread(
            image_path
        )

    except Exception as e:

        print(
            "  WARNING: image read failed:",
            e
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # LOAD CRACKS
    # --------------------------------------------------------

    try:

        polygons = load_crack_polygons(
            json_path
        )

    except Exception as e:

        print(
            "  WARNING: JSON read failed:",
            e
        )

        failed += 1

        continue


    image_height = image.shape[0]
    image_width = image.shape[1]

    image_area = (
        image_width
        *
        image_height
    )


    total_crack_area = sum(
        polygon_area(
            polygon
        )
        for polygon in polygons
    )


    actual_crack_ratio = (
        total_crack_area
        /
        image_area
        if image_area > 0
        else 0.0
    )


    # --------------------------------------------------------
    # MODEL VALUES
    # --------------------------------------------------------

    current_damage = float(
        row[
            "current_damage"
        ]
    )

    predicted_damage = float(
        row[
            "predicted_future_crack_area_ratio"
        ]
    )

    actual_next_damage = float(
        row[
            "future_crack_area_ratio_t1"
        ]
    )

    prediction_error = float(
        row[
            "absolute_error"
        ]
    )

    predicted_change = float(
        row[
            "predicted_damage_change"
        ]
    )

    predicted_change_percent = float(
        row[
            "predicted_change_percent"
        ]
    )

    status = str(
        row[
            "predicted_damage_status"
        ]
    )

    severity = str(
        row[
            "predicted_damage_level"
        ]
    )


    # --------------------------------------------------------
    # FIGURE
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

    for index, polygon in enumerate(
        polygons,
        start=1
    ):

        patch = Polygon(
            polygon,
            closed=True,
            fill=False,
            linewidth=2
        )

        ax.add_patch(
            patch
        )


        x_min = np.min(
            polygon[:, 0]
        )

        y_min = np.min(
            polygon[:, 1]
        )

        x_max = np.max(
            polygon[:, 0]
        )

        y_max = np.max(
            polygon[:, 1]
        )


        rect = Rectangle(
            (
                x_min,
                y_min
            ),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            linewidth=1
        )

        ax.add_patch(
            rect
        )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    ax.set_title(
        "BELTCRACK9KD DAMAGE INTELLIGENCE",
        fontsize=18,
        fontweight="bold"
    )


    ax.text(
        0.5,
        0.965,
        (
            f"Sequence: {SELECTED_SEQUENCE}    "
            f"Frame: {frame_number}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12
    )


    ax.axis(
        "off"
    )


    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    panel_text = (
        f"CRACKS DETECTED        : {len(polygons)}\n"
        f"CURRENT DAMAGE         : {current_damage:.6f}\n"
        f"PREDICTED NEXT DAMAGE  : {predicted_damage:.6f}\n"
        f"ACTUAL NEXT DAMAGE     : {actual_next_damage:.6f}\n"
        f"PREDICTED CHANGE       : {predicted_change:+.6f}\n"
        f"PREDICTED CHANGE       : {predicted_change_percent:+.2f}%\n"
        f"STATUS                 : {status}\n"
        f"SEVERITY               : {severity}\n"
        f"PREDICTION ERROR       : {prediction_error:.6f}"
    )


    ax.text(
        0.015,
        0.02,
        panel_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            alpha=0.88
        )
    )


    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    filename = (
        f"frame_{frame_number:04d}.png"
    )

    output_path = os.path.join(
        FRAME_DIR,
        filename
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # STORE RECORD
    # --------------------------------------------------------

    records.append(
        {
            "sequence": SELECTED_SEQUENCE,
            "frame": frame_number,
            "image_path": image_path,
            "json_path": json_path,
            "crack_count": len(polygons),
            "computed_crack_area_ratio":
                float(actual_crack_ratio),
            "current_damage":
                current_damage,
            "predicted_next_damage":
                predicted_damage,
            "actual_next_damage":
                actual_next_damage,
            "predicted_change":
                predicted_change,
            "predicted_change_percent":
                predicted_change_percent,
            "predicted_status":
                status,
            "predicted_severity":
                severity,
            "prediction_error":
                prediction_error,
            "frame_output":
                output_path
        }
    )


# ============================================================
# SAVE SEQUENTIAL DATA
# ============================================================

print_header(
    "SAVING SEQUENTIAL DAMAGE DATA"
)

result_df = pd.DataFrame(
    records
)

result_df.to_csv(
    CSV_OUTPUT,
    index=False
)


print(
    "Generated frames:",
    len(result_df)
)

print(
    "Failed frames:",
    failed
)

print(
    "CSV:",
    CSV_OUTPUT
)


# ============================================================
# VIDEO CREATION
# ============================================================

print_header(
    "CREATING MP4 VIDEO"
)

video_created = False

try:

    import cv2

    frame_files = sorted(
        [
            os.path.join(
                FRAME_DIR,
                f
            )
            for f in os.listdir(
                FRAME_DIR
            )
            if f.lower().endswith(
                ".png"
            )
        ]
    )


    if len(frame_files) == 0:

        raise RuntimeError(
            "No PNG frames available."
        )


    first_frame = cv2.imread(
        frame_files[0]
    )


    if first_frame is None:

        raise RuntimeError(
            "Could not read first frame."
        )


    height, width = (
        first_frame.shape[:2]
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        VIDEO_OUTPUT,
        fourcc,
        VIDEO_FPS,
        (
            width,
            height
        )
    )


    if not writer.isOpened():

        raise RuntimeError(
            "Could not open video writer."
        )


    for frame_file in frame_files:

        frame = cv2.imread(
            frame_file
        )

        if frame is None:
            continue

        if (
            frame.shape[1] != width
            or
            frame.shape[0] != height
        ):

            frame = cv2.resize(
                frame,
                (
                    width,
                    height
                )
            )

        writer.write(
            frame
        )


    writer.release()

    video_created = os.path.exists(
        VIDEO_OUTPUT
    )


except Exception as e:

    print(
        "WARNING: video creation failed:"
    )

    print(
        str(e)
    )

    print(
        "PNG frame generation remains valid."
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage": "3.30",

    "status": (
        "PASS"
        if len(result_df) > 0
        else "FAIL"
    ),

    "sequence": SELECTED_SEQUENCE,

    "total_requested_frames": int(
        len(sequence_data)
    ),

    "generated_frames": int(
        len(result_df)
    ),

    "failed_frames": int(
        failed
    ),

    "video_created": bool(
        video_created
    ),

    "video_path": (
        VIDEO_OUTPUT
        if video_created
        else None
    ),

    "frame_directory": FRAME_DIR,

    "csv_output": CSV_OUTPUT,

    "mean_current_damage": (
        float(
            result_df[
                "current_damage"
            ].mean()
        )
        if len(result_df)
        else None
    ),

    "mean_predicted_damage": (
        float(
            result_df[
                "predicted_next_damage"
            ].mean()
        )
        if len(result_df)
        else None
    ),

    "mean_absolute_prediction_error": (
        float(
            result_df[
                "prediction_error"
            ].mean()
        )
        if len(result_df)
        else None
    ),

    "status_distribution": (
        {
            str(k): int(v)
            for k, v in
            result_df[
                "predicted_status"
            ].value_counts().items()
        }
        if len(result_df)
        else {}
    ),

    "severity_distribution": (
        {
            str(k): int(v)
            for k, v in
            result_df[
                "predicted_severity"
            ].value_counts().items()
        }
        if len(result_df)
        else {}
    )
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
    "STAGE 3.30 FINAL SUMMARY"
)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Requested frames:",
    len(sequence_data)
)

print(
    "Generated frames:",
    len(result_df)
)

print(
    "Failed frames:",
    failed
)

print(
    "Video created:",
    video_created
)

if video_created:

    print(
        "Video:",
        VIDEO_OUTPUT
    )

print(
    "\nFrames:",
    FRAME_DIR
)

print(
    "\nSequential CSV:",
    CSV_OUTPUT
)

print(
    "\nSummary:",
    SUMMARY_OUTPUT
)

print(
    "\n" + "=" * 78
)

print(
    "STAGE 3.30 STATUS:",
    "PASS" if len(result_df) > 0 else "FAIL"
)

print(
    "=" * 78
)