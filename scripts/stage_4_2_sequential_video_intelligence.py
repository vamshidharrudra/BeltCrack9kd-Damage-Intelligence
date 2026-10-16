import os
import re
import json
import shutil
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# STAGE 4.2
# SEQUENTIAL VIDEO DAMAGE INTELLIGENCE
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
    "stage_4_2_sequential_video"
)

FRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)

VIDEO_PATH = os.path.join(
    OUTPUT_DIR,
    "belt_damage_intelligence.mp4"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "sequential_damage_intelligence.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_2_summary.json"
)


# ============================================================
# USER CONFIGURATION
# ============================================================

SELECTED_SEQUENCE = "test/data24"

FPS = 10

# Resize video frames to reduce video size.
MAX_WIDTH = 1280

# Set to None to process the complete sequence.
MAX_FRAMES = None


# ============================================================
# HELPERS
# ============================================================

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

    if "source_image" in row.index:

        value = str(
            row["source_image"]
        )

        if value not in [
            "",
            "nan",
            "None"
        ]:

            candidates.append(value)

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

    if "source_json" in row.index:

        value = str(
            row["source_json"]
        )

        if os.path.exists(value):

            return value

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


def load_polygons(json_path):

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

        if len(points) >= 3:

            polygons.append(
                np.asarray(
                    points,
                    dtype=np.float32
                )
            )

    return polygons


def load_font(size):

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf"
    ]

    for path in candidates:

        if os.path.exists(path):

            try:

                return ImageFont.truetype(
                    path,
                    size
                )

            except Exception:
                pass

    return ImageFont.load_default()


def resize_keep_ratio(
    image,
    max_width
):

    height, width = image.shape[:2]

    if width <= max_width:

        return image

    scale = (
        max_width
        /
        float(width)
    )

    new_width = max_width

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


def draw_polygon(
    draw,
    polygon,
    scale_x,
    scale_y
):

    points = []

    for x, y in polygon:

        points.append(
            (
                int(x * scale_x),
                int(y * scale_y)
            )
        )

    if len(points) >= 3:

        draw.line(
            points + [points[0]],
            fill=(255, 70, 70),
            width=3
        )


def safe_float(value):

    if pd.isna(value):

        return None

    return float(value)


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("STAGE 4.2 - SEQUENTIAL VIDEO DAMAGE INTELLIGENCE")
print("=" * 80)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "FPS:",
    FPS
)


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

for path in [
    PREDICTION_PATH,
    MANIFEST_PATH
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            "Missing file:\n" + path
        )

    print(
        "FOUND:",
        path
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if os.path.exists(FRAME_DIR):

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

print()
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

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
# PREPARE KEYS
# ============================================================

predictions[
    "sequence_key"
] = predictions[
    "sequence_key"
].astype(str)


predictions[
    "frame_key"
] = predictions[
    "frame_key"
].apply(
    lambda x: int(float(x))
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


# ============================================================
# SELECT SEQUENCE
# ============================================================

sequence_df = predictions[
    predictions[
        "sequence_key"
    ]
    ==
    SELECTED_SEQUENCE
].copy()


sequence_df = sequence_df.sort_values(
    "frame_key"
).reset_index(
    drop=True
)


if len(sequence_df) == 0:

    raise ValueError(
        "Selected sequence not found: "
        +
        SELECTED_SEQUENCE
    )


print()
print(
    "Sequence records:",
    len(sequence_df)
)

print(
    "First frame:",
    int(
        sequence_df[
            "frame_key"
        ].iloc[0]
    )
)

print(
    "Last frame:",
    int(
        sequence_df[
            "frame_key"
        ].iloc[-1]
    )
)


# ============================================================
# LIMIT FRAMES IF REQUESTED
# ============================================================

if MAX_FRAMES is not None:

    sequence_df = sequence_df.iloc[
        :MAX_FRAMES
    ].copy()


# ============================================================
# CREATE FRAME LOOKUP
# ============================================================

manifest_sequence = manifest[
    manifest[
        "sequence_key"
    ]
    ==
    SELECTED_SEQUENCE
].copy()


manifest_lookup = {}

for _, row in manifest_sequence.iterrows():

    key = int(
        row["frame_key"]
    )

    manifest_lookup[key] = row


print(
    "Manifest sequence records:",
    len(manifest_sequence)
)


# ============================================================
# VIDEO INITIALIZATION
# ============================================================

video_writer = None

generated_frames = 0
failed_frames = 0

records = []


# ============================================================
# FONT
# ============================================================

title_font = load_font(30)
main_font = load_font(24)
small_font = load_font(18)


# ============================================================
# PROCESS FRAMES
# ============================================================

print()
print("=" * 80)
print("GENERATING SEQUENTIAL FRAMES")
print("=" * 80)


for index, prediction_row in sequence_df.iterrows():

    frame_number = int(
        prediction_row[
            "frame_key"
        ]
    )

    try:

        if frame_number not in manifest_lookup:

            raise FileNotFoundError(
                "Manifest frame missing"
            )


        manifest_row = (
            manifest_lookup[
                frame_number
            ]
        )


        image_path = resolve_image_path(
            manifest_row
        )

        json_path = resolve_json_path(
            manifest_row
        )


        if image_path is None:

            raise FileNotFoundError(
                "Image could not be resolved"
            )


        if json_path is None:

            raise FileNotFoundError(
                "JSON could not be resolved"
            )


        image = cv2.imread(
            image_path
        )


        if image is None:

            raise ValueError(
                "OpenCV could not read image"
            )


        original_height, original_width = (
            image.shape[:2]
        )


        image = resize_keep_ratio(
            image,
            MAX_WIDTH
        )


        new_height, new_width = (
            image.shape[:2]
        )


        scale_x = (
            new_width
            /
            float(original_width)
        )

        scale_y = (
            new_height
            /
            float(original_height)
        )


        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        pil_image = Image.fromarray(
            image_rgb
        )


        draw = ImageDraw.Draw(
            pil_image
        )


        polygons = load_polygons(
            json_path
        )


        # ----------------------------------------------------
        # DRAW CRACK POLYGONS
        # ----------------------------------------------------

        for polygon in polygons:

            draw_polygon(
                draw,
                polygon,
                scale_x,
                scale_y
            )


        # ----------------------------------------------------
        # PREDICTION VALUES
        # ----------------------------------------------------

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

        error = float(
            prediction_row[
                "absolute_error"
            ]
        )

        status = str(
            prediction_row[
                "predicted_damage_status"
            ]
        )

        severity = str(
            prediction_row[
                "predicted_damage_level"
            ]
        )


        # ----------------------------------------------------
        # INFORMATION PANEL
        # ----------------------------------------------------

        panel_height = 245

        panel = Image.new(
            "RGB",
            (
                new_width,
                panel_height
            ),
            (20, 20, 20)
        )


        panel_draw = ImageDraw.Draw(
            panel
        )


        panel_draw.text(
            (25, 15),
            "BELTCRACK9KD DAMAGE INTELLIGENCE",
            fill=(255, 255, 255),
            font=title_font
        )


        panel_draw.text(
            (25, 65),
            (
                f"Sequence: {SELECTED_SEQUENCE}    "
                f"Frame: {frame_number}"
            ),
            fill=(220, 220, 220),
            font=small_font
        )


        panel_draw.text(
            (25, 105),
            (
                f"Current Damage: {current_damage:.6f}"
            ),
            fill=(255, 255, 255),
            font=main_font
        )


        panel_draw.text(
            (350, 105),
            (
                f"Predicted Next: {predicted_damage:.6f}"
            ),
            fill=(255, 255, 255),
            font=main_font
        )


        panel_draw.text(
            (720, 105),
            (
                f"Actual Next: {actual_damage:.6f}"
            ),
            fill=(255, 255, 255),
            font=main_font
        )


        panel_draw.text(
            (25, 150),
            (
                f"Error: {error:.6f}"
            ),
            fill=(220, 220, 220),
            font=main_font
        )


        panel_draw.text(
            (300, 150),
            (
                f"Status: {status}"
            ),
            fill=(255, 255, 255),
            font=main_font
        )


        panel_draw.text(
            (570, 150),
            (
                f"Severity: {severity}"
            ),
            fill=(255, 255, 255),
            font=main_font
        )


        panel_draw.text(
            (25, 195),
            (
                f"Crack polygons: {len(polygons)}"
            ),
            fill=(220, 220, 220),
            font=small_font
        )


        panel_draw.text(
            (300, 195),
            (
                f"Progress: {index + 1}/{len(sequence_df)}"
            ),
            fill=(220, 220, 220),
            font=small_font
        )


        # ----------------------------------------------------
        # COMBINE PANEL + IMAGE
        # ----------------------------------------------------

        final_image = Image.new(
            "RGB",
            (
                new_width,
                panel_height + new_height
            )
        )


        final_image.paste(
            panel,
            (0, 0)
        )


        final_image.paste(
            pil_image,
            (0, panel_height)
        )


        frame_array = np.asarray(
            final_image
        )


        frame_bgr = cv2.cvtColor(
            frame_array,
            cv2.COLOR_RGB2BGR
        )


        # ----------------------------------------------------
        # INITIALIZE VIDEO WRITER
        # ----------------------------------------------------

        if video_writer is None:

            output_height = (
                frame_bgr.shape[0]
            )

            output_width = (
                frame_bgr.shape[1]
            )


            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            video_writer = cv2.VideoWriter(
                VIDEO_PATH,
                fourcc,
                FPS,
                (
                    output_width,
                    output_height
                )
            )


            if not video_writer.isOpened():

                raise RuntimeError(
                    "Could not initialize video writer"
                )


        video_writer.write(
            frame_bgr
        )


        # ----------------------------------------------------
        # RECORD DATA
        # ----------------------------------------------------

        records.append(
            {
                "sequence":
                    SELECTED_SEQUENCE,

                "frame":
                    frame_number,

                "crack_count":
                    int(len(polygons)),

                "current_damage":
                    current_damage,

                "predicted_next_damage":
                    predicted_damage,

                "actual_next_damage":
                    actual_damage,

                "prediction_error":
                    error,

                "status":
                    status,

                "severity":
                    severity,

                "image_path":
                    image_path,

                "json_path":
                    json_path
            }
        )


        generated_frames += 1


        # ----------------------------------------------------
        # OPTIONAL FRAME EXPORT
        # ----------------------------------------------------

        frame_path = os.path.join(
            FRAME_DIR,
            f"frame_{frame_number:06d}.jpg"
        )


        cv2.imwrite(
            frame_path,
            frame_bgr
        )


        if (
            generated_frames == 1
            or
            generated_frames % 50 == 0
            or
            generated_frames == len(sequence_df)
        ):

            print(
                f"Processed "
                f"{generated_frames}/"
                f"{len(sequence_df)}"
            )


    except Exception as exc:

        failed_frames += 1

        print(
            f"FAILED frame {frame_number}: "
            f"{exc}"
        )


# ============================================================
# RELEASE VIDEO
# ============================================================

if video_writer is not None:

    video_writer.release()


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

if len(results_df) > 0:

    mae = float(
        np.mean(
            np.abs(
                results_df[
                    "predicted_next_damage"
                ]
                -
                results_df[
                    "actual_next_damage"
                ]
            )
        )
    )


    rmse = float(
        np.sqrt(
            np.mean(
                (
                    results_df[
                        "predicted_next_damage"
                    ]
                    -
                    results_df[
                        "actual_next_damage"
                    ]
                ) ** 2
            )
        )
    )


    mean_current = float(
        results_df[
            "current_damage"
        ].mean()
    )


    mean_predicted = float(
        results_df[
            "predicted_next_damage"
        ].mean()
    )


    mean_actual = float(
        results_df[
            "actual_next_damage"
        ].mean()
    )

else:

    mae = None
    rmse = None
    mean_current = None
    mean_predicted = None
    mean_actual = None


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "stage":
        "4.2",

    "status":
        "PASS"
        if (
            generated_frames > 0
            and
            failed_frames == 0
            and
            video_writer is not None
        )
        else "REVIEW",

    "sequence":
        SELECTED_SEQUENCE,

    "fps":
        FPS,

    "requested_frames":
        int(len(sequence_df)),

    "generated_frames":
        int(generated_frames),

    "failed_frames":
        int(failed_frames),

    "sequence_mae":
        mae,

    "sequence_rmse":
        rmse,

    "mean_current_damage":
        mean_current,

    "mean_predicted_next_damage":
        mean_predicted,

    "mean_actual_next_damage":
        mean_actual,

    "video_created":
        bool(
            os.path.exists(
                VIDEO_PATH
            )
        ),

    "video_path":
        VIDEO_PATH,

    "frame_directory":
        FRAME_DIR,

    "csv_path":
        CSV_PATH,

    "summary_path":
        SUMMARY_PATH
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("STAGE 4.2 FINAL SUMMARY")
print("=" * 80)

print(
    "Sequence:",
    SELECTED_SEQUENCE
)

print(
    "Requested frames:",
    len(sequence_df)
)

print(
    "Generated frames:",
    generated_frames
)

print(
    "Failed frames:",
    failed_frames
)

print(
    "Sequence MAE:",
    f"{mae:.8f}"
    if mae is not None
    else "N/A"
)

print(
    "Sequence RMSE:",
    f"{rmse:.8f}"
    if rmse is not None
    else "N/A"
)

print(
    "Video created:",
    os.path.exists(
        VIDEO_PATH
    )
)

print()
print(
    "Video:",
    VIDEO_PATH
)

print(
    "Frames:",
    FRAME_DIR
)

print(
    "CSV:",
    CSV_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print()
print("=" * 80)

if (
    generated_frames > 0
    and
    failed_frames == 0
    and
    os.path.exists(VIDEO_PATH)
):

    print(
        "STAGE 4.2 STATUS: PASS"
    )

else:

    print(
        "STAGE 4.2 STATUS: REVIEW"
    )

print("=" * 80)