import os
import json
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw


# ============================================================
# STAGE 2.9
# LOCAL CRACK-REGION TRANSITION ANALYSIS
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

RAW_DATASET_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

STAGE_2_7_CSV = os.path.join(
    AUDIT_DIR,
    "stage_2_7_extreme_transition_validation.csv"
)

OUTPUT_CSV = os.path.join(
    AUDIT_DIR,
    "stage_2_9_local_crack_transition_analysis.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_2_9_local_crack_transition_summary.json"
)

VISUAL_DIR = os.path.join(
    AUDIT_DIR,
    "stage_2_9_local_transition_visualizations"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_N = 30

ROI_PADDING_RATIO = 0.15

MAX_ROI_WIDTH = 640
MAX_ROI_HEIGHT = 640

VISUALIZATION_SIZE = 640


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_dataset_path(path_value):

    if path_value is None:
        return None

    path_value = str(path_value).strip()

    if not path_value:
        return None

    if os.path.isabs(path_value):
        return os.path.normpath(path_value)

    normalized = path_value.replace(
        "/",
        os.sep
    )

    return os.path.normpath(
        os.path.join(
            RAW_DATASET_ROOT,
            normalized
        )
    )


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# POLYGON EXTRACTION
# ============================================================

def get_crack_polygons(data):

    polygons = []

    shapes = data.get(
        "shapes",
        []
    )

    for shape in shapes:

        label = str(
            shape.get(
                "label",
                ""
            )
        ).strip().lower()

        if label != "crack":
            continue

        points = shape.get(
            "points",
            []
        )

        if not isinstance(
            points,
            list
        ):
            continue

        if len(points) < 3:
            continue

        valid_points = []

        for point in points:

            if not isinstance(
                point,
                (list, tuple)
            ):
                continue

            if len(point) < 2:
                continue

            try:

                x = float(point[0])
                y = float(point[1])

                valid_points.append(
                    (x, y)
                )

            except Exception:
                continue

        if len(valid_points) >= 3:

            polygons.append(
                valid_points
            )

    return polygons


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(polygon):

    if len(polygon) < 3:
        return 0.0

    area = 0.0

    for i in range(
        len(polygon)
    ):

        x1, y1 = polygon[i]

        x2, y2 = polygon[
            (i + 1) % len(polygon)
        ]

        area += (
            x1 * y2
            -
            x2 * y1
        )

    return abs(area) * 0.5


# ============================================================
# POLYGON STATISTICS
# ============================================================

def polygon_statistics(
    polygons
):

    if not polygons:

        return {
            "polygon_count": 0,
            "total_area": 0.0,
            "largest_area": 0.0,
            "smallest_area": 0.0,
            "bbox": None
        }

    areas = []

    xs = []
    ys = []

    for polygon in polygons:

        area = polygon_area(
            polygon
        )

        areas.append(
            area
        )

        for x, y in polygon:

            xs.append(x)
            ys.append(y)

    bbox = (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )

    return {
        "polygon_count": len(
            polygons
        ),

        "total_area": float(
            sum(areas)
        ),

        "largest_area": float(
            max(areas)
        ),

        "smallest_area": float(
            min(areas)
        ),

        "bbox": bbox
    }


# ============================================================
# BOUNDING BOX
# ============================================================

def expand_bbox(
    bbox,
    width,
    height,
    padding_ratio
):

    if bbox is None:

        return (
            0,
            0,
            width,
            height
        )

    x1, y1, x2, y2 = bbox

    bbox_width = max(
        1.0,
        x2 - x1
    )

    bbox_height = max(
        1.0,
        y2 - y1
    )

    padding_x = (
        bbox_width
        *
        padding_ratio
    )

    padding_y = (
        bbox_height
        *
        padding_ratio
    )

    x1 = max(
        0,
        int(
            np.floor(
                x1 - padding_x
            )
        )
    )

    y1 = max(
        0,
        int(
            np.floor(
                y1 - padding_y
            )
        )
    )

    x2 = min(
        width,
        int(
            np.ceil(
                x2 + padding_x
            )
        )
    )

    y2 = min(
        height,
        int(
            np.ceil(
                y2 + padding_y
            )
        )
    )

    if x2 <= x1:
        x2 = min(
            width,
            x1 + 1
        )

    if y2 <= y1:
        y2 = min(
            height,
            y1 + 1
        )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# UNION OF TWO BBOXES
# ============================================================

def union_bboxes(
    bbox1,
    bbox2,
    width,
    height,
    padding_ratio
):

    boxes = []

    if bbox1 is not None:
        boxes.append(bbox1)

    if bbox2 is not None:
        boxes.append(bbox2)

    if not boxes:

        return (
            0,
            0,
            width,
            height
        )

    x1 = min(
        box[0]
        for box in boxes
    )

    y1 = min(
        box[1]
        for box in boxes
    )

    x2 = max(
        box[2]
        for box in boxes
    )

    y2 = max(
        box[3]
        for box in boxes
    )

    return expand_bbox(
        (
            x1,
            y1,
            x2,
            y2
        ),
        width,
        height,
        padding_ratio
    )


# ============================================================
# BBOX CENTER
# ============================================================

def bbox_center(
    bbox
):

    if bbox is None:

        return (
            np.nan,
            np.nan
        )

    return (
        (
            bbox[0]
            +
            bbox[2]
        )
        /
        2.0,

        (
            bbox[1]
            +
            bbox[3]
        )
        /
        2.0
    )


# ============================================================
# BBOX IOU
# ============================================================

def bbox_iou(
    bbox1,
    bbox2
):

    if (
        bbox1 is None
        or
        bbox2 is None
    ):

        return np.nan

    intersection_x1 = max(
        bbox1[0],
        bbox2[0]
    )

    intersection_y1 = max(
        bbox1[1],
        bbox2[1]
    )

    intersection_x2 = min(
        bbox1[2],
        bbox2[2]
    )

    intersection_y2 = min(
        bbox1[3],
        bbox2[3]
    )

    intersection_width = max(
        0.0,
        intersection_x2
        -
        intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2
        -
        intersection_y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = (
        max(
            0.0,
            bbox1[2]
            -
            bbox1[0]
        )
        *
        max(
            0.0,
            bbox1[3]
            -
            bbox1[1]
        )
    )

    area2 = (
        max(
            0.0,
            bbox2[2]
            -
            bbox2[0]
        )
        *
        max(
            0.0,
            bbox2[3]
            -
            bbox2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return np.nan

    return float(
        intersection / union
    )


# ============================================================
# BBOX CENTER DISTANCE
# ============================================================

def bbox_center_distance(
    bbox1,
    bbox2,
    width,
    height
):

    if (
        bbox1 is None
        or
        bbox2 is None
    ):

        return np.nan

    c1 = bbox_center(
        bbox1
    )

    c2 = bbox_center(
        bbox2
    )

    dx = (
        c2[0]
        -
        c1[0]
    )

    dy = (
        c2[1]
        -
        c1[1]
    )

    distance = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    diagonal = np.sqrt(
        width ** 2
        +
        height ** 2
    )

    if diagonal <= 0:
        return np.nan

    return float(
        distance / diagonal
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_gray(
    image,
    size
):

    gray = image.convert(
        "L"
    )

    gray = gray.resize(
        size
    )

    return np.asarray(
        gray,
        dtype=np.float32
    )


# ============================================================
# IMAGE COMPARISON
# ============================================================

def compare_images(
    current,
    future
):

    width = min(
        current.width,
        future.width
    )

    height = min(
        current.height,
        future.height
    )

    width = max(
        64,
        min(
            width,
            512
        )
    )

    height = max(
        64,
        min(
            height,
            512
        )
    )

    size = (
        width,
        height
    )

    current_gray = prepare_gray(
        current,
        size
    )

    future_gray = prepare_gray(
        future,
        size
    )

    a = current_gray.reshape(
        -1
    )

    b = future_gray.reshape(
        -1
    )

    std_a = float(
        a.std()
    )

    std_b = float(
        b.std()
    )

    if (
        std_a > 1e-8
        and
        std_b > 1e-8
    ):

        correlation = float(
            np.corrcoef(
                a,
                b
            )[0, 1]
        )

    else:

        correlation = np.nan

    mean_absolute_difference = float(
        np.mean(
            np.abs(
                current_gray
                -
                future_gray
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    current_gray
                    -
                    future_gray
                )
                ** 2
            )
        )
    )

    normalized_difference = (
        mean_absolute_difference
        /
        255.0
    )

    return {
        "local_image_correlation":
            correlation,

        "local_mean_absolute_difference":
            mean_absolute_difference,

        "local_pixel_rmse":
            rmse,

        "local_normalized_difference":
            normalized_difference
    }


# ============================================================
# TRANSITION CLASSIFICATION
# ============================================================

def classify_transition(
    local_correlation,
    damage_change,
    local_difference,
    bbox_iou_value
):

    if pd.isna(
        local_correlation
    ):

        return "UNKNOWN"

    absolute_damage_change = abs(
        float(
            damage_change
        )
    )

    large_damage = (
        absolute_damage_change
        >=
        0.005
    )

    strong_local_similarity = (
        local_correlation
        >=
        0.95
    )

    strong_local_change = (
        local_difference
        >=
        0.08
    )

    low_bbox_overlap = (
        not pd.isna(
            bbox_iou_value
        )
        and
        bbox_iou_value
        <
        0.25
    )

    if (
        large_damage
        and
        strong_local_similarity
        and
        not strong_local_change
    ):

        return (
            "ANNOTATION_DOMINANT_OR_SUBTLE_CHANGE"
        )

    if (
        large_damage
        and
        strong_local_change
    ):

        return (
            "VISUALLY_SUPPORTED_CHANGE"
        )

    if (
        large_damage
        and
        low_bbox_overlap
    ):

        return (
            "SPATIAL_TRANSITION"
        )

    if large_damage:

        return (
            "AMBIGUOUS_LARGE_CHANGE"
        )

    return "NORMAL_TRANSITION"


# ============================================================
# DRAW POLYGONS
# ============================================================

def draw_polygons(
    image,
    polygons,
    bbox
):

    image = image.copy()

    draw = ImageDraw.Draw(
        image
    )

    for polygon in polygons:

        points = []

        for x, y in polygon:

            points.append(
                (
                    int(x),
                    int(y)
                )
            )

        if len(points) >= 3:

            draw.line(
                points + [
                    points[0]
                ],
                fill=(255, 255, 255),
                width=4
            )

    if bbox is not None:

        draw.rectangle(
            (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3])
            ),
            outline=(255, 255, 255),
            width=5
        )

    return image


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    current_image,
    future_image,
    current_polygons,
    future_polygons,
    current_bbox,
    future_bbox,
    output_path
):

    current = draw_polygons(
        current_image,
        current_polygons,
        current_bbox
    )

    future = draw_polygons(
        future_image,
        future_polygons,
        future_bbox
    )

    current.thumbnail(
        (
            VISUALIZATION_SIZE,
            VISUALIZATION_SIZE
        )
    )

    future.thumbnail(
        (
            VISUALIZATION_SIZE,
            VISUALIZATION_SIZE
        )
    )

    canvas_width = (
        current.width
        +
        future.width
    )

    canvas_height = max(
        current.height,
        future.height
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height
        )
    )

    canvas.paste(
        current,
        (
            0,
            0
        )
    )

    canvas.paste(
        future,
        (
            current.width,
            0
        )
    )

    canvas.save(
        output_path,
        quality=90
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 2.9 - LOCAL CRACK-REGION TRANSITION ANALYSIS"
)
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

print()
print(
    "Checking raw dataset root..."
)

if os.path.exists(
    RAW_DATASET_ROOT
):

    print(
        "Raw dataset: FOUND"
    )

    print(
        RAW_DATASET_ROOT
    )

else:

    raise FileNotFoundError(
        "Raw dataset not found:\n"
        +
        RAW_DATASET_ROOT
    )


# ============================================================
# CHECK STAGE 2.7
# ============================================================

print()
print(
    "Checking Stage 2.7 validation file..."
)

if not os.path.exists(
    STAGE_2_7_CSV
):

    raise FileNotFoundError(
        "Stage 2.7 CSV not found:\n"
        +
        STAGE_2_7_CSV
    )

print(
    "Stage 2.7 validation: FOUND"
)


# ============================================================
# CREATE VISUAL DIRECTORY
# ============================================================

os.makedirs(
    VISUAL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CASES
# ============================================================

print()
print(
    "Loading Stage 2.7 extreme transitions..."
)

cases = pd.read_csv(
    STAGE_2_7_CSV
)

print(
    f"Available validation records: "
    f"{len(cases)}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "sequence",
    "component",
    "split",
    "current_frame",
    "future_frame",
    "current_damage",
    "future_damage",
    "damage_change",
    "absolute_damage_change",
    "model_absolute_error",
    "current_image",
    "future_image",
    "current_json",
    "future_json"
]

missing_columns = [
    column
    for column in required_columns
    if column not in cases.columns
]

if missing_columns:

    raise ValueError(
        "Stage 2.7 CSV is missing required columns:\n"
        +
        ", ".join(
            missing_columns
        )
    )


# ============================================================
# TOP EXTREME EVENTS
# ============================================================

cases = cases.sort_values(
    "absolute_damage_change",
    ascending=False
).head(
    TOP_N
).copy()

print(
    f"Extreme events selected: "
    f"{len(cases)}"
)


# ============================================================
# ANALYSIS
# ============================================================

print()
print("=" * 70)
print(
    "ANALYZING LOCAL CRACK REGIONS"
)
print("=" * 70)


results = []


for _, row in cases.iterrows():

    current_path = resolve_dataset_path(
        row[
            "current_image"
        ]
    )

    future_path = resolve_dataset_path(
        row[
            "future_image"
        ]
    )

    current_json_path = resolve_dataset_path(
        row[
            "current_json"
        ]
    )

    future_json_path = resolve_dataset_path(
        row[
            "future_json"
        ]
    )


    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    required_paths = [
        current_path,
        future_path,
        current_json_path,
        future_json_path
    ]

    if not all(
        path is not None
        and
        os.path.exists(path)
        for path in required_paths
    ):

        print(
            "WARNING: missing source file:"
        )

        print(
            row["sequence"],
            row["current_frame"],
            row["future_frame"]
        )

        continue


    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    try:

        current_image = Image.open(
            current_path
        ).convert(
            "RGB"
        )

        future_image = Image.open(
            future_path
        ).convert(
            "RGB"
        )

    except Exception as exc:

        print(
            "WARNING: image loading failed:"
        )

        print(
            str(exc)
        )

        continue


    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    try:

        current_data = load_json(
            current_json_path
        )

        future_data = load_json(
            future_json_path
        )

    except Exception as exc:

        print(
            "WARNING: JSON loading failed:"
        )

        print(
            str(exc)
        )

        continue


    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    current_width = (
        current_image.width
    )

    current_height = (
        current_image.height
    )

    future_width = (
        future_image.width
    )

    future_height = (
        future_image.height
    )


    # --------------------------------------------------------
    # POLYGONS
    # --------------------------------------------------------

    current_polygons = (
        get_crack_polygons(
            current_data
        )
    )

    future_polygons = (
        get_crack_polygons(
            future_data
        )
    )


    # --------------------------------------------------------
    # POLYGON STATISTICS
    # --------------------------------------------------------

    current_stats = polygon_statistics(
        current_polygons
    )

    future_stats = polygon_statistics(
        future_polygons
    )


    current_bbox = (
        current_stats["bbox"]
    )

    future_bbox = (
        future_stats["bbox"]
    )


    # --------------------------------------------------------
    # COMMON ROI
    # --------------------------------------------------------

    common_width = min(
        current_width,
        future_width
    )

    common_height = min(
        current_height,
        future_height
    )

    union_bbox = union_bboxes(
        current_bbox,
        future_bbox,
        common_width,
        common_height,
        ROI_PADDING_RATIO
    )


    # --------------------------------------------------------
    # LIMIT ROI TO IMAGE BOUNDS
    # --------------------------------------------------------

    roi_x1 = max(
        0,
        min(
            union_bbox[0],
            common_width - 1
        )
    )

    roi_y1 = max(
        0,
        min(
            union_bbox[1],
            common_height - 1
        )
    )

    roi_x2 = max(
        roi_x1 + 1,
        min(
            union_bbox[2],
            common_width
        )
    )

    roi_y2 = max(
        roi_y1 + 1,
        min(
            union_bbox[3],
            common_height
        )
    )

    union_bbox = (
        int(roi_x1),
        int(roi_y1),
        int(roi_x2),
        int(roi_y2)
    )


    # --------------------------------------------------------
    # CROP LOCAL REGION
    # --------------------------------------------------------

    current_roi = current_image.crop(
        union_bbox
    )

    future_roi = future_image.crop(
        union_bbox
    )


    # --------------------------------------------------------
    # LOCAL IMAGE METRICS
    # --------------------------------------------------------

    local_metrics = compare_images(
        current_roi,
        future_roi
    )


    # --------------------------------------------------------
    # SPATIAL METRICS
    # --------------------------------------------------------

    spatial_iou = bbox_iou(
        current_bbox,
        future_bbox
    )

    spatial_distance = bbox_center_distance(
        current_bbox,
        future_bbox,
        common_width,
        common_height
    )


    # --------------------------------------------------------
    # ANNOTATION AREA RATIOS
    # --------------------------------------------------------

    current_image_area = (
        current_width
        *
        current_height
    )

    future_image_area = (
        future_width
        *
        future_height
    )

    if current_image_area > 0:

        current_area_ratio = (
            current_stats[
                "total_area"
            ]
            /
            current_image_area
        )

    else:

        current_area_ratio = np.nan


    if future_image_area > 0:

        future_area_ratio = (
            future_stats[
                "total_area"
            ]
            /
            future_image_area
        )

    else:

        future_area_ratio = np.nan


    annotation_ratio_change = (
        future_area_ratio
        -
        current_area_ratio
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    transition_class = classify_transition(
        local_metrics[
            "local_image_correlation"
        ],
        float(
            row[
                "damage_change"
            ]
        ),
        local_metrics[
            "local_normalized_difference"
        ],
        spatial_iou
    )


    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    event_number = (
        len(results)
        +
        1
    )

    sequence_name = str(
        row["sequence"]
    ).replace(
        "/",
        "_"
    )

    visualization_filename = (
        f"{event_number:02d}_"
        f"{sequence_name}_"
        f"{int(row['current_frame'])}_"
        f"to_"
        f"{int(row['future_frame'])}.jpg"
    )

    visualization_path = os.path.join(
        VISUAL_DIR,
        visualization_filename
    )

    try:

        create_visualization(
            current_image,
            future_image,
            current_polygons,
            future_polygons,
            current_bbox,
            future_bbox,
            visualization_path
        )

    except Exception as exc:

        print(
            "WARNING: visualization failed:"
        )

        print(
            str(exc)
        )

        visualization_path = ""


    # --------------------------------------------------------
    # RESULT RECORD
    # --------------------------------------------------------

    result = {

        "sequence":
            row["sequence"],

        "component":
            int(
                row["component"]
            ),

        "split":
            row["split"],

        "current_frame":
            int(
                row["current_frame"]
            ),

        "future_frame":
            int(
                row["future_frame"]
            ),

        "current_damage":
            float(
                row["current_damage"]
            ),

        "future_damage":
            float(
                row["future_damage"]
            ),

        "damage_change":
            float(
                row["damage_change"]
            ),

        "absolute_damage_change":
            float(
                row[
                    "absolute_damage_change"
                ]
            ),

        "model_absolute_error":
            float(
                row[
                    "model_absolute_error"
                ]
            ),

        "current_polygon_count":
            int(
                current_stats[
                    "polygon_count"
                ]
            ),

        "future_polygon_count":
            int(
                future_stats[
                    "polygon_count"
                ]
            ),

        "polygon_count_change":
            int(
                future_stats[
                    "polygon_count"
                ]
                -
                current_stats[
                    "polygon_count"
                ]
            ),

        "current_total_polygon_area":
            float(
                current_stats[
                    "total_area"
                ]
            ),

        "future_total_polygon_area":
            float(
                future_stats[
                    "total_area"
                ]
            ),

        "current_area_ratio":
            float(
                current_area_ratio
            ),

        "future_area_ratio":
            float(
                future_area_ratio
            ),

        "annotation_ratio_change":
            float(
                annotation_ratio_change
            ),

        "absolute_annotation_ratio_change":
            float(
                abs(
                    annotation_ratio_change
                )
            ),

        "bbox_iou":
            spatial_iou,

        "bbox_center_distance_normalized":
            spatial_distance,

        "local_image_correlation":
            local_metrics[
                "local_image_correlation"
            ],

        "local_mean_absolute_difference":
            local_metrics[
                "local_mean_absolute_difference"
            ],

        "local_pixel_rmse":
            local_metrics[
                "local_pixel_rmse"
            ],

        "local_normalized_difference":
            local_metrics[
                "local_normalized_difference"
            ],

        "transition_class":
            transition_class,

        "current_image":
            current_path,

        "future_image":
            future_path,

        "current_json":
            current_json_path,

        "future_json":
            future_json_path,

        "visualization":
            visualization_path
    }

    results.append(
        result
    )

    print(
        f"Analyzed {len(results):02d}/"
        f"{len(cases):02d}: "
        f"{row['sequence']} "
        f"{int(row['current_frame'])} -> "
        f"{int(row['future_frame'])} "
        f"| local corr="
        f"{local_metrics['local_image_correlation']:.4f} "
        f"| class="
        f"{transition_class}"
    )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "No transitions were successfully analyzed."
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = (
    results_df[
        "transition_class"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# INDIVIDUAL COUNTS
# ============================================================

visually_supported = int(
    (
        results_df[
            "transition_class"
        ]
        ==
        "VISUALLY_SUPPORTED_CHANGE"
    ).sum()
)

annotation_dominant = int(
    (
        results_df[
            "transition_class"
        ]
        ==
        "ANNOTATION_DOMINANT_OR_SUBTLE_CHANGE"
    ).sum()
)

spatial_transition = int(
    (
        results_df[
            "transition_class"
        ]
        ==
        "SPATIAL_TRANSITION"
    ).sum()
)

ambiguous = int(
    (
        results_df[
            "transition_class"
        ]
        ==
        "AMBIGUOUS_LARGE_CHANGE"
    ).sum()
)

unknown = int(
    (
        results_df[
            "transition_class"
        ]
        ==
        "UNKNOWN"
    ).sum()
)


# ============================================================
# CORRELATIONS
# ============================================================

damage_corr_data = results_df[
    [
        "local_image_correlation",
        "absolute_damage_change"
    ]
].dropna()

if len(damage_corr_data) >= 3:

    local_corr_vs_damage = float(
        damage_corr_data[
            "local_image_correlation"
        ].corr(
            damage_corr_data[
                "absolute_damage_change"
            ]
        )
    )

else:

    local_corr_vs_damage = None


error_corr_data = results_df[
    [
        "local_image_correlation",
        "model_absolute_error"
    ]
].dropna()

if len(error_corr_data) >= 3:

    local_corr_vs_error = float(
        error_corr_data[
            "local_image_correlation"
        ].corr(
            error_corr_data[
                "model_absolute_error"
            ]
        )
    )

else:

    local_corr_vs_error = None


# ============================================================
# VALID LOCAL METRICS
# ============================================================

valid_local_corr = (
    results_df[
        "local_image_correlation"
    ]
    .dropna()
)

valid_local_difference = (
    results_df[
        "local_normalized_difference"
    ]
    .dropna()
)

valid_bbox_iou = (
    results_df[
        "bbox_iou"
    ]
    .dropna()
)

valid_center_distance = (
    results_df[
        "bbox_center_distance_normalized"
    ]
    .dropna()
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "2.9",

    "status":
        "PASS",

    "events_requested":
        TOP_N,

    "events_analyzed":
        int(
            len(results_df)
        ),

    "transition_class_counts":
        class_counts,

    "visually_supported_changes":
        visually_supported,

    "annotation_dominant_or_subtle_changes":
        annotation_dominant,

    "spatial_transitions":
        spatial_transition,

    "ambiguous_large_changes":
        ambiguous,

    "unknown_transitions":
        unknown,

    "mean_local_image_correlation":
        float(
            valid_local_corr.mean()
        )
        if len(valid_local_corr) > 0
        else None,

    "median_local_image_correlation":
        float(
            valid_local_corr.median()
        )
        if len(valid_local_corr) > 0
        else None,

    "mean_local_normalized_difference":
        float(
            valid_local_difference.mean()
        )
        if len(valid_local_difference) > 0
        else None,

    "median_local_normalized_difference":
        float(
            valid_local_difference.median()
        )
        if len(valid_local_difference) > 0
        else None,

    "mean_bbox_iou":
        float(
            valid_bbox_iou.mean()
        )
        if len(valid_bbox_iou) > 0
        else None,

    "mean_normalized_bbox_center_distance":
        float(
            valid_center_distance.mean()
        )
        if len(valid_center_distance) > 0
        else None,

    "local_image_correlation_vs_damage_change":
        local_corr_vs_damage,

    "local_image_correlation_vs_model_error":
        local_corr_vs_error,

    "interpretation":
        (
            "Stage 2.9 compares the local region containing "
            "the current and future crack annotations. "
            "Large target changes are classified according "
            "to local visual change, spatial displacement, "
            "or visually subtle transitions."
        ),

    "output_csv":
        OUTPUT_CSV,

    "visualization_directory":
        VISUAL_DIR
}


# ============================================================
# SAVE SUMMARY JSON
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
# TOP RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "TOP LOCAL TRANSITION RESULTS"
)
print("=" * 70)

display_columns = [
    "sequence",
    "current_frame",
    "future_frame",
    "current_damage",
    "future_damage",
    "absolute_damage_change",
    "local_image_correlation",
    "local_normalized_difference",
    "bbox_iou",
    "transition_class"
]

print(
    results_df.sort_values(
        "absolute_damage_change",
        ascending=False
    )[
        display_columns
    ].head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 2.9 SUMMARY"
)
print("=" * 70)

print(
    f"Events requested: "
    f"{TOP_N}"
)

print(
    f"Events analyzed: "
    f"{len(results_df)}"
)

print()
print(
    "Transition classes:"
)

for class_name in sorted(
    class_counts.keys()
):

    print(
        f"  {class_name}: "
        f"{class_counts[class_name]}"
    )

print()
print(
    f"Visually supported changes: "
    f"{visually_supported}"
)

print(
    f"Annotation-dominant or subtle changes: "
    f"{annotation_dominant}"
)

print(
    f"Spatial transitions: "
    f"{spatial_transition}"
)

print(
    f"Ambiguous large changes: "
    f"{ambiguous}"
)

print(
    f"Unknown transitions: "
    f"{unknown}"
)

print()
print(
    "Local image statistics:"
)

if len(valid_local_corr) > 0:

    print(
        f"  Mean local image correlation: "
        f"{valid_local_corr.mean():.6f}"
    )

    print(
        f"  Median local image correlation: "
        f"{valid_local_corr.median():.6f}"
    )

else:

    print(
        "  Local image correlation: unavailable"
    )

if len(valid_local_difference) > 0:

    print(
        f"  Mean local normalized difference: "
        f"{valid_local_difference.mean():.6f}"
    )

    print(
        f"  Median local normalized difference: "
        f"{valid_local_difference.median():.6f}"
    )

else:

    print(
        "  Local normalized difference: unavailable"
    )

print()

print(
    "Spatial statistics:"
)

if len(valid_bbox_iou) > 0:

    print(
        f"  Mean bbox IoU: "
        f"{valid_bbox_iou.mean():.6f}"
    )

else:

    print(
        "  Mean bbox IoU: unavailable"
    )

if len(valid_center_distance) > 0:

    print(
        f"  Mean normalized bbox center distance: "
        f"{valid_center_distance.mean():.6f}"
    )

else:

    print(
        "  Mean normalized bbox center distance: "
        "unavailable"
    )

print()

print(
    "Correlation with absolute damage change: "
    f"{local_corr_vs_damage}"
)

print(
    "Correlation with model absolute error: "
    f"{local_corr_vs_error}"
)

print()
print(
    "Output CSV:"
)

print(
    OUTPUT_CSV
)

print()
print(
    "Visualizations:"
)

print(
    VISUAL_DIR
)

print()
print(
    "Summary JSON:"
)

print(
    OUTPUT_SUMMARY
)

print()
print(
    "FINAL STATUS: PASS"
)