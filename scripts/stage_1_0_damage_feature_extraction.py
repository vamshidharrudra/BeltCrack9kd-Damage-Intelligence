import os
import csv
import json
import math
from collections import Counter

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# STAGE 1.0
# PER-FRAME PHYSICAL DAMAGE FEATURE EXTRACTION
#
# Input:
#   canonical_manifest.csv
#   canonical annotation JSON files
#   original source images
#
# Output:
#   audit_outputs/stage_1_0_damage_features.csv
#   audit_outputs/stage_1_0_damage_feature_summary.json
#
# IMPORTANT:
#   No images are copied or modified.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

RAW_DATASET = r"C:\Users\vamsh\Downloads\BeltCrack9kd"

CANONICAL_DIR = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset"
)

MANIFEST_PATH = os.path.join(
    CANONICAL_DIR,
    "canonical_manifest.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

FEATURE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_0_damage_features.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stage_1_0_damage_feature_summary.json"
)


# ============================================================
# EXPECTED DATASET
# ============================================================

EXPECTED_TOTAL = 14087

EXPECTED_SPLITS = {
    "train": 9819,
    "val": 2102,
    "test": 2166
}


# ============================================================
# HELPERS
# ============================================================

def resolve_source_image(source_value):

    if not source_value:
        return None

    source_value = str(
        source_value
    ).strip()

    # Absolute path
    if os.path.isabs(source_value):

        path = os.path.normpath(
            source_value
        )

        if os.path.isfile(path):
            return path

    # Relative to raw dataset
    candidate = os.path.normpath(
        os.path.join(
            RAW_DATASET,
            source_value
        )
    )

    if os.path.isfile(candidate):
        return candidate

    # Relative to project
    candidate = os.path.normpath(
        os.path.join(
            PROJECT_ROOT,
            source_value
        )
    )

    if os.path.isfile(candidate):
        return candidate

    return None


def resolve_canonical_json(
    split,
    filename
):

    if not filename:
        return None

    split = str(
        split
    ).strip().lower()

    filename = str(
        filename
    ).strip()

    # Primary expected location
    candidate = os.path.join(
        CANONICAL_DIR,
        split,
        "labels",
        filename
    )

    if os.path.isfile(candidate):
        return os.path.normpath(
            candidate
        )

    # Fallback recursive search
    labels_dir = os.path.join(
        CANONICAL_DIR,
        split,
        "labels"
    )

    if os.path.isdir(labels_dir):

        base_name = os.path.basename(
            filename
        )

        for root, dirs, files in os.walk(
            labels_dir
        ):

            if base_name in files:

                return os.path.join(
                    root,
                    base_name
                )

    return None


def polygon_area(points):

    """
    Shoelace formula.

    points:
        [(x1,y1), (x2,y2), ...]
    """

    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(
        len(points)
    ):

        x1, y1 = points[i]

        x2, y2 = points[
            (i + 1) % len(points)
        ]

        area += (
            x1 * y2
            - x2 * y1
        )

    return abs(area) / 2.0


def polygon_perimeter(points):

    if len(points) < 2:
        return 0.0

    perimeter = 0.0

    for i in range(
        len(points)
    ):

        x1, y1 = points[i]

        x2, y2 = points[
            (i + 1) % len(points)
        ]

        dx = x2 - x1
        dy = y2 - y1

        perimeter += math.sqrt(
            dx * dx
            + dy * dy
        )

    return perimeter


def polygon_centroid(points):

    """
    Area-weighted polygon centroid.

    Falls back to mean coordinates
    for degenerate polygons.
    """

    if len(points) < 3:

        if not points:
            return 0.0, 0.0

        x_mean = sum(
            p[0] for p in points
        ) / len(points)

        y_mean = sum(
            p[1] for p in points
        ) / len(points)

        return x_mean, y_mean


    signed_area = 0.0

    cx = 0.0
    cy = 0.0


    for i in range(
        len(points)
    ):

        x1, y1 = points[i]

        x2, y2 = points[
            (i + 1) % len(points)
        ]

        cross = (
            x1 * y2
            - x2 * y1
        )

        signed_area += cross

        cx += (
            x1 + x2
        ) * cross

        cy += (
            y1 + y2
        ) * cross


    signed_area *= 0.5


    if abs(
        signed_area
    ) < 1e-12:

        x_mean = sum(
            p[0] for p in points
        ) / len(points)

        y_mean = sum(
            p[1] for p in points
        ) / len(points)

        return x_mean, y_mean


    cx /= (
        6.0
        * signed_area
    )

    cy /= (
        6.0
        * signed_area
    )

    return cx, cy


def polygon_bbox(points):

    if not points:

        return (
            0.0,
            0.0,
            0.0,
            0.0
        )

    xs = [
        p[0]
        for p in points
    ]

    ys = [
        p[1]
        for p in points
    ]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


def safe_mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def safe_median(values):

    if not values:
        return 0.0

    values = sorted(values)

    n = len(values)

    middle = n // 2

    if n % 2 == 0:

        return (
            values[middle - 1]
            + values[middle]
        ) / 2.0

    return values[middle]


def safe_max(values):

    if not values:
        return 0.0

    return max(values)


def safe_min(values):

    if not values:
        return 0.0

    return min(values)


def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
    )

    lower = int(
        math.floor(position)
    )

    upper = int(
        math.ceil(position)
    )

    if lower == upper:
        return values[lower]

    weight = (
        position - lower
    )

    return (
        values[lower]
        * (1.0 - weight)
        +
        values[upper]
        * weight
    )


def round_value(value):

    return round(
        float(value),
        6
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 1.0 - PER-FRAME PHYSICAL DAMAGE FEATURE EXTRACTION")
print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK DEPENDENCY
# ============================================================

if Image is None:

    print(
        "\nERROR: Pillow is not installed."
    )

    print(
        "Install it using:"
    )

    print(
        "python -m pip install pillow"
    )

    raise SystemExit(1)


# ============================================================
# CHECK MANIFEST
# ============================================================

if not os.path.isfile(
    MANIFEST_PATH
):

    print(
        "\nERROR: canonical_manifest.csv not found."
    )

    print(
        MANIFEST_PATH
    )

    raise SystemExit(1)


# ============================================================
# LOAD MANIFEST
# ============================================================

print(
    "\nLoading canonical manifest..."
)


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print(
    "Manifest records:",
    len(rows)
)


if len(rows) != EXPECTED_TOTAL:

    print(
        "\nERROR: Expected",
        EXPECTED_TOTAL,
        "records but found",
        len(rows)
    )

    raise SystemExit(1)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [

    "split",

    "component",

    "sequence",

    "original_image_filename",

    "canonical_image_filename",

    "canonical_image_path",

    "original_json_filename",

    "canonical_json_filename",

    "source_image",

    "source_json"
]


actual_columns = list(
    rows[0].keys()
)


missing_columns = [

    c

    for c in required_columns

    if c not in actual_columns
]


if missing_columns:

    print(
        "\nERROR: Missing manifest columns:"
    )

    for column in missing_columns:
        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [

    # Identity
    "record_id",
    "split",
    "component",
    "sequence",

    # Original identity
    "original_image_filename",
    "original_json_filename",

    # Canonical identity
    "canonical_image_filename",
    "canonical_json_filename",

    # Source paths
    "source_image",
    "source_json",

    # Image information
    "image_width_px",
    "image_height_px",
    "image_area_px",

    # Damage count
    "crack_count",

    # Area
    "total_crack_area_px",
    "crack_area_ratio",
    "largest_crack_area_px",
    "largest_crack_area_ratio",
    "smallest_crack_area_px",
    "mean_crack_area_px",
    "median_crack_area_px",

    # Perimeter
    "total_crack_perimeter_px",
    "mean_crack_perimeter_px",
    "max_crack_perimeter_px",

    # Bounding box
    "mean_crack_bbox_width_px",
    "mean_crack_bbox_height_px",
    "mean_crack_bbox_area_px",

    "max_crack_bbox_width_px",
    "max_crack_bbox_height_px",

    # Centroid
    "mean_crack_centroid_x_px",
    "mean_crack_centroid_y_px",

    # Normalized centroid
    "mean_crack_centroid_x_norm",
    "mean_crack_centroid_y_norm",

    # Spatial spread
    "crack_centroid_x_std_px",
    "crack_centroid_y_std_px",

    # Relative spatial spread
    "crack_centroid_x_std_norm",
    "crack_centroid_y_std_norm",

    # Occupancy
    "crack_bbox_union_width_px",
    "crack_bbox_union_height_px",

    # JSON metadata
    "json_image_width_px",
    "json_image_height_px",

    # Quality
    "annotation_shape_count",
    "valid_crack_polygon_count",
    "invalid_shape_count"
]


# ============================================================
# PREPARE OUTPUT
# ============================================================

feature_rows = []

summary = {

    "stage": "1.0",

    "status": "RUNNING",

    "input_manifest": MANIFEST_PATH,

    "output_features": FEATURE_OUTPUT,

    "total_manifest_records": len(rows),

    "processed_records": 0,

    "failed_records": 0,

    "missing_images": 0,

    "missing_json": 0,

    "invalid_json": 0,

    "invalid_shapes": 0,

    "non_crack_shapes": 0,

    "split_counts": {},

    "crack_count": {},

    "damage_area_ratio": {},

    "image_dimensions": {},

    "errors": []
}


split_counter = Counter()

crack_count_values = []

area_ratio_values = []

total_area_values = []

perimeter_values = []


# ============================================================
# PROCESS RECORDS
# ============================================================

print(
    "\nExtracting physical damage features..."
)

print(
    "Total records:",
    len(rows)
)

print()


for index, row in enumerate(
    rows,
    start=1
):

    try:

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        split = str(
            row["split"]
        ).strip().lower()

        component = str(
            row["component"]
        ).strip()

        sequence = str(
            row["sequence"]
        ).strip()


        original_image_filename = str(
            row[
                "original_image_filename"
            ]
        ).strip()


        original_json_filename = str(
            row[
                "original_json_filename"
            ]
        ).strip()


        canonical_image_filename = str(
            row[
                "canonical_image_filename"
            ]
        ).strip()


        canonical_json_filename = str(
            row[
                "canonical_json_filename"
            ]
        ).strip()


        source_image_value = str(
            row[
                "source_image"
            ]
        ).strip()


        source_json_value = str(
            row[
                "source_json"
            ]
        ).strip()


        # ----------------------------------------------------
        # Resolve image
        # ----------------------------------------------------

        image_path = resolve_source_image(
            source_image_value
        )


        if (
            image_path is None
            or not os.path.isfile(
                image_path
            )
        ):

            summary[
                "missing_images"
            ] += 1

            raise RuntimeError(
                "Source image not found: "
                + source_image_value
            )


        # ----------------------------------------------------
        # Resolve canonical JSON
        # ----------------------------------------------------

        json_path = resolve_canonical_json(
            split,
            canonical_json_filename
        )


        if (
            json_path is None
            or not os.path.isfile(
                json_path
            )
        ):

            summary[
                "missing_json"
            ] += 1

            raise RuntimeError(
                "Canonical JSON not found: "
                + canonical_json_filename
            )


        # ----------------------------------------------------
        # Read image dimensions
        # ----------------------------------------------------

        with Image.open(
            image_path
        ) as img:

            image_width = int(
                img.width
            )

            image_height = int(
                img.height
            )


        image_area = (
            image_width
            * image_height
        )


        # ----------------------------------------------------
        # Load annotation JSON
        # ----------------------------------------------------

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                annotation = json.load(f)

        except Exception as e:

            summary[
                "invalid_json"
            ] += 1

            raise RuntimeError(
                "Invalid JSON: "
                + str(e)
            )


        # ----------------------------------------------------
        # JSON dimensions
        # ----------------------------------------------------

        json_width = annotation.get(
            "imageWidth"
        )

        json_height = annotation.get(
            "imageHeight"
        )


        if json_width is not None:

            json_width = int(
                json_width
            )

        if json_height is not None:

            json_height = int(
                json_height
            )


        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        shapes = annotation.get(
            "shapes",
            []
        )


        annotation_shape_count = len(
            shapes
        )


        # ----------------------------------------------------
        # Feature containers
        # ----------------------------------------------------

        areas = []

        perimeters = []

        bbox_widths = []

        bbox_heights = []

        bbox_areas = []

        centroid_x = []

        centroid_y = []


        valid_polygon_count = 0

        invalid_shape_count = 0

        non_crack_shape_count = 0


        # ----------------------------------------------------
        # Process every shape
        # ----------------------------------------------------

        for shape in shapes:

            label = str(
                shape.get(
                    "label",
                    ""
                )
            ).strip().lower()


            if label != "crack":

                non_crack_shape_count += 1

                continue


            points_raw = shape.get(
                "points",
                []
            )


            points = []


            for point in points_raw:

                if (
                    isinstance(point, list)
                    and len(point) >= 2
                ):

                    try:

                        x = float(
                            point[0]
                        )

                        y = float(
                            point[1]
                        )

                        points.append(
                            (x, y)
                        )

                    except Exception:

                        pass


            if len(points) < 3:

                invalid_shape_count += 1

                continue


            # ------------------------------------------------
            # Polygon area
            # ------------------------------------------------

            area = polygon_area(
                points
            )


            # ------------------------------------------------
            # Perimeter
            # ------------------------------------------------

            perimeter = polygon_perimeter(
                points
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            xmin, ymin, xmax, ymax = polygon_bbox(
                points
            )


            bbox_width = max(
                0.0,
                xmax - xmin
            )


            bbox_height = max(
                0.0,
                ymax - ymin
            )


            bbox_area = (
                bbox_width
                * bbox_height
            )


            # ------------------------------------------------
            # Centroid
            # ------------------------------------------------

            cx, cy = polygon_centroid(
                points
            )


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            areas.append(
                area
            )

            perimeters.append(
                perimeter
            )

            bbox_widths.append(
                bbox_width
            )

            bbox_heights.append(
                bbox_height
            )

            bbox_areas.append(
                bbox_area
            )

            centroid_x.append(
                cx
            )

            centroid_y.append(
                cy
            )


            valid_polygon_count += 1


        # ----------------------------------------------------
        # Damage count
        # ----------------------------------------------------

        crack_count = len(
            areas
        )


        # ----------------------------------------------------
        # Area features
        # ----------------------------------------------------

        total_crack_area = sum(
            areas
        )


        crack_area_ratio = 0.0


        if image_area > 0:

            crack_area_ratio = (
                total_crack_area
                / image_area
            )


        largest_crack_area = safe_max(
            areas
        )


        smallest_crack_area = safe_min(
            areas
        )


        mean_crack_area = safe_mean(
            areas
        )


        median_crack_area = safe_median(
            areas
        )


        largest_crack_area_ratio = 0.0


        if image_area > 0:

            largest_crack_area_ratio = (
                largest_crack_area
                / image_area
            )


        # ----------------------------------------------------
        # Perimeter features
        # ----------------------------------------------------

        total_perimeter = sum(
            perimeters
        )


        mean_perimeter = safe_mean(
            perimeters
        )


        max_perimeter = safe_max(
            perimeters
        )


        # ----------------------------------------------------
        # Bounding box features
        # ----------------------------------------------------

        mean_bbox_width = safe_mean(
            bbox_widths
        )


        mean_bbox_height = safe_mean(
            bbox_heights
        )


        mean_bbox_area = safe_mean(
            bbox_areas
        )


        max_bbox_width = safe_max(
            bbox_widths
        )


        max_bbox_height = safe_max(
            bbox_heights
        )


        # ----------------------------------------------------
        # Centroid features
        # ----------------------------------------------------

        mean_cx = safe_mean(
            centroid_x
        )


        mean_cy = safe_mean(
            centroid_y
        )


        mean_cx_norm = 0.0

        mean_cy_norm = 0.0


        if image_width > 0:

            mean_cx_norm = (
                mean_cx
                / image_width
            )


        if image_height > 0:

            mean_cy_norm = (
                mean_cy
                / image_height
            )


        # ----------------------------------------------------
        # Centroid spread
        # ----------------------------------------------------

        if centroid_x:

            variance_x = safe_mean(
                [
                    (
                        x - mean_cx
                    ) ** 2

                    for x in centroid_x
                ]
            )

            std_cx = math.sqrt(
                variance_x
            )

        else:

            std_cx = 0.0


        if centroid_y:

            variance_y = safe_mean(
                [
                    (
                        y - mean_cy
                    ) ** 2

                    for y in centroid_y
                ]
            )

            std_cy = math.sqrt(
                variance_y
            )

        else:

            std_cy = 0.0


        std_cx_norm = 0.0

        std_cy_norm = 0.0


        if image_width > 0:

            std_cx_norm = (
                std_cx
                / image_width
            )


        if image_height > 0:

            std_cy_norm = (
                std_cy
                / image_height
            )


        # ----------------------------------------------------
        # Overall crack bounding extent
        # ----------------------------------------------------

        if bbox_widths:

            min_x = min(
                [
                    polygon_bbox(
                        [
                            (
                                float(p[0]),
                                float(p[1])
                            )

                            for p in shape.get(
                                "points",
                                []
                            )
                        ]
                    )[0]

                    for shape in shapes

                    if str(
                        shape.get(
                            "label",
                            ""
                        )
                    ).strip().lower()
                    == "crack"

                    and len(
                        shape.get(
                            "points",
                            []
                        )
                    ) >= 3
                ]
            )

            max_x = max(
                [
                    polygon_bbox(
                        [
                            (
                                float(p[0]),
                                float(p[1])
                            )

                            for p in shape.get(
                                "points",
                                []
                            )
                        ]
                    )[2]

                    for shape in shapes

                    if str(
                        shape.get(
                            "label",
                            ""
                        )
                    ).strip().lower()
                    == "crack"

                    and len(
                        shape.get(
                            "points",
                            []
                        )
                    ) >= 3
                ]
            )

            min_y = min(
                [
                    polygon_bbox(
                        [
                            (
                                float(p[0]),
                                float(p[1])
                            )

                            for p in shape.get(
                                "points",
                                []
                            )
                        ]
                    )[1]

                    for shape in shapes

                    if str(
                        shape.get(
                            "label",
                            ""
                        )
                    ).strip().lower()
                    == "crack"

                    and len(
                        shape.get(
                            "points",
                            []
                        )
                    ) >= 3
                ]
            )

            max_y = max(
                [
                    polygon_bbox(
                        [
                            (
                                float(p[0]),
                                float(p[1])
                            )

                            for p in shape.get(
                                "points",
                                []
                            )
                        ]
                    )[3]

                    for shape in shapes

                    if str(
                        shape.get(
                            "label",
                            ""
                        )
                    ).strip().lower()
                    == "crack"

                    and len(
                        shape.get(
                            "points",
                            []
                        )
                    ) >= 3
                ]
            )


            crack_bbox_union_width = max(
                0.0,
                max_x - min_x
            )


            crack_bbox_union_height = max(
                0.0,
                max_y - min_y
            )

        else:

            crack_bbox_union_width = 0.0

            crack_bbox_union_height = 0.0


        # ----------------------------------------------------
        # Output row
        # ----------------------------------------------------

        output_row = {

            "record_id":
                index,

            "split":
                split,

            "component":
                component,

            "sequence":
                sequence,

            "original_image_filename":
                original_image_filename,

            "original_json_filename":
                original_json_filename,

            "canonical_image_filename":
                canonical_image_filename,

            "canonical_json_filename":
                canonical_json_filename,

            "source_image":
                source_image_value,

            "source_json":
                source_json_value,

            "image_width_px":
                image_width,

            "image_height_px":
                image_height,

            "image_area_px":
                image_area,

            "crack_count":
                crack_count,

            "total_crack_area_px":
                round_value(
                    total_crack_area
                ),

            "crack_area_ratio":
                round_value(
                    crack_area_ratio
                ),

            "largest_crack_area_px":
                round_value(
                    largest_crack_area
                ),

            "largest_crack_area_ratio":
                round_value(
                    largest_crack_area_ratio
                ),

            "smallest_crack_area_px":
                round_value(
                    smallest_crack_area
                ),

            "mean_crack_area_px":
                round_value(
                    mean_crack_area
                ),

            "median_crack_area_px":
                round_value(
                    median_crack_area
                ),

            "total_crack_perimeter_px":
                round_value(
                    total_perimeter
                ),

            "mean_crack_perimeter_px":
                round_value(
                    mean_perimeter
                ),

            "max_crack_perimeter_px":
                round_value(
                    max_perimeter
                ),

            "mean_crack_bbox_width_px":
                round_value(
                    mean_bbox_width
                ),

            "mean_crack_bbox_height_px":
                round_value(
                    mean_bbox_height
                ),

            "mean_crack_bbox_area_px":
                round_value(
                    mean_bbox_area
                ),

            "max_crack_bbox_width_px":
                round_value(
                    max_bbox_width
                ),

            "max_crack_bbox_height_px":
                round_value(
                    max_bbox_height
                ),

            "mean_crack_centroid_x_px":
                round_value(
                    mean_cx
                ),

            "mean_crack_centroid_y_px":
                round_value(
                    mean_cy
                ),

            "mean_crack_centroid_x_norm":
                round_value(
                    mean_cx_norm
                ),

            "mean_crack_centroid_y_norm":
                round_value(
                    mean_cy_norm
                ),

            "crack_centroid_x_std_px":
                round_value(
                    std_cx
                ),

            "crack_centroid_y_std_px":
                round_value(
                    std_cy
                ),

            "crack_centroid_x_std_norm":
                round_value(
                    std_cx_norm
                ),

            "crack_centroid_y_std_norm":
                round_value(
                    std_cy_norm
                ),

            "crack_bbox_union_width_px":
                round_value(
                    crack_bbox_union_width
                ),

            "crack_bbox_union_height_px":
                round_value(
                    crack_bbox_union_height
                ),

            "json_image_width_px":
                json_width
                if json_width is not None
                else "",

            "json_image_height_px":
                json_height
                if json_height is not None
                else "",

            "annotation_shape_count":
                annotation_shape_count,

            "valid_crack_polygon_count":
                valid_polygon_count,

            "invalid_shape_count":
                invalid_shape_count
        }


        feature_rows.append(
            output_row
        )


        # ----------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------

        split_counter[
            split
        ] += 1

        crack_count_values.append(
            crack_count
        )

        area_ratio_values.append(
            crack_area_ratio
        )

        total_area_values.append(
            total_crack_area
        )

        perimeter_values.append(
            total_perimeter
        )


        if invalid_shape_count > 0:

            summary[
                "invalid_shapes"
            ] += invalid_shape_count


        summary[
            "non_crack_shapes"
        ] += non_crack_shape_count


        summary[
            "processed_records"
        ] += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 500 == 0:

            print(
                "  Processed:",
                index,
                "/",
                len(rows)
            )


    except Exception as e:

        summary[
            "failed_records"
        ] += 1

        error_record = {

            "record_id":
                index,

            "split":
                row.get(
                    "split",
                    ""
                ),

            "sequence":
                row.get(
                    "sequence",
                    ""
                ),

            "source_image":
                row.get(
                    "source_image",
                    ""
                ),

            "error":
                str(e)
        }


        if len(
            summary[
                "errors"
            ]
        ) < 50:

            summary[
                "errors"
            ].append(
                error_record
            )


        print(
            "  ERROR at record",
            index,
            ":",
            str(e)
        )


# ============================================================
# WRITE FEATURE CSV
# ============================================================

print(
    "\nWriting feature dataset..."
)


with open(
    FEATURE_OUTPUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=output_columns
    )

    writer.writeheader()

    writer.writerows(
        feature_rows
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

summary[
    "split_counts"
] = dict(
    split_counter
)


summary[
    "crack_count"
] = {

    "mean":
        round_value(
            safe_mean(
                crack_count_values
            )
        ),

    "median":
        round_value(
            safe_median(
                crack_count_values
            )
        ),

    "min":
        int(
            safe_min(
                crack_count_values
            )
        ),

    "max":
        int(
            safe_max(
                crack_count_values
            )
        ),

    "p25":
        round_value(
            percentile(
                crack_count_values,
                0.25
            )
        ),

    "p75":
        round_value(
            percentile(
                crack_count_values,
                0.75
            )
        )
}


summary[
    "damage_area_ratio"
] = {

    "mean":
        round_value(
            safe_mean(
                area_ratio_values
            )
        ),

    "median":
        round_value(
            safe_median(
                area_ratio_values
            )
        ),

    "min":
        round_value(
            safe_min(
                area_ratio_values
            )
        ),

    "max":
        round_value(
            safe_max(
                area_ratio_values
            )
        ),

    "p25":
        round_value(
            percentile(
                area_ratio_values,
                0.25
            )
        ),

    "p75":
        round_value(
            percentile(
                area_ratio_values,
                0.75
            )
        )
}


summary[
    "total_crack_area_px"
] = {

    "mean":
        round_value(
            safe_mean(
                total_area_values
            )
        ),

    "median":
        round_value(
            safe_median(
                total_area_values
            )
        ),

    "max":
        round_value(
            safe_max(
                total_area_values
            )
        )
}


summary[
    "total_crack_perimeter_px"
] = {

    "mean":
        round_value(
            safe_mean(
                perimeter_values
            )
        ),

    "median":
        round_value(
            safe_median(
                perimeter_values
            )
        ),

    "max":
        round_value(
            safe_max(
                perimeter_values
            )
        )
}


# ============================================================
# FINAL STATUS
# ============================================================

if (
    summary[
        "processed_records"
    ] == EXPECTED_TOTAL

    and

    summary[
        "failed_records"
    ] == 0
):

    summary[
        "status"
    ] = "PASS"

else:

    summary[
        "status"
    ] = "FAIL"


summary[
    "feature_columns"
] = output_columns


summary[
    "feature_row_count"
] = len(
    feature_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

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
# FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 1.0 SUMMARY"
)

print(
    "=" * 70
)


print(
    "Manifest records:",
    len(rows)
)

print(
    "Processed records:",
    summary[
        "processed_records"
    ]
)

print(
    "Failed records:",
    summary[
        "failed_records"
    ]
)

print(
    "Missing images:",
    summary[
        "missing_images"
    ]
)

print(
    "Missing JSON:",
    summary[
        "missing_json"
    ]
)

print(
    "Invalid JSON:",
    summary[
        "invalid_json"
    ]
)

print(
    "Invalid shapes:",
    summary[
        "invalid_shapes"
    ]
)

print(
    "Feature rows:",
    len(feature_rows)
)


print(
    "\nSplit counts:"
)

for split in [
    "train",
    "val",
    "test"
]:

    print(
        "  ",
        split.upper(),
        ":",
        split_counter.get(
            split,
            0
        )
    )


print(
    "\nCrack count statistics:"
)

print(
    "  Mean:",
    summary[
        "crack_count"
    ][
        "mean"
    ]
)

print(
    "  Median:",
    summary[
        "crack_count"
    ][
        "median"
    ]
)

print(
    "  Min:",
    summary[
        "crack_count"
    ][
        "min"
    ]
)

print(
    "  Max:",
    summary[
        "crack_count"
    ][
        "max"
    ]
)


print(
    "\nDamage area ratio:"
)

print(
    "  Mean:",
    summary[
        "damage_area_ratio"
    ][
        "mean"
    ]
)

print(
    "  Median:",
    summary[
        "damage_area_ratio"
    ][
        "median"
    ]
)

print(
    "  Max:",
    summary[
        "damage_area_ratio"
    ][
        "max"
    ]
)


print(
    "\nOutput feature file:"
)

print(
    FEATURE_OUTPUT
)


print(
    "\nOutput summary:"
)

print(
    SUMMARY_OUTPUT
)


print(
    "\nFinal status:",
    summary[
        "status"
    ]
)


print(
    "=" * 70
)


if summary[
    "status"
] == "PASS":

    print(
        "STAGE 1.0 COMPLETE - FEATURE EXTRACTION PASSED"
    )

else:

    print(
        "STAGE 1.0 FAILED - DO NOT PROCEED"
    )


print(
    "=" * 70
)