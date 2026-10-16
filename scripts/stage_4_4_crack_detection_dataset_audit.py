import os
import json
import re
import numpy as np
import pandas as pd


# ============================================================
# STAGE 4.4
# CRACK DETECTION / SEGMENTATION DATASET AUDIT
#
# IMPORTANT:
# The canonical manifest is used only for verified identity
# and source-path fields.
#
# Image dimensions are read directly from the actual LabelMe
# JSON annotations.
#
# No assumed manifest dimension columns are used.
# ============================================================


PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

DATASET_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd"
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs",
    "stage_4_4_crack_detection_audit"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_4_crack_detection_audit.json"
)

RECORDS_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_4_crack_annotation_records.csv"
)

SPLIT_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_4_split_summary.csv"
)

SEQUENCE_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_4_4_sequence_summary.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# None = audit the complete canonical dataset.
SAMPLE_LIMIT = None


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

    return int(
        numbers[-1]
    )


def polygon_area(points):

    if len(points) < 3:
        return 0.0

    pts = np.asarray(
        points,
        dtype=np.float64
    )

    if pts.ndim != 2:
        return 0.0

    if pts.shape[1] != 2:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    return float(
        0.5
        *
        abs(
            np.dot(
                x,
                np.roll(y, 1)
            )
            -
            np.dot(
                y,
                np.roll(x, 1)
            )
        )
    )


def polygon_perimeter(points):

    if len(points) < 2:
        return 0.0

    pts = np.asarray(
        points,
        dtype=np.float64
    )

    shifted = np.roll(
        pts,
        -1,
        axis=0
    )

    distances = np.sqrt(
        np.sum(
            (
                pts
                -
                shifted
            ) ** 2,
            axis=1
        )
    )

    return float(
        distances.sum()
    )


def polygon_bbox(points):

    pts = np.asarray(
        points,
        dtype=np.float64
    )

    return (
        float(
            np.min(
                pts[:, 0]
            )
        ),
        float(
            np.min(
                pts[:, 1]
            )
        ),
        float(
            np.max(
                pts[:, 0]
            )
        ),
        float(
            np.max(
                pts[:, 1]
            )
        )
    )


def polygon_centroid(points):

    pts = np.asarray(
        points,
        dtype=np.float64
    )

    return (
        float(
            np.mean(
                pts[:, 0]
            )
        ),
        float(
            np.mean(
                pts[:, 1]
            )
        )
    )


def resolve_json_path(row):

    # --------------------------------------------------------
    # FIRST: use the verified source_json from the manifest.
    # --------------------------------------------------------

    source_json = str(
        row[
            "source_json"
        ]
    )

    if (
        source_json
        and
        source_json.lower()
        not in [
            "nan",
            "none"
        ]
        and
        os.path.exists(
            source_json
        )
    ):

        return os.path.abspath(
            source_json
        )


    # --------------------------------------------------------
    # SECOND: reconstruct ONLY from verified manifest fields.
    #
    # Verified structure:
    #
    # BeltCrack9kd1/
    #   json/
    #     split/
    #       dataN/
    #         filename.json
    # --------------------------------------------------------

    sequence = str(
        row[
            "sequence"
        ]
    )

    sequence_parts = sequence.split(
        "/"
    )

    if len(sequence_parts) != 2:

        return None

    split = sequence_parts[0]
    data_folder = sequence_parts[1]

    original_json_filename = str(
        row[
            "original_json_filename"
        ]
    )

    # Package 1 is the canonical annotation authority
    # established during Stage 0.8/0.9.
    candidate = os.path.join(
        DATASET_ROOT,
        "BeltCrack9kd1",
        "json",
        split,
        data_folder,
        original_json_filename
    )

    if os.path.exists(
        candidate
    ):

        return os.path.abspath(
            candidate
        )

    return None


def validate_json_structure(data):

    if not isinstance(
        data,
        dict
    ):

        return False

    if "shapes" not in data:

        return False

    if not isinstance(
        data[
            "shapes"
        ],
        list
    ):

        return False

    return True


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("STAGE 4.4 - CRACK DETECTION DATASET AUDIT")
print("=" * 80)

print()
print(
    "Purpose:"
)

print(
    "Audit the verified LabelMe crack polygons before"
)

print(
    "building an image-level crack segmentation model."
)

print()
print(
    "No ML model will be trained in this stage."
)


# ============================================================
# CHECK MANIFEST
# ============================================================

print()
print("=" * 80)
print("CHECKING CANONICAL MANIFEST")
print("=" * 80)


if not os.path.exists(
    MANIFEST_PATH
):

    raise FileNotFoundError(
        "Canonical manifest not found:\n"
        +
        MANIFEST_PATH
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)


print(
    "Manifest records:",
    len(manifest)
)


# These are the fields already established by the
# canonical-dataset construction stage.

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


missing_columns = [

    column
    for column in required_columns
    if column not in manifest.columns

]


print(
    "Verified manifest columns:",
    len(manifest.columns)
)


if missing_columns:

    raise ValueError(
        "Canonical manifest is missing verified "
        "required fields:\n"
        +
        str(
            missing_columns
        )
    )


print(
    "Manifest schema check: PASS"
)


# ============================================================
# SELECT RECORDS
# ============================================================

if SAMPLE_LIMIT is None:

    audit_df = manifest.copy()

else:

    audit_df = manifest.head(
        SAMPLE_LIMIT
    ).copy()


print(
    "Records selected for audit:",
    len(audit_df)
)


# ============================================================
# AUDIT COUNTERS
# ============================================================

records = []


json_missing = 0
json_invalid = 0
json_read_errors = 0

dimension_missing = 0
dimension_invalid = 0

total_shapes = 0
total_crack_shapes = 0

valid_crack_polygons = 0
invalid_crack_shapes = 0

zero_area_cracks = 0
tiny_cracks = 0

nonfinite_points = 0
out_of_bounds_points = 0

images_with_cracks = 0
images_without_cracks = 0

negative_area_records = 0


# ============================================================
# PROCESS JSON ANNOTATIONS
# ============================================================

print()
print("=" * 80)
print("AUDITING ACTUAL CRACK ANNOTATIONS")
print("=" * 80)


for index, row in audit_df.iterrows():

    json_path = resolve_json_path(
        row
    )


    # --------------------------------------------------------
    # JSON PATH
    # --------------------------------------------------------

    if json_path is None:

        json_missing += 1

        continue


    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:

        json_read_errors += 1

        continue


    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    if not validate_json_structure(
        data
    ):

        json_invalid += 1

        continue


    # --------------------------------------------------------
    # DIMENSIONS
    #
    # Read from the actual JSON.
    # --------------------------------------------------------

    image_width = data.get(
        "imageWidth"
    )

    image_height = data.get(
        "imageHeight"
    )


    if (
        image_width is None
        or
        image_height is None
    ):

        dimension_missing += 1

        continue


    try:

        image_width = float(
            image_width
        )

        image_height = float(
            image_height
        )

    except Exception:

        dimension_invalid += 1

        continue


    if (
        image_width <= 0
        or
        image_height <= 0
    ):

        dimension_invalid += 1

        continue


    image_area = (
        image_width
        *
        image_height
    )


    # --------------------------------------------------------
    # SHAPES
    # --------------------------------------------------------

    shapes = data.get(
        "shapes",
        []
    )


    total_shapes += len(
        shapes
    )


    crack_count = 0
    valid_crack_count = 0

    total_area = 0.0
    total_perimeter = 0.0

    largest_area = 0.0
    smallest_area = np.inf

    largest_bbox_area = 0.0

    centroid_x_values = []
    centroid_y_values = []


    # --------------------------------------------------------
    # PROCESS SHAPES
    # --------------------------------------------------------

    for shape in shapes:

        label = str(
            shape.get(
                "label",
                ""
            )
        ).lower().strip()


        if label != "crack":

            continue


        total_crack_shapes += 1
        crack_count += 1


        points = shape.get(
            "points",
            []
        )


        if not isinstance(
            points,
            list
        ):

            invalid_crack_shapes += 1

            continue


        if len(points) < 3:

            invalid_crack_shapes += 1

            continue


        try:

            pts = np.asarray(
                points,
                dtype=np.float64
            )

        except Exception:

            invalid_crack_shapes += 1

            continue


        if (
            pts.ndim != 2
            or
            pts.shape[1] != 2
        ):

            invalid_crack_shapes += 1

            continue


        # ----------------------------------------------------
        # FINITE CHECK
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(
                pts
            )
        ):

            nonfinite_points += 1

            invalid_crack_shapes += 1

            continue


        # ----------------------------------------------------
        # IMAGE BOUNDARY CHECK
        #
        # We do NOT alter the annotation.
        # We only measure whether points are outside.
        # ----------------------------------------------------

        outside = (

            (pts[:, 0] < 0)
            |
            (pts[:, 0] > image_width)
            |
            (pts[:, 1] < 0)
            |
            (pts[:, 1] > image_height)

        )


        if np.any(
            outside
        ):

            out_of_bounds_points += 1


        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        area = polygon_area(
            pts
        )


        perimeter = polygon_perimeter(
            pts
        )


        if area < 0:

            negative_area_records += 1


        if area <= 0:

            zero_area_cracks += 1

            invalid_crack_shapes += 1

            continue


        min_x, min_y, max_x, max_y = (
            polygon_bbox(
                pts
            )
        )


        bbox_width = (
            max_x
            -
            min_x
        )

        bbox_height = (
            max_y
            -
            min_y
        )


        bbox_area = (
            bbox_width
            *
            bbox_height
        )


        cx, cy = polygon_centroid(
            pts
        )


        if area < 10.0:

            tiny_cracks += 1


        valid_crack_count += 1

        valid_crack_polygons += 1

        total_area += area

        total_perimeter += perimeter

        largest_area = max(
            largest_area,
            area
        )

        smallest_area = min(
            smallest_area,
            area
        )

        largest_bbox_area = max(
            largest_bbox_area,
            bbox_area
        )

        centroid_x_values.append(
            cx
        )

        centroid_y_values.append(
            cy
        )


    # --------------------------------------------------------
    # IMAGE-LEVEL COUNTS
    # --------------------------------------------------------

    if valid_crack_count > 0:

        images_with_cracks += 1

    else:

        images_without_cracks += 1


    # --------------------------------------------------------
    # EMPTY CRACK CASE
    # --------------------------------------------------------

    if smallest_area == np.inf:

        smallest_area = 0.0


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    if len(
        centroid_x_values
    ) > 0:

        mean_centroid_x = float(
            np.mean(
                centroid_x_values
            )
        )

        mean_centroid_y = float(
            np.mean(
                centroid_y_values
            )
        )

    else:

        mean_centroid_x = np.nan
        mean_centroid_y = np.nan


    # --------------------------------------------------------
    # DAMAGE AREA RATIO
    # --------------------------------------------------------

    crack_area_ratio = (
        total_area
        /
        image_area
    )


    # --------------------------------------------------------
    # FRAME
    # --------------------------------------------------------

    frame = extract_frame(
        row[
            "canonical_image_filename"
        ]
    )


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    records.append(
        {
            "split":
                str(
                    row[
                        "split"
                    ]
                ),

            "component":
                str(
                    row[
                        "component"
                    ]
                ),

            "sequence":
                str(
                    row[
                        "sequence"
                    ]
                ),

            "frame":
                frame,

            "original_image_filename":
                str(
                    row[
                        "original_image_filename"
                    ]
                ),

            "source_image":
                str(
                    row[
                        "source_image"
                    ]
                ),

            "source_json":
                json_path,

            "image_width_px":
                image_width,

            "image_height_px":
                image_height,

            "image_area_px":
                image_area,

            "shape_count":
                int(
                    len(shapes)
                ),

            "crack_shape_count":
                int(
                    crack_count
                ),

            "valid_crack_polygon_count":
                int(
                    valid_crack_count
                ),

            "crack_area_px":
                float(
                    total_area
                ),

            "crack_area_ratio":
                float(
                    crack_area_ratio
                ),

            "largest_crack_area_px":
                float(
                    largest_area
                ),

            "smallest_crack_area_px":
                float(
                    smallest_area
                ),

            "total_crack_perimeter_px":
                float(
                    total_perimeter
                ),

            "largest_bbox_area_px":
                float(
                    largest_bbox_area
                ),

            "mean_centroid_x_px":
                mean_centroid_x,

            "mean_centroid_y_px":
                mean_centroid_y
        }
    )


    processed = (
        index + 1
    )


    if (
        processed == 1
        or
        processed % 1000 == 0
        or
        processed == len(audit_df)
    ):

        print(
            "Processed:",
            processed,
            "/",
            len(audit_df)
        )


# ============================================================
# BUILD OUTPUT DATAFRAME
# ============================================================

records_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE RECORDS
# ============================================================

records_df.to_csv(
    RECORDS_PATH,
    index=False
)


# ============================================================
# BASIC RESULTS
# ============================================================

print()
print("=" * 80)
print("ANNOTATION AUDIT RESULTS")
print("=" * 80)


print(
    "Manifest records selected:",
    len(audit_df)
)

print(
    "Records successfully analyzed:",
    len(records_df)
)

print(
    "JSON missing:",
    json_missing
)

print(
    "JSON invalid:",
    json_invalid
)

print(
    "JSON read errors:",
    json_read_errors
)

print(
    "Missing dimensions:",
    dimension_missing
)

print(
    "Invalid dimensions:",
    dimension_invalid
)

print(
    "Total shapes:",
    total_shapes
)

print(
    "Total crack shapes:",
    total_crack_shapes
)

print(
    "Valid crack polygons:",
    valid_crack_polygons
)

print(
    "Invalid crack shapes:",
    invalid_crack_shapes
)

print(
    "Zero-area cracks:",
    zero_area_cracks
)

print(
    "Tiny cracks (<10 px2):",
    tiny_cracks
)

print(
    "Non-finite point records:",
    nonfinite_points
)

print(
    "Out-of-bounds point records:",
    out_of_bounds_points
)

print(
    "Images with valid cracks:",
    images_with_cracks
)

print(
    "Images without valid cracks:",
    images_without_cracks
)


# ============================================================
# AREA STATISTICS
# ============================================================

print()
print("=" * 80)
print("CRACK AREA RATIO")
print("=" * 80)


if len(records_df) > 0:

    area_ratio = records_df[
        "crack_area_ratio"
    ].dropna()


    if len(area_ratio) > 0:

        print(
            "Mean:",
            f"{area_ratio.mean():.8f}"
        )

        print(
            "Median:",
            f"{area_ratio.median():.8f}"
        )

        print(
            "P90:",
            f"{area_ratio.quantile(0.90):.8f}"
        )

        print(
            "P95:",
            f"{area_ratio.quantile(0.95):.8f}"
        )

        print(
            "P99:",
            f"{area_ratio.quantile(0.99):.8f}"
        )

        print(
            "Maximum:",
            f"{area_ratio.max():.8f}"
        )


# ============================================================
# POLYGON STATISTICS
# ============================================================

print()
print("=" * 80)
print("CRACK POLYGONS PER IMAGE")
print("=" * 80)


if len(records_df) > 0:

    print(
        records_df[
            "valid_crack_polygon_count"
        ].describe().to_string()
    )


# ============================================================
# SPLIT ANALYSIS
# ============================================================

print()
print("=" * 80)
print("SPLIT ANALYSIS")
print("=" * 80)


if len(records_df) > 0:

    split_summary = (
        records_df
        .groupby(
            "split"
        )
        .agg(
            records=(
                "sequence",
                "count"
            ),

            mean_cracks=(
                "valid_crack_polygon_count",
                "mean"
            ),

            median_cracks=(
                "valid_crack_polygon_count",
                "median"
            ),

            mean_area_ratio=(
                "crack_area_ratio",
                "mean"
            ),

            median_area_ratio=(
                "crack_area_ratio",
                "median"
            ),

            p95_area_ratio=(
                "crack_area_ratio",
                lambda x:
                x.quantile(
                    0.95
                )
            ),

            max_area_ratio=(
                "crack_area_ratio",
                "max"
            )
        )
        .reset_index()
    )

else:

    split_summary = pd.DataFrame()


if len(split_summary) > 0:

    print(
        split_summary.to_string(
            index=False
        )
    )


split_summary.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False
)


# ============================================================
# SEQUENCE ANALYSIS
# ============================================================

print()
print("=" * 80)
print("SEQUENCE COVERAGE")
print("=" * 80)


if len(records_df) > 0:

    sequence_summary = (
        records_df
        .groupby(
            "sequence"
        )
        .agg(
            records=(
                "frame",
                "count"
            ),

            mean_cracks=(
                "valid_crack_polygon_count",
                "mean"
            ),

            mean_area_ratio=(
                "crack_area_ratio",
                "mean"
            ),

            max_area_ratio=(
                "crack_area_ratio",
                "max"
            )
        )
        .reset_index()
        .sort_values(
            "sequence"
        )
    )

else:

    sequence_summary = pd.DataFrame()


print(
    sequence_summary.to_string(
        index=False
    )
)


sequence_summary.to_csv(
    SEQUENCE_SUMMARY_PATH,
    index=False
)


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

print()
print("=" * 80)
print("IMAGE DIMENSIONS FROM ACTUAL JSON")
print("=" * 80)


if len(records_df) > 0:

    dimension_summary = (
        records_df
        .groupby(
            [
                "image_width_px",
                "image_height_px"
            ]
        )
        .size()
        .reset_index(
            name="records"
        )
        .sort_values(
            "records",
            ascending=False
        )
    )


    print(
        dimension_summary.to_string(
            index=False
        )
    )


# ============================================================
# TRAINING SUITABILITY
# ============================================================

print()
print("=" * 80)
print("SEGMENTATION DATASET SUITABILITY")
print("=" * 80)


checks = {}


checks[
    "manifest_records_present"
] = (
    len(audit_df) > 0
)


checks[
    "all_records_analyzed"
] = (
    len(records_df)
    ==
    len(audit_df)
)


checks[
    "json_missing_zero"
] = (
    json_missing == 0
)


checks[
    "json_invalid_zero"
] = (
    json_invalid == 0
)


checks[
    "json_read_errors_zero"
] = (
    json_read_errors == 0
)


checks[
    "dimensions_available"
] = (
    dimension_missing == 0
    and
    dimension_invalid == 0
)


checks[
    "crack_annotations_present"
] = (
    total_crack_shapes > 0
)


checks[
    "valid_crack_polygons_present"
] = (
    valid_crack_polygons > 0
)


checks[
    "finite_geometry"
] = (
    nonfinite_points == 0
)


for name, value in checks.items():

    print(
        name + ":",
        "PASS"
        if value
        else "REVIEW"
    )


# ============================================================
# IMPORTANT:
# OUT-OF-BOUNDS IS REPORTED, NOT AUTOMATICALLY FAILED
# ============================================================

print()

print(
    "Out-of-bounds polygon-point records:",
    out_of_bounds_points
)

print(
    "This is reported separately and does not"
)

print(
    "automatically invalidate the dataset."
)


# ============================================================
# FINAL DECISION
# ============================================================

core_ready = all(
    checks.values()
)


if core_ready:

    decision = (
        "SEGMENTATION_DATASET_READY_FOR_MASK_GENERATION"
    )

else:

    decision = (
        "SEGMENTATION_DATASET_REQUIRES_REVIEW"
    )


print()
print(
    "Decision:",
    decision
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "4.4",

    "status":
        "PASS"
        if core_ready
        else "REVIEW",

    "decision":
        decision,

    "manifest_records":
        int(len(manifest)),

    "records_selected":
        int(len(audit_df)),

    "records_analyzed":
        int(len(records_df)),

    "json_missing":
        int(json_missing),

    "json_invalid":
        int(json_invalid),

    "json_read_errors":
        int(json_read_errors),

    "dimension_missing":
        int(dimension_missing),

    "dimension_invalid":
        int(dimension_invalid),

    "total_shapes":
        int(total_shapes),

    "total_crack_shapes":
        int(total_crack_shapes),

    "valid_crack_polygons":
        int(valid_crack_polygons),

    "invalid_crack_shapes":
        int(invalid_crack_shapes),

    "zero_area_cracks":
        int(zero_area_cracks),

    "tiny_cracks_under_10px2":
        int(tiny_cracks),

    "nonfinite_points":
        int(nonfinite_points),

    "out_of_bounds_points":
        int(out_of_bounds_points),

    "images_with_valid_cracks":
        int(images_with_cracks),

    "images_without_valid_cracks":
        int(images_without_cracks),

    "checks":
        {
            str(k):
                bool(v)
            for k, v
            in checks.items()
        },

    "outputs":
        {
            "annotation_records":
                RECORDS_PATH,

            "split_summary":
                SPLIT_SUMMARY_PATH,

            "sequence_summary":
                SEQUENCE_SUMMARY_PATH,

            "summary":
                SUMMARY_PATH
        },

    "method_note":
        (
            "Image dimensions were read directly from "
            "the actual LabelMe JSON files. The canonical "
            "manifest was not assumed to contain dimension "
            "columns."
        ),

    "next_stage_note":
        (
            "If the core checks pass, the next stage should "
            "generate segmentation masks from the verified "
            "LabelMe polygons. No segmentation model is "
            "trained in Stage 4.4."
        )
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

if core_ready:

    print(
        "STAGE 4.4 STATUS: PASS"
    )

else:

    print(
        "STAGE 4.4 STATUS: REVIEW"
    )

print("=" * 80)

print()
print(
    "Annotation records:",
    RECORDS_PATH
)

print(
    "Split summary:",
    SPLIT_SUMMARY_PATH
)

print(
    "Sequence summary:",
    SEQUENCE_SUMMARY_PATH
)

print(
    "Audit summary:",
    SUMMARY_PATH
)