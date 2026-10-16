import os
import json
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

RAW_DATASET_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd"

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

RAPID_CASES = os.path.join(
    AUDIT_DIR,
    "stage_2_6_rapid_damage_cases.csv"
)

MANIFEST = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_DIR = os.path.join(
    AUDIT_DIR,
    "stage_2_7_extreme_transition_validation"
)

OUTPUT_CASES = os.path.join(
    AUDIT_DIR,
    "stage_2_7_extreme_transition_validation.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_2_7_extreme_transition_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_N = 30

IMAGE_MAX_WIDTH = 900
IMAGE_MAX_HEIGHT = 600

POLYGON_WIDTH = 3


# ============================================================
# HELPERS
# ============================================================

def resolve_dataset_path(path_value):

    if path_value is None:
        return None

    path_value = str(
        path_value
    ).strip()

    if not path_value:
        return None

    # Already absolute.
    if os.path.isabs(path_value):

        return os.path.normpath(
            path_value
        )

    # Convert slash style.
    normalized = path_value.replace(
        "/",
        os.sep
    )

    # Source paths in the manifest look like:
    # BeltCrack9kd1/images/train/data2/213.jpg
    #
    # They are relative to the raw dataset root.
    candidate = os.path.join(
        RAW_DATASET_ROOT,
        normalized
    )

    return os.path.normpath(
        candidate
    )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def polygon_area(points):

    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(
        len(points)
    ):

        x1, y1 = points[i]

        x2, y2 = points[
            (i + 1)
            %
            len(points)
        ]

        area += (
            x1 * y2
            -
            x2 * y1
        )

    return abs(area) / 2.0


def extract_crack_shapes(
    annotation
):

    shapes = annotation.get(
        "shapes",
        []
    )

    crack_shapes = []

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

        clean_points = []

        for point in points:

            if (
                isinstance(point, list)
                and
                len(point) >= 2
            ):

                try:

                    x = float(
                        point[0]
                    )

                    y = float(
                        point[1]
                    )

                    clean_points.append(
                        [x, y]
                    )

                except Exception:
                    pass

        if len(clean_points) >= 3:

            crack_shapes.append(
                clean_points
            )

    return crack_shapes


def calculate_annotation_statistics(
    annotation
):

    crack_shapes = extract_crack_shapes(
        annotation
    )

    areas = []

    for points in crack_shapes:

        areas.append(
            polygon_area(
                points
            )
        )

    if not areas:

        return {
            "shape_count": 0,
            "total_area": 0.0,
            "largest_area": 0.0,
            "smallest_area": 0.0
        }

    return {
        "shape_count":
            len(areas),

        "total_area":
            float(
                sum(areas)
            ),

        "largest_area":
            float(
                max(areas)
            ),

        "smallest_area":
            float(
                min(areas)
            )
    }


def image_resize_info(
    width,
    height
):

    scale = min(
        IMAGE_MAX_WIDTH / width,
        IMAGE_MAX_HEIGHT / height,
        1.0
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        )
    )

    return (
        new_width,
        new_height,
        scale
    )


def draw_annotation(
    image,
    annotation
):

    crack_shapes = extract_crack_shapes(
        annotation
    )

    width, height = image.size

    new_width, new_height, scale = (
        image_resize_info(
            width,
            height
        )
    )

    if (
        new_width != width
        or
        new_height != height
    ):

        image = image.resize(
            (
                new_width,
                new_height
            )
        )

    draw = ImageDraw.Draw(
        image
    )

    for points in crack_shapes:

        scaled = []

        for x, y in points:

            scaled.append(
                (
                    int(
                        round(
                            x * scale
                        )
                    ),
                    int(
                        round(
                            y * scale
                        )
                    )
                )
            )

        if len(scaled) >= 3:

            draw.line(
                scaled
                +
                [scaled[0]],
                width=POLYGON_WIDTH
            )

    return image


def make_side_by_side(
    current_image,
    future_image
):

    width1, height1 = (
        current_image.size
    )

    width2, height2 = (
        future_image.size
    )

    target_height = max(
        height1,
        height2
    )

    if height1 != target_height:

        scale1 = (
            target_height
            /
            height1
        )

        current_image = (
            current_image.resize(
                (
                    int(
                        width1
                        *
                        scale1
                    ),
                    target_height
                )
            )
        )

    if height2 != target_height:

        scale2 = (
            target_height
            /
            height2
        )

        future_image = (
            future_image.resize(
                (
                    int(
                        width2
                        *
                        scale2
                    ),
                    target_height
                )
            )
        )

    combined = Image.new(
        "RGB",
        (
            current_image.width
            +
            future_image.width,
            target_height
        )
    )

    combined.paste(
        current_image,
        (
            0,
            0
        )
    )

    combined.paste(
        future_image,
        (
            current_image.width,
            0
        )
    )

    return combined


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 2.7 - EXTREME TRANSITION "
    "VISUAL AND ANNOTATION VALIDATION"
)
print("=" * 70)


# ============================================================
# VERIFY RAW DATASET
# ============================================================

print()
print(
    "Checking raw dataset root..."
)

if not os.path.isdir(
    RAW_DATASET_ROOT
):

    raise FileNotFoundError(
        "Raw dataset root not found:\n"
        +
        RAW_DATASET_ROOT
    )

print(
    f"Raw dataset: FOUND"
)

print(
    RAW_DATASET_ROOT
)


# ============================================================
# LOAD RAPID CASES
# ============================================================

print()
print(
    "Loading Stage 2.6 rapid-damage cases..."
)

rapid = pd.read_csv(
    RAPID_CASES
)

print(
    f"Rapid case records: "
    f"{len(rapid)}"
)


# ============================================================
# SELECT TOP EVENTS
# ============================================================

rapid = rapid.sort_values(
    "absolute_damage_change",
    ascending=False
).head(
    TOP_N
).copy()

print(
    f"Events selected for validation: "
    f"{len(rapid)}"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

print()
print(
    "Loading canonical manifest..."
)

manifest = pd.read_csv(
    MANIFEST
)

print(
    f"Manifest records: "
    f"{len(manifest)}"
)


# ============================================================
# NORMALIZE
# ============================================================

rapid["sequence"] = (
    rapid["sequence"].astype(str)
)

rapid["component"] = (
    rapid["component"].astype(int)
)

rapid["frame_number"] = (
    rapid["frame_number"].astype(int)
)

manifest["sequence"] = (
    manifest["sequence"].astype(str)
)

manifest["component"] = (
    manifest["component"].astype(int)
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VALIDATION
# ============================================================

results = []

print()
print("=" * 70)
print(
    "VALIDATING EXTREME TRANSITIONS"
)
print("=" * 70)


for rank, (_, event) in enumerate(
    rapid.iterrows(),
    start=1
):

    sequence = str(
        event["sequence"]
    )

    component = int(
        event["component"]
    )

    frame_number = int(
        event["frame_number"]
    )

    current_frame = frame_number

    future_frame = (
        frame_number + 1
    )


    # --------------------------------------------------------
    # FIND CURRENT RECORD
    # --------------------------------------------------------

    current_matches = manifest[
        (
            manifest["sequence"]
            ==
            sequence
        )
        &
        (
            manifest["component"]
            ==
            component
        )
        &
        (
            manifest[
                "original_image_filename"
            ]
            .astype(str)
            .str.replace(
                ".jpg",
                "",
                regex=False
            )
            .str.replace(
                ".jpeg",
                "",
                regex=False
            )
            .str.replace(
                ".JPG",
                "",
                regex=False
            )
            ==
            str(current_frame)
        )
    ]


    # --------------------------------------------------------
    # FIND FUTURE RECORD
    # --------------------------------------------------------

    future_matches = manifest[
        (
            manifest["sequence"]
            ==
            sequence
        )
        &
        (
            manifest["component"]
            ==
            component
        )
        &
        (
            manifest[
                "original_image_filename"
            ]
            .astype(str)
            .str.replace(
                ".jpg",
                "",
                regex=False
            )
            .str.replace(
                ".jpeg",
                "",
                regex=False
            )
            .str.replace(
                ".JPG",
                "",
                regex=False
            )
            ==
            str(future_frame)
        )
    ]


    if len(current_matches) != 1:

        print(
            f"WARNING: current manifest record "
            f"not found: {sequence} "
            f"frame {current_frame}"
        )

        continue


    if len(future_matches) != 1:

        print(
            f"WARNING: future manifest record "
            f"not found: {sequence} "
            f"frame {future_frame}"
        )

        continue


    current_row = (
        current_matches.iloc[0]
    )

    future_row = (
        future_matches.iloc[0]
    )


    # --------------------------------------------------------
    # RESOLVE PATHS
    # --------------------------------------------------------

    current_image_path = (
        resolve_dataset_path(
            current_row[
                "source_image"
            ]
        )
    )

    future_image_path = (
        resolve_dataset_path(
            future_row[
                "source_image"
            ]
        )
    )

    current_json_path = (
        resolve_dataset_path(
            current_row[
                "source_json"
            ]
        )
    )

    future_json_path = (
        resolve_dataset_path(
            future_row[
                "source_json"
            ]
        )
    )


    # --------------------------------------------------------
    # CHECK PATHS
    # --------------------------------------------------------

    if not os.path.exists(
        current_image_path
    ):

        print(
            "WARNING: missing current image:"
        )

        print(
            current_image_path
        )

        continue


    if not os.path.exists(
        future_image_path
    ):

        print(
            "WARNING: missing future image:"
        )

        print(
            future_image_path
        )

        continue


    if not os.path.exists(
        current_json_path
    ):

        print(
            "WARNING: missing current JSON:"
        )

        print(
            current_json_path
        )

        continue


    if not os.path.exists(
        future_json_path
    ):

        print(
            "WARNING: missing future JSON:"
        )

        print(
            future_json_path
        )

        continue


    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    try:

        current_annotation = (
            load_json(
                current_json_path
            )
        )

        future_annotation = (
            load_json(
                future_json_path
            )
        )

    except Exception as exc:

        print(
            f"WARNING: JSON error: {exc}"
        )

        continue


    # --------------------------------------------------------
    # ANNOTATION STATISTICS
    # --------------------------------------------------------

    current_stats = (
        calculate_annotation_statistics(
            current_annotation
        )
    )

    future_stats = (
        calculate_annotation_statistics(
            future_annotation
        )
    )


    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    try:

        current_image = (
            Image.open(
                current_image_path
            ).convert(
                "RGB"
            )
        )

        future_image = (
            Image.open(
                future_image_path
            ).convert(
                "RGB"
            )
        )

    except Exception as exc:

        print(
            f"WARNING: image error: {exc}"
        )

        continue


    # --------------------------------------------------------
    # DRAW ANNOTATIONS
    # --------------------------------------------------------

    current_overlay = (
        draw_annotation(
            current_image,
            current_annotation
        )
    )

    future_overlay = (
        draw_annotation(
            future_image,
            future_annotation
        )
    )


    # --------------------------------------------------------
    # SIDE BY SIDE
    # --------------------------------------------------------

    combined = make_side_by_side(
        current_overlay,
        future_overlay
    )


    # --------------------------------------------------------
    # ADD LABELS
    # --------------------------------------------------------

    label_height = 50

    labeled = Image.new(
        "RGB",
        (
            combined.width,
            combined.height
            +
            label_height
        ),
        "white"
    )

    labeled.paste(
        combined,
        (
            0,
            label_height
        )
    )

    draw = ImageDraw.Draw(
        labeled
    )

    draw.text(
        (
            10,
            15
        ),
        (
            f"CURRENT frame {current_frame}"
        ),
        fill="black"
    )

    draw.text(
        (
            current_overlay.width
            +
            10,
            15
        ),
        (
            f"FUTURE frame {future_frame}"
        ),
        fill="black"
    )


    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    safe_sequence = (
        sequence
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
    )

    output_name = (
        f"{rank:04d}_"
        f"{safe_sequence}_"
        f"frame_{current_frame}_"
        f"to_{future_frame}.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    labeled.save(
        output_path,
        quality=92
    )


    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    current_width = int(
        current_annotation.get(
            "imageWidth",
            current_image.width
        )
    )

    current_height = int(
        current_annotation.get(
            "imageHeight",
            current_image.height
        )
    )

    future_width = int(
        future_annotation.get(
            "imageWidth",
            future_image.width
        )
    )

    future_height = int(
        future_annotation.get(
            "imageHeight",
            future_image.height
        )
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

        current_annotation_ratio = (
            current_stats[
                "total_area"
            ]
            /
            current_image_area
        )

    else:

        current_annotation_ratio = np.nan


    if future_image_area > 0:

        future_annotation_ratio = (
            future_stats[
                "total_area"
            ]
            /
            future_image_area
        )

    else:

        future_annotation_ratio = np.nan


    # --------------------------------------------------------
    # CHANGES
    # --------------------------------------------------------

    polygon_count_change = (
        future_stats[
            "shape_count"
        ]
        -
        current_stats[
            "shape_count"
        ]
    )

    annotation_area_change = (
        future_stats[
            "total_area"
        ]
        -
        current_stats[
            "total_area"
        ]
    )

    damage_change = (
        float(
            event[
                "future_damage"
            ]
        )
        -
        float(
            event[
                "crack_area_ratio"
            ]
        )
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    results.append({

        "rank":
            rank,

        "sequence":
            sequence,

        "component":
            component,

        "split":
            str(
                current_row[
                    "split"
                ]
            ),

        "current_frame":
            current_frame,

        "future_frame":
            future_frame,

        "current_damage":
            float(
                event[
                    "crack_area_ratio"
                ]
            ),

        "future_damage":
            float(
                event[
                    "future_damage"
                ]
            ),

        "damage_change":
            damage_change,

        "absolute_damage_change":
            abs(
                damage_change
            ),

        "model_prediction":
            float(
                event[
                    "model_prediction"
                ]
            ),

        "model_absolute_error":
            float(
                event[
                    "absolute_error"
                ]
            ),

        "current_polygon_count":
            current_stats[
                "shape_count"
            ],

        "future_polygon_count":
            future_stats[
                "shape_count"
            ],

        "polygon_count_change":
            polygon_count_change,

        "current_polygon_area_px":
            current_stats[
                "total_area"
            ],

        "future_polygon_area_px":
            future_stats[
                "total_area"
            ],

        "polygon_area_change_px":
            annotation_area_change,

        "current_annotation_area_ratio":
            current_annotation_ratio,

        "future_annotation_area_ratio":
            future_annotation_ratio,

        "current_image_width":
            current_width,

        "current_image_height":
            current_height,

        "future_image_width":
            future_width,

        "future_image_height":
            future_height,

        "current_image":
            current_image_path,

        "future_image":
            future_image_path,

        "current_json":
            current_json_path,

        "future_json":
            future_json_path,

        "visualization":
            output_path
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# FLAGS
# ============================================================

if len(results_df) > 0:

    results_df[
        "polygon_count_changed"
    ] = (
        results_df[
            "polygon_count_change"
        ].abs()
        >
        0
    )

    results_df[
        "annotation_area_changed"
    ] = (
        results_df[
            "polygon_area_change_px"
        ].abs()
        >
        0
    )

    results_df[
        "image_dimensions_changed"
    ] = (
        (
            results_df[
                "current_image_width"
            ]
            !=
            results_df[
                "future_image_width"
            ]
        )
        |
        (
            results_df[
                "current_image_height"
            ]
            !=
            results_df[
                "future_image_height"
            ]
        )
    )

    results_df[
        "large_polygon_count_change"
    ] = (
        results_df[
            "polygon_count_change"
        ].abs()
        >=
        2
    )

    results_df[
        "large_annotation_area_change"
    ] = (
        results_df[
            "polygon_area_change_px"
        ].abs()
        >
        (
            results_df[
                "current_polygon_area_px"
            ]
            .replace(
                0,
                np.nan
            )
            *
            0.50
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CASES,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

total = len(
    results_df
)

if total > 0:

    polygon_changed = int(
        results_df[
            "polygon_count_changed"
        ].sum()
    )

    large_polygon_changes = int(
        results_df[
            "large_polygon_count_change"
        ].sum()
    )

    large_annotation_changes = int(
        results_df[
            "large_annotation_area_change"
        ]
        .fillna(False)
        .sum()
    )

    dimensions_changed = int(
        results_df[
            "image_dimensions_changed"
        ].sum()
    )

    mean_polygon_change = float(
        results_df[
            "polygon_count_change"
        ]
        .abs()
        .mean()
    )

    mean_annotation_ratio_change = float(
        (
            results_df[
                "future_annotation_area_ratio"
            ]
            -
            results_df[
                "current_annotation_area_ratio"
            ]
        )
        .abs()
        .mean()
    )

else:

    polygon_changed = 0
    large_polygon_changes = 0
    large_annotation_changes = 0
    dimensions_changed = 0
    mean_polygon_change = None
    mean_annotation_ratio_change = None


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "stage":
        "2.7",

    "status":
        "PASS",

    "events_requested":
        TOP_N,

    "events_validated":
        total,

    "annotation_validation": {

        "polygon_count_changed":
            polygon_changed,

        "polygon_count_changed_fraction":
            (
                polygon_changed / total
                if total > 0
                else 0
            ),

        "large_polygon_count_changes":
            large_polygon_changes,

        "large_annotation_area_changes":
            large_annotation_changes,

        "mean_absolute_polygon_count_change":
            mean_polygon_change,

        "mean_absolute_annotation_area_ratio_change":
            mean_annotation_ratio_change
    },

    "image_validation": {

        "image_dimensions_changed":
            dimensions_changed,

        "image_dimensions_changed_fraction":
            (
                dimensions_changed / total
                if total > 0
                else 0
            )
    },

    "output_directory":
        OUTPUT_DIR,

    "output_cases":
        OUTPUT_CASES,

    "interpretation":
        "Extreme transitions were extracted with "
        "current and future images and annotations. "
        "Manual visual inspection is required before "
        "classifying transitions as genuine physical "
        "damage progression or annotation artifacts."
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
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STAGE 2.7 SUMMARY")
print("=" * 70)

print(
    f"Extreme events requested: "
    f"{TOP_N}"
)

print(
    f"Extreme events validated: "
    f"{total}"
)

print()
print(
    f"Polygon count changed: "
    f"{polygon_changed}/{total}"
)

print(
    f"Large polygon count changes: "
    f"{large_polygon_changes}/{total}"
)

print(
    f"Large annotation area changes: "
    f"{large_annotation_changes}/{total}"
)

print(
    f"Image dimensions changed: "
    f"{dimensions_changed}/{total}"
)

print()
print(
    "Visualization directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Validation CSV:"
)

print(
    OUTPUT_CASES
)

print()
print(
    "Summary:"
)

print(
    OUTPUT_SUMMARY
)

print()
print(
    "FINAL STATUS: PASS"
)