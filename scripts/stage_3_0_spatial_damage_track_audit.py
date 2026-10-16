import os
import json
import numpy as np
import pandas as pd

from PIL import Image


# ============================================================
# STAGE 3.0
# SPATIAL DAMAGE TRACK CONSTRUCTION AND CONTINUITY AUDIT
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

FEATURE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_1_0_damage_features.csv"
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset",
    "canonical_manifest.csv"
)

OUTPUT_PAIR_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_pairs.csv"
)

OUTPUT_SEQUENCE_FILE = os.path.join(
    AUDIT_DIR,
    "stage_3_0_sequence_track_summary.csv"
)

OUTPUT_SUMMARY = os.path.join(
    AUDIT_DIR,
    "stage_3_0_spatial_track_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Minimum IoU required for a strong spatial match.
IOU_THRESHOLD = 0.10

# Maximum normalized centroid movement allowed
# for a normal spatial match.
MAX_CENTER_DISTANCE = 0.25

# Area similarity threshold.
MIN_AREA_SIMILARITY = 0.20

# Combined score threshold.
MIN_MATCH_SCORE = 0.20


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_dataset_path(
    path_value
):

    if path_value is None:
        return None

    path_value = str(
        path_value
    ).strip()

    if not path_value:
        return None

    if os.path.isabs(
        path_value
    ):

        return os.path.normpath(
            path_value
        )

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
# JSON
# ============================================================

def load_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# POLYGON EXTRACTION
# ============================================================

def get_crack_polygons(
    data
):

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

                x = float(
                    point[0]
                )

                y = float(
                    point[1]
                )

                valid_points.append(
                    (
                        x,
                        y
                    )
                )

            except Exception:

                continue

        if len(
            valid_points
        ) >= 3:

            polygons.append(
                valid_points
            )

    return polygons


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(
    polygon
):

    if len(
        polygon
    ) < 3:

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

    return abs(
        area
    ) * 0.5


# ============================================================
# POLYGON BBOX
# ============================================================

def polygon_bbox(
    polygon
):

    xs = [
        point[0]
        for point in polygon
    ]

    ys = [
        point[1]
        for point in polygon
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


# ============================================================
# POLYGON CENTER
# ============================================================

def polygon_center(
    polygon
):

    bbox = polygon_bbox(
        polygon
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

    x1 = max(
        bbox1[0],
        bbox2[0]
    )

    y1 = max(
        bbox1[1],
        bbox2[1]
    )

    x2 = min(
        bbox1[2],
        bbox2[2]
    )

    y2 = min(
        bbox1[3],
        bbox2[3]
    )

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = (
        max(
            0.0,
            bbox1[2] - bbox1[0]
        )
        *
        max(
            0.0,
            bbox1[3] - bbox1[1]
        )
    )

    area2 = (
        max(
            0.0,
            bbox2[2] - bbox2[0]
        )
        *
        max(
            0.0,
            bbox2[3] - bbox2[1]
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

        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# NORMALIZED CENTER DISTANCE
# ============================================================

def normalized_center_distance(
    center1,
    center2,
    width,
    height
):

    dx = (
        center2[0]
        -
        center1[0]
    )

    dy = (
        center2[1]
        -
        center1[1]
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

        return 1.0

    return float(
        distance
        /
        diagonal
    )


# ============================================================
# AREA SIMILARITY
# ============================================================

def area_similarity(
    area1,
    area2
):

    if (
        area1 <= 0
        or
        area2 <= 0
    ):

        return 0.0

    smaller = min(
        area1,
        area2
    )

    larger = max(
        area1,
        area2
    )

    return float(
        smaller
        /
        larger
    )


# ============================================================
# CRACK OBJECT REPRESENTATION
# ============================================================

def build_objects(
    polygons
):

    objects = []

    for index, polygon in enumerate(
        polygons
    ):

        bbox = polygon_bbox(
            polygon
        )

        center = polygon_center(
            polygon
        )

        area = polygon_area(
            polygon
        )

        objects.append(
            {
                "id": index,
                "polygon": polygon,
                "bbox": bbox,
                "center": center,
                "area": area
            }
        )

    return objects


# ============================================================
# MATCH SCORE
# ============================================================

def calculate_match(
    current_object,
    future_object,
    image_width,
    image_height
):

    iou = bbox_iou(
        current_object["bbox"],
        future_object["bbox"]
    )

    center_distance = normalized_center_distance(
        current_object["center"],
        future_object["center"],
        image_width,
        image_height
    )

    similarity = area_similarity(
        current_object["area"],
        future_object["area"]
    )

    center_score = max(
        0.0,
        1.0
        -
        (
            center_distance
            /
            max(
                MAX_CENTER_DISTANCE,
                1e-8
            )
        )
    )

    score = (
        0.50 * iou
        +
        0.30 * center_score
        +
        0.20 * similarity
    )

    valid = (
        (
            iou
            >=
            IOU_THRESHOLD
        )
        or
        (
            center_distance
            <=
            MAX_CENTER_DISTANCE
            and
            similarity
            >=
            MIN_AREA_SIMILARITY
            and
            score
            >=
            MIN_MATCH_SCORE
        )
    )

    return {
        "iou": float(iou),
        "center_distance": float(
            center_distance
        ),
        "area_similarity": float(
            similarity
        ),
        "score": float(
            score
        ),
        "valid": bool(
            valid
        )
    }


# ============================================================
# GREEDY MATCHING
# ============================================================

def match_objects(
    current_objects,
    future_objects,
    image_width,
    image_height
):

    candidates = []

    for current in current_objects:

        for future in future_objects:

            metrics = calculate_match(
                current,
                future,
                image_width,
                image_height
            )

            if metrics["valid"]:

                candidates.append(
                    (
                        metrics["score"],
                        current["id"],
                        future["id"],
                        metrics
                    )
                )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    used_current = set()

    used_future = set()

    matches = []

    for (
        score,
        current_id,
        future_id,
        metrics
    ) in candidates:

        if current_id in used_current:
            continue

        if future_id in used_future:
            continue

        used_current.add(
            current_id
        )

        used_future.add(
            future_id
        )

        matches.append(
            {
                "current_id":
                    current_id,

                "future_id":
                    future_id,

                "score":
                    metrics["score"],

                "iou":
                    metrics["iou"],

                "center_distance":
                    metrics["center_distance"],

                "area_similarity":
                    metrics["area_similarity"]
            }
        )

    unmatched_current = [
        obj["id"]
        for obj in current_objects
        if obj["id"]
        not in used_current
    ]

    unmatched_future = [
        obj["id"]
        for obj in future_objects
        if obj["id"]
        not in used_future
    ]

    return (
        matches,
        unmatched_current,
        unmatched_future
    )


# ============================================================
# PROCESS ONE FRAME PAIR
# ============================================================

def process_pair(
    current_row,
    future_row
):

    current_json = resolve_dataset_path(
        current_row[
            "source_json"
        ]
    )

    future_json = resolve_dataset_path(
        future_row[
            "source_json"
        ]
    )

    current_image = resolve_dataset_path(
        current_row[
            "source_image"
        ]
    )

    future_image = resolve_dataset_path(
        future_row[
            "source_image"
        ]
    )

    if not os.path.exists(
        current_json
    ):

        return None

    if not os.path.exists(
        future_json
    ):

        return None

    if not os.path.exists(
        current_image
    ):

        return None

    if not os.path.exists(
        future_image
    ):

        return None

    try:

        current_data = load_json(
            current_json
        )

        future_data = load_json(
            future_json
        )

    except Exception:

        return None

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

    current_objects = build_objects(
        current_polygons
    )

    future_objects = build_objects(
        future_polygons
    )

    try:

        current_img = Image.open(
            current_image
        )

        width, height = (
            current_img.size
        )

    except Exception:

        return None

    (
        matches,
        unmatched_current,
        unmatched_future
    ) = match_objects(
        current_objects,
        future_objects,
        width,
        height
    )

    # --------------------------------------------------------
    # MATCH METRICS
    # --------------------------------------------------------

    matched_count = len(
        matches
    )

    current_count = len(
        current_objects
    )

    future_count = len(
        future_objects
    )

    persistent_fraction = (
        matched_count
        /
        max(
            current_count,
            1
        )
    )

    future_coverage = (
        matched_count
        /
        max(
            future_count,
            1
        )
    )

    if matches:

        mean_iou = float(
            np.mean(
                [
                    item["iou"]
                    for item in matches
                ]
            )
        )

        mean_center_distance = float(
            np.mean(
                [
                    item[
                        "center_distance"
                    ]
                    for item in matches
                ]
            )
        )

        mean_area_similarity = float(
            np.mean(
                [
                    item[
                        "area_similarity"
                    ]
                    for item in matches
                ]
            )
        )

        mean_match_score = float(
            np.mean(
                [
                    item["score"]
                    for item in matches
                ]
            )
        )

    else:

        mean_iou = 0.0

        mean_center_distance = np.nan

        mean_area_similarity = 0.0

        mean_match_score = 0.0


    # --------------------------------------------------------
    # MATCHED AREA CHANGES
    # --------------------------------------------------------

    area_growth_values = []

    area_change_values = []

    matched_details = []

    for match in matches:

        current_object = (
            current_objects[
                match[
                    "current_id"
                ]
            ]
        )

        future_object = (
            future_objects[
                match[
                    "future_id"
                ]
            ]
        )

        current_area = (
            current_object[
                "area"
            ]
        )

        future_area = (
            future_object[
                "area"
            ]
        )

        absolute_change = (
            future_area
            -
            current_area
        )

        if current_area > 0:

            relative_change = (
                future_area
                -
                current_area
            ) / current_area

        else:

            relative_change = np.nan

        area_change_values.append(
            absolute_change
        )

        if not np.isnan(
            relative_change
        ):

            area_growth_values.append(
                relative_change
            )

        matched_details.append(
            {
                "current_id":
                    match[
                        "current_id"
                    ],

                "future_id":
                    match[
                        "future_id"
                    ],

                "current_area":
                    current_area,

                "future_area":
                    future_area,

                "absolute_area_change":
                    absolute_change,

                "relative_area_change":
                    relative_change,

                "iou":
                    match["iou"],

                "center_distance":
                    match[
                        "center_distance"
                    ],

                "area_similarity":
                    match[
                        "area_similarity"
                    ],

                "match_score":
                    match[
                        "score"
                    ]
            }
        )


    if area_growth_values:

        mean_relative_area_change = float(
            np.mean(
                area_growth_values
            )
        )

        median_relative_area_change = float(
            np.median(
                area_growth_values
            )
        )

        max_relative_area_change = float(
            np.max(
                area_growth_values
            )
        )

        min_relative_area_change = float(
            np.min(
                area_growth_values
            )
        )

    else:

        mean_relative_area_change = np.nan
        median_relative_area_change = np.nan
        max_relative_area_change = np.nan
        min_relative_area_change = np.nan


    # --------------------------------------------------------
    # TOTAL ANNOTATED AREA
    # --------------------------------------------------------

    current_total_area = float(
        sum(
            obj["area"]
            for obj in current_objects
        )
    )

    future_total_area = float(
        sum(
            obj["area"]
            for obj in future_objects
        )
    )

    if current_total_area > 0:

        total_area_relative_change = (
            future_total_area
            -
            current_total_area
        ) / current_total_area

    else:

        total_area_relative_change = np.nan


    # --------------------------------------------------------
    # TRACK CONTINUITY CLASS
    # --------------------------------------------------------

    if current_count == 0:

        continuity_class = (
            "NO_CURRENT_DAMAGE"
        )

    elif future_count == 0:

        continuity_class = (
            "DAMAGE_DISAPPEARED"
        )

    elif persistent_fraction >= 0.75:

        continuity_class = (
            "HIGH_CONTINUITY"
        )

    elif persistent_fraction >= 0.40:

        continuity_class = (
            "MODERATE_CONTINUITY"
        )

    else:

        continuity_class = (
            "LOW_CONTINUITY"
        )


    # --------------------------------------------------------
    # PAIR RESULT
    # --------------------------------------------------------

    return {

        "sequence":
            current_row[
                "sequence"
            ],

        "component":
            int(
                current_row[
                    "component"
                ]
            ),

        "split":
            current_row[
                "split"
            ],

        "current_frame":
            int(
                current_row[
                    "frame_number"
                ]
            ),

        "future_frame":
            int(
                future_row[
                    "frame_number"
                ]
            ),

        "current_crack_count":
            current_count,

        "future_crack_count":
            future_count,

        "matched_crack_count":
            matched_count,

        "new_crack_count":
            len(
                unmatched_future
            ),

        "disappeared_crack_count":
            len(
                unmatched_current
            ),

        "persistent_fraction":
            float(
                persistent_fraction
            ),

        "future_coverage":
            float(
                future_coverage
            ),

        "mean_match_iou":
            mean_iou,

        "mean_center_distance":
            mean_center_distance,

        "mean_area_similarity":
            mean_area_similarity,

        "mean_match_score":
            mean_match_score,

        "mean_relative_matched_area_change":
            mean_relative_area_change,

        "median_relative_matched_area_change":
            median_relative_area_change,

        "max_relative_matched_area_change":
            max_relative_area_change,

        "min_relative_matched_area_change":
            min_relative_area_change,

        "current_total_area":
            current_total_area,

        "future_total_area":
            future_total_area,

        "total_area_relative_change":
            total_area_relative_change,

        "continuity_class":
            continuity_class,

        "current_source_image":
            current_image,

        "future_source_image":
            future_image,

        "matched_details":
            matched_details
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "STAGE 3.0 - SPATIAL DAMAGE TRACK CONSTRUCTION"
)
print(
    "AND CONTINUITY AUDIT"
)
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print(
    "Checking feature file..."
)

if not os.path.exists(
    FEATURE_FILE
):

    raise FileNotFoundError(
        "Stage 1.0 feature file not found:\n"
        +
        FEATURE_FILE
    )

print(
    "Feature file: FOUND"
)


print()
print(
    "Checking canonical manifest..."
)

if not os.path.exists(
    MANIFEST_FILE
):

    raise FileNotFoundError(
        "Canonical manifest not found:\n"
        +
        MANIFEST_FILE
    )

print(
    "Manifest: FOUND"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print(
    "Loading Stage 1.0 features..."
)

features = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Feature records: "
    f"{len(features)}"
)


print()
print(
    "Loading canonical manifest..."
)

manifest = pd.read_csv(
    MANIFEST_FILE
)

print(
    f"Manifest records: "
    f"{len(manifest)}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

feature_required = [
    "split",
    "component",
    "sequence",
    "source_image",
    "source_json",
    "crack_count",
    "crack_area_ratio"
]

manifest_required = [
    "split",
    "component",
    "sequence",
    "source_image",
    "source_json"
]

missing_features = [
    col
    for col in feature_required
    if col not in features.columns
]

missing_manifest = [
    col
    for col in manifest_required
    if col not in manifest.columns
]

if missing_features:

    raise ValueError(
        "Missing feature columns:\n"
        +
        ", ".join(
            missing_features
        )
    )

if missing_manifest:

    raise ValueError(
        "Missing manifest columns:\n"
        +
        ", ".join(
            missing_manifest
        )
    )


# ============================================================
# FRAME NUMBER
# ============================================================

def extract_frame_number(
    filename
):

    name = os.path.basename(
        str(filename)
    )

    stem = os.path.splitext(
        name
    )[0]

    try:

        return int(
            stem
        )

    except Exception:

        return np.nan


features[
    "frame_number"
] = features[
    "original_image_filename"
].apply(
    extract_frame_number
)


# ============================================================
# CHECK FRAME NUMBERS
# ============================================================

missing_frame_numbers = int(
    features[
        "frame_number"
    ].isna().sum()
)

if missing_frame_numbers > 0:

    raise ValueError(
        f"Could not determine frame numbers "
        f"for {missing_frame_numbers} records."
    )


features[
    "frame_number"
] = features[
    "frame_number"
].astype(
    int
)


# ============================================================
# SORT
# ============================================================

features = features.sort_values(
    [
        "sequence",
        "frame_number"
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD FRAME PAIRS
# ============================================================

print()
print(
    "Constructing consecutive frame pairs..."
)

pair_results = []

sequence_count = 0

for sequence_name, sequence_df in (
    features.groupby(
        "sequence",
        sort=False
    )
):

    sequence_count += 1

    sequence_df = sequence_df.sort_values(
        "frame_number"
    ).reset_index(
        drop=True
    )

    for i in range(
        len(sequence_df) - 1
    ):

        current_row = sequence_df.iloc[
            i
        ]

        future_row = sequence_df.iloc[
            i + 1
        ]

        current_frame = int(
            current_row[
                "frame_number"
            ]
        )

        future_frame = int(
            future_row[
                "frame_number"
            ]
        )

        if future_frame != (
            current_frame + 1
        ):

            continue

        result = process_pair(
            current_row,
            future_row
        )

        if result is not None:

            pair_results.append(
                result
            )

    print(
        f"Processed sequence "
        f"{sequence_count:02d}: "
        f"{sequence_name}"
    )


# ============================================================
# DATAFRAME
# ============================================================

if len(
    pair_results
) == 0:

    raise RuntimeError(
        "No spatial frame pairs were constructed."
    )


pair_df = pd.DataFrame(
    pair_results
)


# ============================================================
# SAVE PAIR RESULTS
# ============================================================

# matched_details contains nested data.
# Save the main pair table without nested details.

pair_output_df = pair_df.drop(
    columns=[
        "matched_details"
    ]
)

pair_output_df.to_csv(
    OUTPUT_PAIR_FILE,
    index=False
)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

sequence_summaries = []

for sequence_name, group in (
    pair_output_df.groupby(
        "sequence"
    )
):

    total_pairs = len(
        group
    )

    high_continuity = int(
        (
            group[
                "continuity_class"
            ]
            ==
            "HIGH_CONTINUITY"
        ).sum()
    )

    moderate_continuity = int(
        (
            group[
                "continuity_class"
            ]
            ==
            "MODERATE_CONTINUITY"
        ).sum()
    )

    low_continuity = int(
        (
            group[
                "continuity_class"
            ]
            ==
            "LOW_CONTINUITY"
        ).sum()
    )

    disappeared = int(
        (
            group[
                "continuity_class"
            ]
            ==
            "DAMAGE_DISAPPEARED"
        ).sum()
    )

    no_current = int(
        (
            group[
                "continuity_class"
            ]
            ==
            "NO_CURRENT_DAMAGE"
        ).sum()
    )

    sequence_summaries.append(
        {
            "sequence":
                sequence_name,

            "component":
                int(
                    group[
                        "component"
                    ].iloc[0]
                ),

            "split":
                group[
                    "split"
                ].iloc[0],

            "pairs":
                total_pairs,

            "mean_matched_crack_count":
                float(
                    group[
                        "matched_crack_count"
                    ].mean()
                ),

            "mean_persistent_fraction":
                float(
                    group[
                        "persistent_fraction"
                    ].mean()
                ),

            "median_persistent_fraction":
                float(
                    group[
                        "persistent_fraction"
                    ].median()
                ),

            "mean_future_coverage":
                float(
                    group[
                        "future_coverage"
                    ].mean()
                ),

            "mean_match_iou":
                float(
                    group[
                        "mean_match_iou"
                    ].mean()
                ),

            "mean_center_distance":
                float(
                    group[
                        "mean_center_distance"
                    ].mean()
                ),

            "mean_area_similarity":
                float(
                    group[
                        "mean_area_similarity"
                    ].mean()
                ),

            "mean_relative_matched_area_change":
                float(
                    group[
                        "mean_relative_matched_area_change"
                    ].mean()
                ),

            "median_relative_matched_area_change":
                float(
                    group[
                        "median_relative_matched_area_change"
                    ].median()
                ),

            "mean_total_area_relative_change":
                float(
                    group[
                        "total_area_relative_change"
                    ].mean()
                ),

            "high_continuity_pairs":
                high_continuity,

            "moderate_continuity_pairs":
                moderate_continuity,

            "low_continuity_pairs":
                low_continuity,

            "disappeared_pairs":
                disappeared,

            "no_current_damage_pairs":
                no_current
        }
    )


sequence_df = pd.DataFrame(
    sequence_summaries
)

sequence_df.to_csv(
    OUTPUT_SEQUENCE_FILE,
    index=False
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_pairs = len(
    pair_output_df
)

high_continuity_pairs = int(
    (
        pair_output_df[
            "continuity_class"
        ]
        ==
        "HIGH_CONTINUITY"
    ).sum()
)

moderate_continuity_pairs = int(
    (
        pair_output_df[
            "continuity_class"
        ]
        ==
        "MODERATE_CONTINUITY"
    ).sum()
)

low_continuity_pairs = int(
    (
        pair_output_df[
            "continuity_class"
        ]
        ==
        "LOW_CONTINUITY"
    ).sum()
)

disappeared_pairs = int(
    (
        pair_output_df[
            "continuity_class"
        ]
        ==
        "DAMAGE_DISAPPEARED"
    ).sum()
)

no_current_damage_pairs = int(
    (
        pair_output_df[
            "continuity_class"
        ]
        ==
        "NO_CURRENT_DAMAGE"
    ).sum()
)


# ============================================================
# CONTINUITY RATES
# ============================================================

high_continuity_rate = (
    high_continuity_pairs
    /
    total_pairs
)

moderate_or_higher_rate = (
    (
        high_continuity_pairs
        +
        moderate_continuity_pairs
    )
    /
    total_pairs
)


# ============================================================
# GLOBAL METRICS
# ============================================================

mean_persistent_fraction = float(
    pair_output_df[
        "persistent_fraction"
    ].mean()
)

median_persistent_fraction = float(
    pair_output_df[
        "persistent_fraction"
    ].median()
)

mean_future_coverage = float(
    pair_output_df[
        "future_coverage"
    ].mean()
)

mean_match_iou = float(
    pair_output_df[
        "mean_match_iou"
    ].mean()
)

median_match_iou = float(
    pair_output_df[
        "mean_match_iou"
    ].median()
)

mean_center_distance = float(
    pair_output_df[
        "mean_center_distance"
    ].mean()
)

mean_area_similarity = float(
    pair_output_df[
        "mean_area_similarity"
    ].mean()
)

mean_relative_area_change = float(
    pair_output_df[
        "mean_relative_matched_area_change"
    ].mean()
)

median_relative_area_change = float(
    pair_output_df[
        "median_relative_matched_area_change"
    ].median()
)


# ============================================================
# DAMAGE CHANGE RELATIONSHIP
# ============================================================

correlation_data = pair_output_df[
    [
        "persistent_fraction",
        "mean_match_iou",
        "mean_center_distance",
        "mean_area_similarity",
        "total_area_relative_change",
        "current_crack_count",
        "future_crack_count"
    ]
].copy()

correlation_matrix = (
    correlation_data.corr()
)


# ============================================================
# EXTREME TRANSITIONS
# ============================================================

extreme_threshold = float(
    pair_output_df[
        "total_area_relative_change"
    ].abs().quantile(
        0.95
    )
)

extreme_pairs = pair_output_df[
    pair_output_df[
        "total_area_relative_change"
    ].abs()
    >=
    extreme_threshold
].copy()


extreme_high_continuity = int(
    (
        extreme_pairs[
            "continuity_class"
        ]
        ==
        "HIGH_CONTINUITY"
    ).sum()
)

extreme_moderate_or_high = int(
    extreme_pairs[
        "continuity_class"
    ].isin(
        [
            "HIGH_CONTINUITY",
            "MODERATE_CONTINUITY"
        ]
    ).sum()
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "stage":
        "3.0",

    "status":
        "PASS",

    "feature_records":
        int(
            len(features)
        ),

    "sequences":
        int(
            features[
                "sequence"
            ].nunique()
        ),

    "consecutive_pairs":
        int(
            total_pairs
        ),

    "continuity": {

        "high_continuity_pairs":
            high_continuity_pairs,

        "moderate_continuity_pairs":
            moderate_continuity_pairs,

        "low_continuity_pairs":
            low_continuity_pairs,

        "damage_disappeared_pairs":
            disappeared_pairs,

        "no_current_damage_pairs":
            no_current_damage_pairs,

        "high_continuity_rate":
            high_continuity_rate,

        "moderate_or_higher_rate":
            moderate_or_higher_rate,

        "mean_persistent_fraction":
            mean_persistent_fraction,

        "median_persistent_fraction":
            median_persistent_fraction,

        "mean_future_coverage":
            mean_future_coverage
    },

    "spatial_matching": {

        "mean_match_iou":
            mean_match_iou,

        "median_match_iou":
            median_match_iou,

        "mean_center_distance":
            mean_center_distance,

        "mean_area_similarity":
            mean_area_similarity
    },

    "matched_damage_evolution": {

        "mean_relative_area_change":
            mean_relative_area_change,

        "median_relative_area_change":
            median_relative_area_change
    },

    "extreme_transition_analysis": {

        "absolute_total_area_change_95th_percentile":
            extreme_threshold,

        "extreme_pair_count":
            int(
                len(
                    extreme_pairs
                )
            ),

        "extreme_high_continuity":
            extreme_high_continuity,

        "extreme_moderate_or_high_continuity":
            extreme_moderate_or_high
    },

    "correlation_matrix":
        correlation_matrix.round(
            6
        ).to_dict(),

    "outputs": {

        "pair_file":
            OUTPUT_PAIR_FILE,

        "sequence_file":
            OUTPUT_SEQUENCE_FILE,

        "summary_file":
            OUTPUT_SUMMARY
    },

    "interpretation":
        (
            "Stage 3.0 evaluates whether annotated crack "
            "regions can be spatially matched between "
            "consecutive frames. High continuity supports "
            "construction of spatial damage tracks. Low "
            "continuity indicates that aggregate damage "
            "prediction should account for region appearance, "
            "disappearance, or spatial movement."
        )
}


# ============================================================
# SAVE SUMMARY
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
# DISPLAY TOP SEQUENCES
# ============================================================

print()
print("=" * 70)
print(
    "SEQUENCE CONTINUITY SUMMARY"
)
print("=" * 70)

display_columns = [
    "sequence",
    "pairs",
    "mean_persistent_fraction",
    "median_persistent_fraction",
    "mean_match_iou",
    "mean_center_distance",
    "mean_area_similarity",
    "high_continuity_pairs",
    "moderate_continuity_pairs",
    "low_continuity_pairs"
]

print(
    sequence_df.sort_values(
        "mean_persistent_fraction",
        ascending=False
    )[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print(
    "STAGE 3.0 SUMMARY"
)
print("=" * 70)

print(
    f"Feature records: "
    f"{len(features)}"
)

print(
    f"Sequences: "
    f"{features['sequence'].nunique()}"
)

print(
    f"Consecutive frame pairs: "
    f"{total_pairs}"
)

print()
print(
    "Continuity:"
)

print(
    f"  High continuity pairs: "
    f"{high_continuity_pairs}"
)

print(
    f"  Moderate continuity pairs: "
    f"{moderate_continuity_pairs}"
)

print(
    f"  Low continuity pairs: "
    f"{low_continuity_pairs}"
)

print(
    f"  Damage disappeared pairs: "
    f"{disappeared_pairs}"
)

print(
    f"  No current damage pairs: "
    f"{no_current_damage_pairs}"
)

print()
print(
    f"High continuity rate: "
    f"{high_continuity_rate:.6f}"
)

print(
    f"Moderate-or-higher continuity rate: "
    f"{moderate_or_higher_rate:.6f}"
)

print()
print(
    "Spatial matching:"
)

print(
    f"  Mean persistent fraction: "
    f"{mean_persistent_fraction:.6f}"
)

print(
    f"  Median persistent fraction: "
    f"{median_persistent_fraction:.6f}"
)

print(
    f"  Mean future coverage: "
    f"{mean_future_coverage:.6f}"
)

print(
    f"  Mean match IoU: "
    f"{mean_match_iou:.6f}"
)

print(
    f"  Median match IoU: "
    f"{median_match_iou:.6f}"
)

print(
    f"  Mean center distance: "
    f"{mean_center_distance:.6f}"
)

print(
    f"  Mean area similarity: "
    f"{mean_area_similarity:.6f}"
)

print()
print(
    "Matched damage evolution:"
)

print(
    f"  Mean relative matched-area change: "
    f"{mean_relative_area_change:.6f}"
)

print(
    f"  Median relative matched-area change: "
    f"{median_relative_area_change:.6f}"
)

print()
print(
    "Extreme transitions:"
)

print(
    f"  95th percentile absolute total-area change: "
    f"{extreme_threshold:.6f}"
)

print(
    f"  Extreme pairs: "
    f"{len(extreme_pairs)}"
)

print(
    f"  Extreme high-continuity pairs: "
    f"{extreme_high_continuity}"
)

print(
    f"  Extreme moderate-or-higher continuity: "
    f"{extreme_moderate_or_high}"
)

print()
print(
    "Output pair file:"
)

print(
    OUTPUT_PAIR_FILE
)

print()
print(
    "Output sequence file:"
)

print(
    OUTPUT_SEQUENCE_FILE
)

print()
print(
    "Output summary:"
)

print(
    OUTPUT_SUMMARY
)

print()
print(
    "FINAL STATUS: PASS"
)