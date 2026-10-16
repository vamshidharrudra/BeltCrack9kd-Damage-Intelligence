from pathlib import Path
import json
import math

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is not installed.")
    print("Install it with:")
    print("pip install pillow")
    raise


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

ANNOTATED_PACKAGES = [
    "BeltCrack9kd1",
    "BeltCrack9kd2",
    "BeltCrack9kd3",
    "BeltCrack9kd4",
]

# Number of neighboring frame pairs to inspect per data folder.
MAX_PAIRS_PER_FOLDER = 50

# Small image size used only for similarity auditing.
COMPARE_SIZE = (64, 64)


# ============================================================
# HELPERS
# ============================================================

def numeric_jpgs(folder):
    """
    Return:
        {numeric_id: jpg_path}

    Only numeric JPG filenames are included.
    Example:
        0.jpg -> 0
        1.jpg -> 1
    """

    result = {}

    if not folder.exists():
        return result

    for path in folder.glob("*.jpg"):

        try:
            index = int(path.stem)
            result[index] = path

        except ValueError:
            pass

    return result


def load_gray(path):
    """
    Load image as grayscale and resize to a small
    comparison resolution.
    """

    image = Image.open(path).convert("L")
    image = image.resize(COMPARE_SIZE)

    # Avoid deprecated Image.getdata()
    return list(image.get_flattened_data())


def normalized_difference(a, b):
    """
    Mean absolute pixel difference normalized to [0, 1].

    Lower = more similar.
    """

    if len(a) != len(b):
        return 1.0

    total = 0.0

    for x, y in zip(a, b):
        total += abs(x - y)

    mean_difference = total / len(a)

    return mean_difference / 255.0


def correlation(a, b):
    """
    Pearson correlation between two grayscale images.

    Higher = more visually similar.
    """

    if len(a) != len(b) or not a:
        return 0.0

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)

    numerator = 0.0
    denominator_a = 0.0
    denominator_b = 0.0

    for x, y in zip(a, b):

        dx = x - mean_a
        dy = y - mean_b

        numerator += dx * dy
        denominator_a += dx * dx
        denominator_b += dy * dy

    denominator = math.sqrt(
        denominator_a * denominator_b
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def polygon_area(points):
    """
    Calculate polygon area using the shoelace formula.
    """

    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):

        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        area += x1 * y2
        area -= x2 * y1

    return abs(area) / 2.0


def annotation_stats(json_path):
    """
    Extract basic annotation statistics.

    Returns:
        polygon_count
        total_polygon_area
        combined_bbox
    """

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception:

        return 0, 0.0, None

    shapes = data.get("shapes", [])

    polygons = []
    total_area = 0.0

    for shape in shapes:

        points = shape.get("points", [])

        if not points:
            continue

        clean_points = []

        for point in points:

            if len(point) >= 2:

                try:

                    x = float(point[0])
                    y = float(point[1])

                    clean_points.append(
                        (x, y)
                    )

                except Exception:
                    pass

        if len(clean_points) >= 3:

            polygons.append(
                clean_points
            )

            total_area += polygon_area(
                clean_points
            )

    if not polygons:

        return 0, 0.0, None

    xs = []
    ys = []

    for polygon in polygons:

        for x, y in polygon:

            xs.append(x)
            ys.append(y)

    bounding_box = (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )

    return (
        len(polygons),
        total_area,
        bounding_box
    )


def select_pairs(consecutive_pairs):
    """
    Select evenly distributed neighboring pairs.
    """

    if not consecutive_pairs:
        return []

    if len(consecutive_pairs) <= MAX_PAIRS_PER_FOLDER:

        return consecutive_pairs

    step = len(consecutive_pairs) / MAX_PAIRS_PER_FOLDER

    selected = []

    for i in range(MAX_PAIRS_PER_FOLDER):

        index = int(i * step)

        index = min(
            index,
            len(consecutive_pairs) - 1
        )

        selected.append(
            consecutive_pairs[index]
        )

    return selected


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 0.7 - SEQUENCE IDENTITY & TEMPORAL EVIDENCE AUDIT")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_ROOT)

print()
print("The audit will NOT modify images or annotations.")
print("It will NOT create sequences.")
print("It will NOT create train/val/test splits.")


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_ROOT.exists():

    print()
    print("ERROR: Dataset root does not exist:")
    print(DATASET_ROOT)
    raise SystemExit(1)


# ============================================================
# DISCOVER DATA FOLDERS
# ============================================================

folders = []

for package_name in ANNOTATED_PACKAGES:

    package_root = (
        DATASET_ROOT
        / package_name
        / "images"
    )

    if not package_root.exists():

        print()
        print(
            "WARNING: Missing package:",
            package_root
        )

        continue

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_root = package_root / split

        if not split_root.exists():
            continue

        for data_dir in sorted(
            split_root.glob("data*")
        ):

            if data_dir.is_dir():

                folders.append(
                    (
                        package_name,
                        split,
                        data_dir
                    )
                )


# ============================================================
# ANALYZE EACH DATA FOLDER
# ============================================================

global_results = []

print()
print("=" * 70)
print("DATA FOLDER SEQUENCE ANALYSIS")
print("=" * 70)


for package_name, split, data_dir in folders:

    images = numeric_jpgs(data_dir)

    if len(images) < 2:
        continue

    ids = sorted(images.keys())

    # --------------------------------------------------------
    # CONSECUTIVE FRAME ANALYSIS
    # --------------------------------------------------------

    consecutive_pairs = []

    for i in range(len(ids) - 1):

        current_id = ids[i]
        next_id = ids[i + 1]

        if next_id == current_id + 1:

            consecutive_pairs.append(
                (
                    current_id,
                    next_id
                )
            )

    selected_pairs = select_pairs(
        consecutive_pairs
    )

    # --------------------------------------------------------
    # IMAGE SIMILARITY
    # --------------------------------------------------------

    differences = []
    correlations = []

    for current_id, next_id in selected_pairs:

        current_path = images[current_id]
        next_path = images[next_id]

        try:

            current_image = load_gray(
                current_path
            )

            next_image = load_gray(
                next_path
            )

            diff = normalized_difference(
                current_image,
                next_image
            )

            corr = correlation(
                current_image,
                next_image
            )

            differences.append(diff)
            correlations.append(corr)

        except Exception:

            continue

    # --------------------------------------------------------
    # ANNOTATION CONTINUITY
    # --------------------------------------------------------

    package_root = (
        DATASET_ROOT
        / package_name
    )

    annotation_pairs = 0
    annotation_both = 0

    annotation_area_differences = []

    for current_id, next_id in selected_pairs:

        current_json = (
            package_root
            / "json"
            / split
            / data_dir.name
            / f"{current_id}.json"
        )

        next_json = (
            package_root
            / "json"
            / split
            / data_dir.name
            / f"{next_id}.json"
        )

        if (
            not current_json.exists()
            or not next_json.exists()
        ):
            continue

        annotation_pairs += 1

        current_count, current_area, current_bbox = (
            annotation_stats(
                current_json
            )
        )

        next_count, next_area, next_bbox = (
            annotation_stats(
                next_json
            )
        )

        if (
            current_count > 0
            and next_count > 0
        ):

            annotation_both += 1

            maximum_area = max(
                current_area,
                next_area
            )

            if maximum_area > 0:

                relative_difference = (
                    abs(
                        current_area
                        - next_area
                    )
                    / maximum_area
                )

            else:

                relative_difference = 0.0

            annotation_area_differences.append(
                relative_difference
            )

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------

    if differences:

        avg_difference = (
            sum(differences)
            / len(differences)
        )

    else:

        avg_difference = None

    if correlations:

        avg_correlation = (
            sum(correlations)
            / len(correlations)
        )

    else:

        avg_correlation = None

    if annotation_area_differences:

        avg_annotation_area_change = (
            sum(annotation_area_differences)
            / len(annotation_area_differences)
        )

    else:

        avg_annotation_area_change = None

    if len(ids) > 1:

        consecutive_coverage = (
            len(consecutive_pairs)
            / (len(ids) - 1)
        )

    else:

        consecutive_coverage = 0.0

    # --------------------------------------------------------
    # TEMPORAL EVIDENCE CLASSIFICATION
    # --------------------------------------------------------

    if (
        avg_correlation is not None
        and avg_correlation >= 0.90
        and consecutive_coverage >= 0.99
    ):

        evidence = "STRONG"

    elif (
        avg_correlation is not None
        and avg_correlation >= 0.70
        and consecutive_coverage >= 0.95
    ):

        evidence = "MODERATE"

    else:

        evidence = "WEAK / UNCERTAIN"

    result = {
        "package": package_name,
        "split": split,
        "data_folder": data_dir.name,
        "frames": len(ids),
        "consecutive_pairs": len(
            consecutive_pairs
        ),
        "consecutive_coverage": (
            consecutive_coverage
        ),
        "sampled_pairs": len(
            selected_pairs
        ),
        "avg_pixel_difference": (
            avg_difference
        ),
        "avg_correlation": (
            avg_correlation
        ),
        "annotation_pairs": (
            annotation_pairs
        ),
        "annotation_both": (
            annotation_both
        ),
        "avg_annotation_area_change": (
            avg_annotation_area_change
        ),
        "evidence": evidence,
    }

    global_results.append(result)

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        f"{split}/{data_dir.name}"
    )
    print("-" * 70)

    print(
        f"Frames                  : "
        f"{len(ids):,}"
    )

    print(
        f"Consecutive pairs       : "
        f"{len(consecutive_pairs):,}"
    )

    print(
        f"Consecutive coverage    : "
        f"{consecutive_coverage * 100:.2f}%"
    )

    print(
        f"Pairs visually sampled  : "
        f"{len(selected_pairs):,}"
    )

    if avg_difference is not None:

        print(
            f"Mean pixel difference   : "
            f"{avg_difference:.4f}"
        )

    else:

        print(
            "Mean pixel difference   : N/A"
        )

    if avg_correlation is not None:

        print(
            f"Mean image correlation : "
            f"{avg_correlation:.4f}"
        )

    else:

        print(
            "Mean image correlation : N/A"
        )

    print(
        f"Annotation pairs        : "
        f"{annotation_pairs:,}"
    )

    print(
        f"Both annotated          : "
        f"{annotation_both:,}"
    )

    if avg_annotation_area_change is not None:

        print(
            f"Mean annotation area change : "
            f"{avg_annotation_area_change * 100:.2f}%"
        )

    else:

        print(
            "Mean annotation area change : N/A"
        )

    print(
        f"Temporal evidence       : "
        f"{evidence}"
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("GLOBAL SEQUENCE EVIDENCE SUMMARY")
print("=" * 70)

strong = 0
moderate = 0
weak = 0

for result in global_results:

    if result["evidence"] == "STRONG":

        strong += 1

    elif result["evidence"] == "MODERATE":

        moderate += 1

    else:

        weak += 1

print()

print(
    f"Folders analyzed : "
    f"{len(global_results):,}"
)

print(
    f"STRONG           : "
    f"{strong:,}"
)

print(
    f"MODERATE         : "
    f"{moderate:,}"
)

print(
    f"WEAK / UNCERTAIN : "
    f"{weak:,}"
)


# ============================================================
# TOP CANDIDATES
# ============================================================

print()
print("=" * 70)
print("TOP SEQUENCE CANDIDATES")
print("=" * 70)

rankable = [
    result
    for result in global_results
    if result["avg_correlation"] is not None
]

rankable.sort(
    key=lambda result:
        result["avg_correlation"],
    reverse=True
)

for result in rankable[:20]:

    print(
        f"{result['split']}/"
        f"{result['data_folder']:<10} "
        f"frames={result['frames']:>5} "
        f"corr={result['avg_correlation']:.4f} "
        f"evidence={result['evidence']}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 0.7 AUDIT COMPLETE")
print("=" * 70)

print()
print("IMPORTANT:")
print(
    "High image similarity does NOT by itself "
    "prove temporal order."
)

print(
    "This audit provides sequence evidence, "
    "not temporal ground truth."
)

print()
print("DO NOT create final sequences yet.")
print(
    "DO NOT create the final train/val/test split yet."
)
print("DO NOT delete duplicate images.")
print(
    "DO NOT train the damage prediction model yet."
)