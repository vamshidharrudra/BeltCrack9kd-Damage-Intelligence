from pathlib import Path
import json
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

PACKAGES = [1, 2, 3, 4]


# ============================================================
# BASIC HELPERS
# ============================================================

def package_root(pkg):
    return DATASET_ROOT / f"BeltCrack9kd{pkg}"


def annotation_root(pkg):
    return package_root(pkg) / "json"


# ============================================================
# SAFE JSON LOADING
# ============================================================

def load_json(path):
    """
    Always returns exactly:

        (data, error)

    data  = parsed JSON dictionary or None
    error = error message or None
    """

    try:

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data, None

    except Exception as e:

        return None, str(e)


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

def normalize_number(value):

    if isinstance(value, float):
        return round(value, 6)

    if isinstance(value, int):
        return value

    return value


# ============================================================
# POINT NORMALIZATION
# ============================================================

def normalize_points(points):

    normalized = []

    if not isinstance(points, list):
        return tuple(normalized)

    for point in points:

        if (
            isinstance(point, (list, tuple))
            and len(point) >= 2
        ):

            x = normalize_number(point[0])
            y = normalize_number(point[1])

            normalized.append((x, y))

    return tuple(normalized)


# ============================================================
# ANNOTATION SIGNATURE
# ============================================================

def annotation_signature(data):

    """
    Creates a deterministic representation of the
    actual annotation content.

    Ignored fields:

        version
        flags
        imagePath
        imageData

    Compared fields:

        imageHeight
        imageWidth
        label
        shape_type
        polygon points
    """

    if data is None:
        return None

    image_height = data.get("imageHeight")
    image_width = data.get("imageWidth")

    shapes = data.get("shapes", [])

    normalized_shapes = []

    for shape in shapes:

        label = shape.get("label")

        shape_type = shape.get("shape_type")

        points = normalize_points(
            shape.get("points", [])
        )

        normalized_shapes.append(
            (
                label,
                shape_type,
                points
            )
        )

    # Annotation order should not matter.
    normalized_shapes.sort(
        key=lambda x: (
            str(x[0]),
            str(x[1]),
            str(x[2])
        )
    )

    return (
        image_height,
        image_width,
        tuple(normalized_shapes)
    )


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(points):

    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):

        x1, y1 = points[i]

        x2, y2 = points[
            (i + 1) % len(points)
        ]

        area += (
            x1 * y2
            -
            x2 * y1
        )

    return abs(area) / 2.0


# ============================================================
# ANNOTATION STATISTICS
# ============================================================

def annotation_statistics(data):

    if data is None:

        return {
            "shape_count": 0,
            "labels": Counter(),
            "areas": []
        }

    shapes = data.get("shapes", [])

    labels = Counter()

    areas = []

    for shape in shapes:

        label = shape.get("label")

        labels[label] += 1

        points = normalize_points(
            shape.get("points", [])
        )

        areas.append(
            polygon_area(points)
        )

    return {
        "shape_count": len(shapes),
        "labels": labels,
        "areas": areas
    }


# ============================================================
# BUILD PACKAGE ANNOTATION INDEX
# ============================================================

def build_annotation_index(pkg):

    root = annotation_root(pkg)

    index = {}

    errors = []

    if not root.exists():

        errors.append(
            {
                "key": "",
                "error": f"Missing directory: {root}"
            }
        )

        return index, errors

    json_files = sorted(
        root.rglob("*.json")
    )

    for json_path in json_files:

        # Example:
        #
        # train/data15/0.json
        #
        # This is the correct identity of the annotation.

        relative_key = (
            json_path
            .relative_to(root)
            .as_posix()
        )

        data, error = load_json(json_path)

        if error is not None:

            errors.append(
                {
                    "key": relative_key,
                    "error": error
                }
            )

            continue

        signature = annotation_signature(
            data
        )

        statistics = annotation_statistics(
            data
        )

        index[relative_key] = {

            "path": json_path,

            "data": data,

            "signature": signature,

            "stats": statistics
        }

    return index, errors


# ============================================================
# PACKAGE SUMMARY
# ============================================================

def print_package_summary(
    pkg,
    index,
    errors
):

    print()
    print("=" * 70)
    print(f"PACKAGE {pkg}")
    print("=" * 70)

    print(
        f"JSON files loaded : {len(index):,}"
    )

    print(
        f"JSON errors       : {len(errors):,}"
    )

    # --------------------------------------------------------
    # Split counts
    # --------------------------------------------------------

    split_counts = Counter()

    label_counts = Counter()

    polygon_counts = Counter()

    for key, item in index.items():

        parts = key.split("/")

        if len(parts) >= 1:

            split = parts[0]

            split_counts[split] += 1

        stats = item["stats"]

        polygon_counts[
            stats["shape_count"]
        ] += 1

        label_counts.update(
            stats["labels"]
        )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    print()
    print("Split distribution:")

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"  {split:8s}: {count:,}"
        )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    print()
    print("Labels:")

    if not label_counts:

        print("  None")

    else:

        for label, count in (
            label_counts
            .most_common()
        ):

            print(
                f"  {str(label):20s}: {count:,}"
            )

    # --------------------------------------------------------
    # POLYGONS PER IMAGE
    # --------------------------------------------------------

    print()
    print("Polygons per image:")

    for polygon_count, image_count in sorted(
        polygon_counts.items()
    ):

        print(
            f"  {polygon_count:3d} polygon(s): "
            f"{image_count:,} image(s)"
        )


# ============================================================
# COMPARE TWO PACKAGES
# ============================================================

def compare_packages(
    index_a,
    index_b,
    pkg_a,
    pkg_b
):

    keys_a = set(index_a.keys())

    keys_b = set(index_b.keys())

    # --------------------------------------------------------
    # KEY OVERLAP
    # --------------------------------------------------------

    common_keys = (
        keys_a & keys_b
    )

    only_a = (
        keys_a - keys_b
    )

    only_b = (
        keys_b - keys_a
    )

    identical = []

    different = []

    # --------------------------------------------------------
    # COMPARE ANNOTATIONS
    # --------------------------------------------------------

    for key in sorted(common_keys):

        signature_a = (
            index_a[key]["signature"]
        )

        signature_b = (
            index_b[key]["signature"]
        )

        if signature_a == signature_b:

            identical.append(key)

        else:

            different.append(key)

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        f"PACKAGE {pkg_a} vs PACKAGE {pkg_b}"
    )
    print("-" * 70)

    print(
        f"Package {pkg_a} annotations : "
        f"{len(keys_a):,}"
    )

    print(
        f"Package {pkg_b} annotations : "
        f"{len(keys_b):,}"
    )

    print(
        f"Common annotation keys     : "
        f"{len(common_keys):,}"
    )

    print(
        f"Only in package {pkg_a}     : "
        f"{len(only_a):,}"
    )

    print(
        f"Only in package {pkg_b}     : "
        f"{len(only_b):,}"
    )

    print(
        f"Identical annotations      : "
        f"{len(identical):,}"
    )

    print(
        f"Different annotations      : "
        f"{len(different):,}"
    )

    # --------------------------------------------------------
    # PERCENTAGES
    # --------------------------------------------------------

    if len(common_keys) > 0:

        identical_percentage = (
            len(identical)
            /
            len(common_keys)
            *
            100
        )

        different_percentage = (
            len(different)
            /
            len(common_keys)
            *
            100
        )

        print()

        print(
            f"Identical among common     : "
            f"{identical_percentage:.2f}%"
        )

        print(
            f"Different among common     : "
            f"{different_percentage:.2f}%"
        )

    # --------------------------------------------------------
    # DIFFERENCE EXAMPLES
    # --------------------------------------------------------

    if different:

        print()
        print(
            "Examples of DIFFERENT annotations:"
        )

        for key in different[:10]:

            item_a = index_a[key]

            item_b = index_b[key]

            stats_a = item_a["stats"]

            stats_b = item_b["stats"]

            print()

            print(
                f"  {key}"
            )

            print(
                f"    Package {pkg_a}: "
                f"{stats_a['shape_count']} polygon(s), "
                f"labels={dict(stats_a['labels'])}"
            )

            print(
                f"    Package {pkg_b}: "
                f"{stats_b['shape_count']} polygon(s), "
                f"labels={dict(stats_b['labels'])}"
            )

    return {

        "common": common_keys,

        "only_a": only_a,

        "only_b": only_b,

        "identical": identical,

        "different": different
    }


# ============================================================
# SUBSET ANALYSIS
# ============================================================

def subset_analysis(
    reference_index,
    candidate_index,
    reference_pkg,
    candidate_pkg
):

    reference_keys = set(
        reference_index.keys()
    )

    candidate_keys = set(
        candidate_index.keys()
    )

    shared = (
        reference_keys
        &
        candidate_keys
    )

    candidate_only = (
        candidate_keys
        -
        reference_keys
    )

    identical = []

    different = []

    for key in sorted(shared):

        if (
            candidate_index[key]["signature"]
            ==
            reference_index[key]["signature"]
        ):

            identical.append(key)

        else:

            different.append(key)

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"SUBSET ANALYSIS: "
        f"PACKAGE {candidate_pkg} "
        f"inside PACKAGE {reference_pkg}"
    )

    print("=" * 70)

    print(
        f"Candidate annotations       : "
        f"{len(candidate_keys):,}"
    )

    print(
        f"Reference annotations       : "
        f"{len(reference_keys):,}"
    )

    print(
        f"Candidate keys in reference : "
        f"{len(shared):,}"
    )

    print(
        f"Candidate-only keys         : "
        f"{len(candidate_only):,}"
    )

    print(
        f"Identical annotations       : "
        f"{len(identical):,}"
    )

    print(
        f"Different annotations       : "
        f"{len(different):,}"
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if len(candidate_keys) > 0:

        coverage = (
            len(shared)
            /
            len(candidate_keys)
            *
            100
        )

        print()

        print(
            f"Candidate annotation coverage "
            f"in reference: {coverage:.2f}%"
        )

    # --------------------------------------------------------
    # DIFFERENCE EXAMPLES
    # --------------------------------------------------------

    if different:

        print()

        print(
            "Examples of annotations that "
            "differ from reference:"
        )

        for key in different[:10]:

            candidate_stats = (
                candidate_index[key]["stats"]
            )

            reference_stats = (
                reference_index[key]["stats"]
            )

            print()

            print(
                f"  {key}"
            )

            print(
                f"    Candidate package "
                f"{candidate_pkg}: "
                f"{candidate_stats['shape_count']} polygon(s), "
                f"labels="
                f"{dict(candidate_stats['labels'])}"
            )

            print(
                f"    Reference package "
                f"{reference_pkg}: "
                f"{reference_stats['shape_count']} polygon(s), "
                f"labels="
                f"{dict(reference_stats['labels'])}"
            )

    return {

        "shared": shared,

        "candidate_only": candidate_only,

        "identical": identical,

        "different": different
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "BELTCRACK9KD - STAGE 0.4"
    )

    print(
        "ANNOTATION OVERLAP & CONSISTENCY AUDIT"
    )

    print("=" * 70)

    print()

    print(
        "Dataset root:"
    )

    print(
        DATASET_ROOT
    )

    print()

    print(
        "Annotation identity:"
    )

    print(
        "json/<split>/<dataN>/<ID>.json"
    )

    print()

    print(
        "imagePath inside JSON is ignored."
    )

    print(
        "No files will be modified."
    )

    # ========================================================
    # LOAD PACKAGES
    # ========================================================

    indexes = {}

    errors = {}

    for pkg in PACKAGES:

        print()

        print(
            f"Loading package {pkg}..."
        )

        index, pkg_errors = (
            build_annotation_index(pkg)
        )

        indexes[pkg] = index

        errors[pkg] = pkg_errors

        print_package_summary(
            pkg,
            index,
            pkg_errors
        )

    # ========================================================
    # PAIRWISE COMPARISONS
    # ========================================================

    pairs = [

        (1, 2),

        (1, 3),

        (1, 4),

        (2, 3),

        (2, 4),

        (3, 4)
    ]

    pair_results = {}

    for pkg_a, pkg_b in pairs:

        result = compare_packages(

            indexes[pkg_a],

            indexes[pkg_b],

            pkg_a,

            pkg_b
        )

        pair_results[
            (pkg_a, pkg_b)
        ] = result

    # ========================================================
    # SUBSET ANALYSIS
    # ========================================================

    subset_analysis(

        reference_index=indexes[1],

        candidate_index=indexes[3],

        reference_pkg=1,

        candidate_pkg=3
    )

    subset_analysis(

        reference_index=indexes[1],

        candidate_index=indexes[4],

        reference_pkg=1,

        candidate_pkg=4
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "STAGE 0.4 AUDIT COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "What we are determining:"
    )

    print(
        "  - Are packages 1 and 2 annotation duplicates?"
    )

    print(
        "  - Is package 3 a subset of package 1?"
    )

    print(
        "  - Is package 4 a subset of package 1?"
    )

    print(
        "  - Are there conflicting annotations?"
    )

    print(
        "  - How many images actually have annotations?"
    )

    print()

    print(
        "DO NOT delete, merge, convert, resize, "
        "or train yet."
    )

    print()

    print(
        "Stage 0.4 determines the annotation version "
        "we should use."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()