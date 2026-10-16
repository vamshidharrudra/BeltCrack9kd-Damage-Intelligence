from pathlib import Path
import hashlib
import json
from collections import defaultdict


# ============================================================
# STAGE 0.3
# PACKAGE AND DUPLICATE AUDIT
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    """
    Calculate SHA256 hash of a file.
    """
    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def get_package_name(path):
    """
    Return BeltCrack9kd1 ... BeltCrack9kd13.
    """

    for part in path.parts:

        if (
            part.startswith("BeltCrack9kd")
            and part != "BeltCrack9kd"
        ):

            return part

    return "UNKNOWN"


def get_relative_image_key(image_path):
    """
    Return image location relative to package.

    Example:

    images/val/data19/639.jpg
    """

    package_name = get_package_name(
        image_path
    )

    package_root = (
        DATASET_ROOT
        / package_name
    )

    try:

        return image_path.relative_to(
            package_root
        ).as_posix()

    except ValueError:

        return image_path.name


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 0.3 - PACKAGE AND DUPLICATE AUDIT")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_ROOT)


if not DATASET_ROOT.exists():

    print()
    print("[ERROR] Dataset root does not exist.")
    raise SystemExit(1)


# ============================================================
# FIND PACKAGES
# ============================================================

packages = []

for path in DATASET_ROOT.iterdir():

    if not path.is_dir():
        continue

    if (
        path.name.startswith("BeltCrack9kd")
        and path.name != "BeltCrack9kd"
    ):

        packages.append(path)


packages = sorted(
    packages,
    key=lambda p: (
        len(p.name),
        p.name
    )
)


print()
print(
    f"Packages found: {len(packages)}"
)

for package in packages:

    print(
        "  "
        + package.name
    )


# ============================================================
# PACKAGE STATISTICS
# ============================================================

package_stats = {}


for package in packages:

    package_name = package.name

    image_files = []

    for path in (
        package / "images"
    ).rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            image_files.append(path)


    json_files = list(
        (
            package / "json"
        ).rglob("*.json")
    ) if (
        package / "json"
    ).exists() else []


    package_stats[
        package_name
    ] = {
        "images": image_files,
        "json": json_files,
    }


# ============================================================
# PACKAGE SUMMARY
# ============================================================

print()
print("=" * 70)
print("PACKAGE SUMMARY")
print("=" * 70)


for package_name in sorted(
    package_stats
):

    stats = package_stats[
        package_name
    ]

    print()
    print(package_name)

    print(
        f"  Images: {len(stats['images']):,}"
    )

    print(
        f"  JSON:   {len(stats['json']):,}"
    )


# ============================================================
# ANNOTATED PACKAGES
# ============================================================

annotated_packages = []

image_only_packages = []


for package_name, stats in package_stats.items():

    if len(stats["json"]) > 0:

        annotated_packages.append(
            package_name
        )

    elif len(stats["images"]) > 0:

        image_only_packages.append(
            package_name
        )


print()
print("=" * 70)
print("ANNOTATION STATUS")
print("=" * 70)

print()
print(
    "Packages with JSON annotations:"
)

for package_name in sorted(
    annotated_packages
):

    print(
        f"  {package_name}"
    )


print()
print(
    "Image-only packages:"
)

for package_name in sorted(
    image_only_packages
):

    print(
        f"  {package_name}"
    )


# ============================================================
# EXACT IMAGE COUNT COMPARISON
# ============================================================

print()
print("=" * 70)
print("IMAGE COUNT COMPARISON")
print("=" * 70)


count_groups = defaultdict(list)


for package_name, stats in package_stats.items():

    count_groups[
        len(stats["images"])
    ].append(
        package_name
    )


for count, package_names in sorted(
    count_groups.items()
):

    if len(package_names) > 1:

        print()
        print(
            f"{count:,} images:"
        )

        for package_name in sorted(
            package_names
        ):

            print(
                f"  {package_name}"
            )


# ============================================================
# JSON COUNT COMPARISON
# ============================================================

print()
print("=" * 70)
print("JSON COUNT COMPARISON")
print("=" * 70)


json_count_groups = defaultdict(list)


for package_name, stats in package_stats.items():

    json_count_groups[
        len(stats["json"])
    ].append(
        package_name
    )


for count, package_names in sorted(
    json_count_groups.items()
):

    if count > 0 and len(package_names) > 1:

        print()
        print(
            f"{count:,} JSON files:"
        )

        for package_name in sorted(
            package_names
        ):

            print(
                f"  {package_name}"
            )


# ============================================================
# HASH SAMPLE
#
# We do NOT hash every image initially.
# We first compare a controlled sample.
# ============================================================

print()
print("=" * 70)
print("CROSS-PACKAGE IMAGE DUPLICATE SAMPLE")
print("=" * 70)

print()
print(
    "Hashing up to 20 images from each annotated package..."
)


HASH_SAMPLE_SIZE = 20


package_hashes = {}


for package_name in sorted(
    annotated_packages
):

    images = package_stats[
        package_name
    ]["images"]

    images = sorted(
        images,
        key=lambda p: p.as_posix()
    )

    sample = images[
        :HASH_SAMPLE_SIZE
    ]

    hashes = {}

    for image_path in sample:

        try:

            file_hash = sha256_file(
                image_path
            )

            hashes[
                get_relative_image_key(
                    image_path
                )
            ] = file_hash

        except Exception as e:

            print()
            print(
                "[WARNING] Could not hash:"
            )

            print(
                image_path
            )

            print(e)


    package_hashes[
        package_name
    ] = hashes


# ============================================================
# COMPARE HASHES
# ============================================================

print()
print("=" * 70)
print("SAMPLE HASH COMPARISON")
print("=" * 70)


package_names = sorted(
    package_hashes
)


for i in range(
    len(package_names)
):

    for j in range(
        i + 1,
        len(package_names)
    ):

        package_a = package_names[i]
        package_b = package_names[j]

        hashes_a = package_hashes[
            package_a
        ]

        hashes_b = package_hashes[
            package_b
        ]

        common_hashes = (
            set(hashes_a.values())
            &
            set(hashes_b.values())
        )

        print()
        print(
            f"{package_a} vs {package_b}"
        )

        print(
            f"  Common sample hashes: "
            f"{len(common_hashes)}"
        )


# ============================================================
# JSON ANNOTATION SAMPLE COMPARISON
# ============================================================

print()
print("=" * 70)
print("ANNOTATION SAMPLE COMPARISON")
print("=" * 70)


def annotation_signature(json_path):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:

        return None

    shapes = data.get(
        "shapes",
        []
    )

    signature = []

    for shape in shapes:

        if not isinstance(
            shape,
            dict
        ):

            continue

        label = shape.get(
            "label",
            ""
        )

        shape_type = shape.get(
            "shape_type",
            ""
        )

        points = shape.get(
            "points",
            []
        )

        signature.append(
            (
                label,
                shape_type,
                points
            )
        )

    return (
        data.get("imageWidth"),
        data.get("imageHeight"),
        tuple(
            signature
        )
    )


annotation_samples = {}


for package_name in sorted(
    annotated_packages
):

    json_files = package_stats[
        package_name
    ]["json"]

    json_files = sorted(
        json_files,
        key=lambda p: p.as_posix()
    )

    sample = json_files[
        :20
    ]

    signatures = []

    for json_path in sample:

        signature = annotation_signature(
            json_path
        )

        signatures.append(
            signature
        )

    annotation_samples[
        package_name
    ] = signatures


for i in range(
    len(annotated_packages)
):

    for j in range(
        i + 1,
        len(annotated_packages)
    ):

        package_a = sorted(
            annotated_packages
        )[i]

        package_b = sorted(
            annotated_packages
        )[j]

        signatures_a = annotation_samples[
            package_a
        ]

        signatures_b = annotation_samples[
            package_b
        ]

        same_count = 0

        for sig_a in signatures_a:

            if sig_a in signatures_b:

                same_count += 1

        print()
        print(
            f"{package_a} vs {package_b}"
        )

        print(
            f"  Matching annotation "
            f"signatures in sample: "
            f"{same_count}/20"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 0.3 COMPLETE")
print("=" * 70)

print()
print(
    "No dataset files were modified."
)

print()
print(
    "IMPORTANT:"
)

print(
    "The hash comparison above is only a SAMPLE."
)

print(
    "We will NOT delete or exclude any package "
    "until the results are interpreted."
)