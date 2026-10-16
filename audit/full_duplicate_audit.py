from pathlib import Path
import hashlib
from collections import defaultdict


# ============================================================
# STAGE 0.3.1
# FULL CROSS-PACKAGE IMAGE DUPLICATE AUDIT
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


PACKAGES = [
    "BeltCrack9kd1",
    "BeltCrack9kd2",
    "BeltCrack9kd3",
    "BeltCrack9kd4",
]


# ============================================================
# HASH FUNCTION
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):

    hasher = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_images(package_name):

    image_root = (
        DATASET_ROOT
        / package_name
        / "images"
    )

    images = []

    for path in image_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

    return sorted(
        images,
        key=lambda p: p.as_posix()
    )


# ============================================================
# GET RELATIVE IMAGE KEY
# ============================================================

def relative_key(
    package_name,
    image_path
):

    package_root = (
        DATASET_ROOT
        / package_name
    )

    return image_path.relative_to(
        package_root
    ).as_posix()


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print(
    "STAGE 0.3.1 - FULL CROSS-PACKAGE IMAGE DUPLICATE AUDIT"
)
print("=" * 70)

print()
print("Dataset:")
print(DATASET_ROOT)


if not DATASET_ROOT.exists():

    print()
    print("[ERROR] Dataset does not exist.")

    raise SystemExit(1)


# ============================================================
# LOAD ALL IMAGE PATHS
# ============================================================

package_images = {}


print()
print("=" * 70)
print("DISCOVERING IMAGES")
print("=" * 70)


for package_name in PACKAGES:

    images = get_images(
        package_name
    )

    package_images[
        package_name
    ] = images

    print()
    print(
        f"{package_name}: "
        f"{len(images):,} images"
    )


# ============================================================
# HASH ALL IMAGES
# ============================================================

package_hashes = {}


print()
print("=" * 70)
print("HASHING ALL IMAGES")
print("=" * 70)

print()
print(
    "This may take several minutes."
)

for package_name in PACKAGES:

    images = package_images[
        package_name
    ]

    hashes = {}

    print()
    print(
        f"Processing {package_name}..."
    )

    for index, image_path in enumerate(
        images,
        start=1
    ):

        try:

            file_hash = sha256_file(
                image_path
            )

            key = relative_key(
                package_name,
                image_path
            )

            hashes[
                key
            ] = file_hash

        except Exception as e:

            print()
            print(
                "[WARNING] Failed:"
            )

            print(
                image_path
            )

            print(e)

        if index % 2000 == 0:

            print(
                f"  Hashed "
                f"{index:,}/"
                f"{len(images):,}"
            )

    package_hashes[
        package_name
    ] = hashes

    print(
        f"Finished {package_name}"
    )


# ============================================================
# HASH FREQUENCY WITHIN EACH PACKAGE
# ============================================================

print()
print("=" * 70)
print("INTERNAL DUPLICATES")
print("=" * 70)


for package_name in PACKAGES:

    hashes = package_hashes[
        package_name
    ]

    reverse = defaultdict(list)

    for key, file_hash in hashes.items():

        reverse[
            file_hash
        ].append(key)

    duplicate_groups = [
        paths
        for paths in reverse.values()
        if len(paths) > 1
    ]

    duplicate_image_count = sum(
        len(group)
        for group in duplicate_groups
    )

    print()
    print(package_name)

    print(
        f"  Unique image hashes: "
        f"{len(reverse):,}"
    )

    print(
        f"  Duplicate hash groups: "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"  Images involved in duplicates: "
        f"{duplicate_image_count:,}"
    )


# ============================================================
# CROSS PACKAGE COMPARISON
# ============================================================

print()
print("=" * 70)
print("CROSS-PACKAGE IMAGE COMPARISON")
print("=" * 70)


for i in range(
    len(PACKAGES)
):

    for j in range(
        i + 1,
        len(PACKAGES)
    ):

        package_a = PACKAGES[i]
        package_b = PACKAGES[j]

        hashes_a = package_hashes[
            package_a
        ]

        hashes_b = package_hashes[
            package_b
        ]

        hash_set_a = set(
            hashes_a.values()
        )

        hash_set_b = set(
            hashes_b.values()
        )

        common_hashes = (
            hash_set_a
            &
            hash_set_b
        )

        # ----------------------------------------------------
        # Exact same relative path + same content
        # ----------------------------------------------------

        common_paths = set(
            hashes_a.keys()
        ) & set(
            hashes_b.keys()
        )

        same_path_same_content = 0
        same_path_different_content = 0

        for key in common_paths:

            if (
                hashes_a[key]
                ==
                hashes_b[key]
            ):

                same_path_same_content += 1

            else:

                same_path_different_content += 1

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print(
            f"{package_a} vs {package_b}"
        )

        print(
            f"  Images A: "
            f"{len(hashes_a):,}"
        )

        print(
            f"  Images B: "
            f"{len(hashes_b):,}"
        )

        print(
            f"  Common content hashes: "
            f"{len(common_hashes):,}"
        )

        print(
            f"  Same path + same content: "
            f"{same_path_same_content:,}"
        )

        print(
            f"  Same path + different content: "
            f"{same_path_different_content:,}"
        )

        if (
            len(hashes_a)
            ==
            len(hashes_b)
            ==
            len(common_hashes)
        ):

            print(
                "  VERDICT: "
                "ALL IMAGE CONTENT IS IDENTICAL"
            )

        elif len(common_hashes) > 0:

            print(
                "  VERDICT: "
                "PARTIAL IMAGE OVERLAP"
            )

        else:

            print(
                "  VERDICT: "
                "NO IMAGE CONTENT OVERLAP"
            )


# ============================================================
# EXACT DUPLICATE GROUPS
# ============================================================

print()
print("=" * 70)
print("GLOBAL CROSS-PACKAGE DUPLICATE GROUPS")
print("=" * 70)


global_hashes = defaultdict(
    list
)


for package_name in PACKAGES:

    for key, file_hash in (
        package_hashes[
            package_name
        ].items()
    ):

        global_hashes[
            file_hash
        ].append(
            (
                package_name,
                key
            )
        )


cross_package_groups = []

for file_hash, locations in (
    global_hashes.items()
):

    packages_present = set(
        package
        for package, _ in locations
    )

    if len(packages_present) > 1:

        cross_package_groups.append(
            (
                file_hash,
                locations
            )
        )


print()
print(
    f"Unique image hashes across "
    f"all four packages: "
    f"{len(global_hashes):,}"
)

print(
    f"Hashes appearing in multiple "
    f"packages: "
    f"{len(cross_package_groups):,}"
)


# ============================================================
# SAMPLE DUPLICATE LOCATIONS
# ============================================================

print()
print("=" * 70)
print("SAMPLE CROSS-PACKAGE DUPLICATE LOCATIONS")
print("=" * 70)


shown = 0

for file_hash, locations in (
    cross_package_groups
):

    if shown >= 20:
        break

    print()
    print(
        f"Hash: {file_hash}"
    )

    for package_name, key in locations:

        print(
            f"  {package_name}: "
            f"{key}"
        )

    shown += 1


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 0.3.1 COMPLETE")
print("=" * 70)

print()
print(
    "No dataset files were modified."
)

print()
print(
    "Do NOT delete or merge packages yet."
)

print(
    "Interpret the complete results first."
)