from pathlib import Path
import hashlib
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

PACKAGE = 1

PACKAGE_ROOT = (
    DATASET_ROOT /
    f"BeltCrack9kd{PACKAGE}"
)

IMAGE_ROOT = (
    PACKAGE_ROOT /
    "images"
)

JSON_ROOT = (
    PACKAGE_ROOT /
    "json"
)


# ============================================================
# HASH FUNCTION
# ============================================================

def calculate_sha256(path, chunk_size=1024 * 1024):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# FIND IMAGE FOR ANNOTATION
# ============================================================

def annotation_to_image_path(json_path):

    relative = (
        json_path
        .relative_to(JSON_ROOT)
    )

    # Example:
    #
    # train/data15/0.json
    #
    # becomes:
    #
    # train/data15/0.jpg

    image_relative = relative.with_suffix(
        ".jpg"
    )

    image_path = (
        IMAGE_ROOT /
        image_relative
    )

    return image_path


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)

    print(
        "BELTCRACK9KD - STAGE 0.5"
    )

    print(
        "LEAKAGE-SAFE DUPLICATE GROUP AUDIT"
    )

    print("=" * 70)

    print()

    print(
        f"Dataset root:"
    )

    print(
        DATASET_ROOT
    )

    print()

    print(
        f"Package:"
    )

    print(
        f"BeltCrack9kd{PACKAGE}"
    )

    print()

    print(
        "This audit is READ-ONLY."
    )

    print(
        "No dataset files will be modified."
    )

    # ========================================================
    # VERIFY DIRECTORIES
    # ========================================================

    if not JSON_ROOT.exists():

        print()

        print(
            f"ERROR: JSON directory not found:"
        )

        print(
            JSON_ROOT
        )

        return

    if not IMAGE_ROOT.exists():

        print()

        print(
            f"ERROR: image directory not found:"
        )

        print(
            IMAGE_ROOT
        )

        return

    # ========================================================
    # FIND ANNOTATIONS
    # ========================================================

    json_files = sorted(
        JSON_ROOT.rglob("*.json")
    )

    print()

    print(
        f"Annotation files found: "
        f"{len(json_files):,}"
    )

    # ========================================================
    # DATA STRUCTURES
    # ========================================================

    hash_to_images = defaultdict(list)

    missing_images = []

    hash_errors = []

    split_counts = Counter()

    processed = 0

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    print()

    print(
        "Hashing annotated images..."
    )

    print(
        "This may take some time because "
        "the images are high resolution."
    )

    print()

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        image_path = annotation_to_image_path(
            json_path
        )

        # ----------------------------------------------------
        # Determine split
        # ----------------------------------------------------

        relative_json = (
            json_path
            .relative_to(JSON_ROOT)
        )

        parts = relative_json.parts

        if len(parts) >= 1:

            split = parts[0]

        else:

            split = "unknown"

        split_counts[split] += 1

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if not image_path.exists():

            missing_images.append(
                {
                    "json": json_path,
                    "image": image_path
                }
            )

            continue

        # ----------------------------------------------------
        # Hash image
        # ----------------------------------------------------

        try:

            image_hash = calculate_sha256(
                image_path
            )

        except Exception as e:

            hash_errors.append(
                {
                    "image": image_path,
                    "error": str(e)
                }
            )

            continue

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        hash_to_images[
            image_hash
        ].append(
            {
                "image": image_path,
                "json": json_path,
                "split": split,
                "relative_image": (
                    image_path
                    .relative_to(PACKAGE_ROOT)
                    .as_posix()
                )
            }
        )

        processed += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 500 == 0
            or
            index == len(json_files)
        ):

            print(
                f"  Processed "
                f"{index:,}/"
                f"{len(json_files):,}"
            )

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "BASIC HASHING RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Annotations found       : "
        f"{len(json_files):,}"
    )

    print(
        f"Images successfully hash: "
        f"{processed:,}"
    )

    print(
        f"Missing images          : "
        f"{len(missing_images):,}"
    )

    print(
        f"Hash errors             : "
        f"{len(hash_errors):,}"
    )

    print()

    print(
        "Original annotation split:"
    )

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"  {split:8s}: {count:,}"
        )

    # ========================================================
    # UNIQUE CONTENT
    # ========================================================

    unique_hash_count = len(
        hash_to_images
    )

    total_images = processed

    duplicate_groups = {

        image_hash: entries

        for image_hash, entries
        in hash_to_images.items()

        if len(entries) > 1
    }

    duplicate_group_count = len(
        duplicate_groups
    )

    duplicate_images_involved = sum(
        len(entries)
        for entries
        in duplicate_groups.values()
    )

    duplicate_extra_copies = (
        duplicate_images_involved
        -
        duplicate_group_count
    )

    # ========================================================
    # DUPLICATE RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "IMAGE CONTENT UNIQUENESS"
    )

    print("=" * 70)

    print()

    print(
        f"Total images hashed       : "
        f"{total_images:,}"
    )

    print(
        f"Unique image contents     : "
        f"{unique_hash_count:,}"
    )

    print(
        f"Duplicate content groups  : "
        f"{duplicate_group_count:,}"
    )

    print(
        f"Images involved in groups : "
        f"{duplicate_images_involved:,}"
    )

    print(
        f"Extra duplicate copies    : "
        f"{duplicate_extra_copies:,}"
    )

    if total_images > 0:

        duplicate_percentage = (
            duplicate_images_involved
            /
            total_images
            *
            100
        )

        print()

        print(
            f"Images participating in "
            f"duplicate groups: "
            f"{duplicate_percentage:.2f}%"
        )

    # ========================================================
    # LEAKAGE ANALYSIS
    # ========================================================

    leakage_groups = {}

    train_val_groups = {}

    train_test_groups = {}

    val_test_groups = {}

    same_split_groups = {}

    for image_hash, entries in (
        duplicate_groups.items()
    ):

        splits = set(
            entry["split"]
            for entry in entries
        )

        if len(splits) > 1:

            leakage_groups[
                image_hash
            ] = entries

        if (
            "train" in splits
            and
            "val" in splits
        ):

            train_val_groups[
                image_hash
            ] = entries

        if (
            "train" in splits
            and
            "test" in splits
        ):

            train_test_groups[
                image_hash
            ] = entries

        if (
            "val" in splits
            and
            "test" in splits
        ):

            val_test_groups[
                image_hash
            ] = entries

        if len(splits) == 1:

            same_split_groups[
                image_hash
            ] = entries

    # ========================================================
    # LEAKAGE RESULTS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TRAIN / VAL / TEST LEAKAGE ANALYSIS"
    )

    print("=" * 70)

    print()

    print(
        f"Duplicate groups crossing splits : "
        f"{len(leakage_groups):,}"
    )

    print()

    print(
        f"TRAIN <-> VAL leakage groups     : "
        f"{len(train_val_groups):,}"
    )

    print(
        f"TRAIN <-> TEST leakage groups    : "
        f"{len(train_test_groups):,}"
    )

    print(
        f"VAL <-> TEST leakage groups      : "
        f"{len(val_test_groups):,}"
    )

    print()

    print(
        f"Duplicate groups contained "
        f"within one split              : "
        f"{len(same_split_groups):,}"
    )

    # ========================================================
    # COUNT LEAKED IMAGES
    # ========================================================

    leaked_images = 0

    train_val_images = 0
    train_test_images = 0
    val_test_images = 0

    for entries in leakage_groups.values():

        leaked_images += len(entries)

    for entries in train_val_groups.values():

        train_val_images += len(entries)

    for entries in train_test_groups.values():

        train_test_images += len(entries)

    for entries in val_test_groups.values():

        val_test_images += len(entries)

    print()

    print(
        f"Images inside cross-split "
        f"duplicate groups: "
        f"{leaked_images:,}"
    )

    print()

    print(
        f"Images in TRAIN/VAL groups : "
        f"{train_val_images:,}"
    )

    print(
        f"Images in TRAIN/TEST groups: "
        f"{train_test_images:,}"
    )

    print(
        f"Images in VAL/TEST groups  : "
        f"{val_test_images:,}"
    )

    # ========================================================
    # SHOW EXAMPLES
    # ========================================================

    print()

    print("=" * 70)

    print(
        "EXAMPLES OF CROSS-SPLIT DUPLICATES"
    )

    print("=" * 70)

    if not leakage_groups:

        print()

        print(
            "NO CROSS-SPLIT DUPLICATE GROUPS FOUND."
        )

    else:

        example_count = 0

        for image_hash, entries in (
            leakage_groups.items()
        ):

            print()

            print(
                f"Hash: {image_hash}"
            )

            print(
                f"Copies: {len(entries)}"
            )

            for entry in entries:

                print(
                    f"  [{entry['split']:5s}] "
                    f"{entry['relative_image']}"
                )

            example_count += 1

            if example_count >= 15:
                break

    # ========================================================
    # DUPLICATE GROUP SIZE DISTRIBUTION
    # ========================================================

    group_size_distribution = Counter()

    for entries in duplicate_groups.values():

        group_size_distribution[
            len(entries)
        ] += 1

    print()

    print("=" * 70)

    print(
        "DUPLICATE GROUP SIZE DISTRIBUTION"
    )

    print("=" * 70)

    print()

    for group_size, count in sorted(
        group_size_distribution.items()
    ):

        print(
            f"  {group_size:3d} identical copies: "
            f"{count:,} group(s)"
        )

    # ========================================================
    # TOP DUPLICATE GROUPS
    # ========================================================

    largest_groups = sorted(

        duplicate_groups.items(),

        key=lambda item: len(
            item[1]
        ),

        reverse=True
    )

    print()

    print("=" * 70)

    print(
        "LARGEST DUPLICATE GROUPS"
    )

    print("=" * 70)

    print()

    for image_hash, entries in (
        largest_groups[:10]
    ):

        split_counter = Counter(
            entry["split"]
            for entry in entries
        )

        print(
            f"Copies: {len(entries):3d} | "
            f"Splits: {dict(split_counter)} | "
            f"Hash: {image_hash[:16]}..."
        )

    # ========================================================
    # MISSING IMAGE EXAMPLES
    # ========================================================

    if missing_images:

        print()

        print("=" * 70)

        print(
            "MISSING IMAGE EXAMPLES"
        )

        print("=" * 70)

        print()

        for item in missing_images[:10]:

            print(
                f"JSON : "
                f"{item['json']}"
            )

            print(
                f"IMAGE: "
                f"{item['image']}"
            )

            print()

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "STAGE 0.5 AUDIT COMPLETE"
    )

    print("=" * 70)

    print()

    if len(leakage_groups) > 0:

        print(
            "RESULT: DATA LEAKAGE EXISTS."
        )

        print()

        print(
            "The existing train/val/test split "
            "cannot be used directly."
        )

        print()

        print(
            "Identical image-content groups must "
            "be kept inside a single split."
        )

    else:

        print(
            "RESULT: NO CROSS-SPLIT IMAGE "
            "LEAKAGE DETECTED."
        )

        print()

        print(
            "The existing split is content-safe "
            "with respect to exact image duplicates."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This stage does NOT create a new split."
    )

    print(
        "It only measures the leakage."
    )

    print()

    print(
        "DO NOT modify the dataset yet."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()