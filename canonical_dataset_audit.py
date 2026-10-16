from pathlib import Path
import json
import hashlib
import csv
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

PACKAGE = "BeltCrack9kd1"

OUTPUT_DIR = PROJECT_ROOT / "audit_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PATHS
# ============================================================

PACKAGE_ROOT = DATASET_ROOT / PACKAGE

IMAGE_ROOT = PACKAGE_ROOT / "images"

JSON_ROOT = PACKAGE_ROOT / "json"


# ============================================================
# HASH FUNCTION
# ============================================================

def file_hash(path, chunk_size=1024 * 1024):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# FIND IMAGE FOR JSON
# ============================================================

def corresponding_image(json_path):

    relative = json_path.relative_to(JSON_ROOT)

    # Example:
    #
    # train/data1/0.json
    #
    # becomes:
    #
    # train/data1/0.jpg

    return (
        IMAGE_ROOT
        / relative.parent
        / f"{relative.stem}.jpg"
    )


# ============================================================
# ANNOTATION SUMMARY
# ============================================================

def annotation_summary(json_path):

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        shapes = data.get("shapes", [])

        labels = Counter(
            str(shape.get("label", ""))
            for shape in shapes
        )

        return {
            "shape_count": len(shapes),
            "labels": "|".join(
                sorted(labels.keys())
            ),
            "width": data.get("imageWidth"),
            "height": data.get("imageHeight"),
        }

    except Exception as e:

        return {
            "shape_count": -1,
            "labels": "",
            "width": None,
            "height": None,
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STAGE 0.9 - CANONICAL DATASET & LEAKAGE AUDIT")
    print("=" * 80)

    print()
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Package      : {PACKAGE}")
    print(f"Image root   : {IMAGE_ROOT}")
    print(f"JSON root    : {JSON_ROOT}")

    # --------------------------------------------------------
    # Collect JSON files
    # --------------------------------------------------------

    json_files = sorted(
        JSON_ROOT.rglob("*.json")
    )

    print()
    print("-" * 80)
    print("ANNOTATION INVENTORY")
    print("-" * 80)

    print(
        f"JSON files found : {len(json_files):,}"
    )

    # --------------------------------------------------------
    # Split counters
    # --------------------------------------------------------

    split_counts = Counter()

    # --------------------------------------------------------
    # Image records
    # --------------------------------------------------------

    records = []

    missing_images = []

    hash_to_records = defaultdict(list)

    hash_errors = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for idx, json_path in enumerate(json_files, start=1):

        relative = json_path.relative_to(JSON_ROOT)

        parts = relative.parts

        if len(parts) < 3:

            continue

        split = parts[0]
        data_folder = parts[1]
        image_id = json_path.stem

        split_counts[split] += 1

        image_path = corresponding_image(
            json_path
        )

        annotation = annotation_summary(
            json_path
        )

        record = {
            "key":
                f"{split}/{data_folder}/{image_id}",

            "split": split,

            "data_folder": data_folder,

            "image_id": image_id,

            "json_path": str(json_path),

            "image_path": str(image_path),

            "exists": image_path.exists(),

            "sha256": None,

            "shape_count":
                annotation["shape_count"],

            "labels":
                annotation["labels"],

            "width":
                annotation["width"],

            "height":
                annotation["height"],
        }

        if not image_path.exists():

            missing_images.append(record)

        else:

            try:

                digest = file_hash(
                    image_path
                )

                record["sha256"] = digest

                hash_to_records[
                    digest
                ].append(record)

            except Exception as e:

                hash_errors.append({
                    "path": str(image_path),
                    "error": str(e)
                })

        records.append(record)

        if idx % 1000 == 0:

            print(
                f"Processed "
                f"{idx:,}/{len(json_files):,}"
            )

    # ========================================================
    # BASIC SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("BASIC SUMMARY")
    print("=" * 80)

    print(
        f"Total annotated images : "
        f"{len(records):,}"
    )

    print()

    print("Split distribution:")

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"  {split:8s}: {count:,}"
        )

    print()

    print(
        f"Missing corresponding images : "
        f"{len(missing_images):,}"
    )

    print(
        f"Image hashing errors           : "
        f"{len(hash_errors):,}"
    )

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    print()
    print("=" * 80)
    print("EXACT IMAGE DUPLICATE ANALYSIS")
    print("=" * 80)

    duplicate_groups = {
        digest: items
        for digest, items
        in hash_to_records.items()
        if len(items) > 1
    }

    duplicate_image_count = sum(
        len(items)
        for items in duplicate_groups.values()
    )

    duplicate_extra_count = sum(
        len(items) - 1
        for items in duplicate_groups.values()
    )

    print(
        f"Unique image hashes       : "
        f"{len(hash_to_records):,}"
    )

    print(
        f"Duplicate hash groups     : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images involved in groups : "
        f"{duplicate_image_count:,}"
    )

    print(
        f"Redundant image copies    : "
        f"{duplicate_extra_count:,}"
    )

    # ========================================================
    # CROSS-SPLIT DUPLICATES
    # ========================================================

    cross_split_groups = {}

    for digest, items in duplicate_groups.items():

        splits = set(
            item["split"]
            for item in items
        )

        if len(splits) > 1:

            cross_split_groups[digest] = items

    cross_split_images = sum(
        len(items)
        for items in cross_split_groups.values()
    )

    print()
    print("-" * 80)
    print("CROSS-SPLIT DUPLICATE LEAKAGE")
    print("-" * 80)

    print(
        f"Cross-split duplicate groups : "
        f"{len(cross_split_groups):,}"
    )

    print(
        f"Images involved              : "
        f"{cross_split_images:,}"
    )

    # ========================================================
    # CROSS-SPLIT DETAILS
    # ========================================================

    split_pair_counter = Counter()

    for digest, items in cross_split_groups.items():

        splits = sorted(
            set(
                item["split"]
                for item in items
            )
        )

        for i in range(len(splits)):

            for j in range(i + 1, len(splits)):

                pair = (
                    splits[i],
                    splits[j]
                )

                split_pair_counter[pair] += 1

    print()

    print("Cross-split duplicate group pairs:")

    if not split_pair_counter:

        print("  None")

    else:

        for pair, count in sorted(
            split_pair_counter.items()
        ):

            print(
                f"  {pair[0]} <-> {pair[1]} : "
                f"{count:,} groups"
            )

    # ========================================================
    # SEQUENCE-LEVEL DUPLICATES
    # ========================================================

    print()
    print("=" * 80)
    print("SEQUENCE-LEVEL DUPLICATE ANALYSIS")
    print("=" * 80)

    internal_sequence_groups = 0
    cross_sequence_groups = 0

    for digest, items in duplicate_groups.items():

        locations = set(
            (
                item["split"],
                item["data_folder"]
            )
            for item in items
        )

        if len(locations) == 1:

            internal_sequence_groups += 1

        else:

            cross_sequence_groups += 1

    print(
        f"Duplicate groups within one sequence : "
        f"{internal_sequence_groups:,}"
    )

    print(
        f"Duplicate groups across sequences     : "
        f"{cross_sequence_groups:,}"
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print("ANNOTATION DISTRIBUTION")
    print("=" * 80)

    label_counter = Counter()

    shape_counter = Counter()

    for record in records:

        if record["labels"]:

            for label in record["labels"].split("|"):

                label_counter[label] += 1

        shape_counter[
            record["shape_count"]
        ] += 1

    print()
    print("Labels:")

    for label, count in (
        label_counter.most_common()
    ):

        print(
            f"  {label:20s}: {count:,}"
        )

    print()
    print("Shapes per image:")

    for count, images in sorted(
        shape_counter.items()
    ):

        print(
            f"  {count:3d} : {images:,}"
        )

    # ========================================================
    # SAVE COMPLETE RECORD CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR /
        "stage_0_9_canonical_image_inventory.csv"
    )

    print()
    print("-" * 80)
    print("SAVING IMAGE INVENTORY")
    print("-" * 80)

    fieldnames = [
        "key",
        "split",
        "data_folder",
        "image_id",
        "json_path",
        "image_path",
        "exists",
        "sha256",
        "shape_count",
        "labels",
        "width",
        "height",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)

    print(
        f"Saved: {csv_path}"
    )

    # ========================================================
    # SAVE DUPLICATE GROUP REPORT
    # ========================================================

    duplicate_csv = (
        OUTPUT_DIR /
        "stage_0_9_duplicate_groups.csv"
    )

    duplicate_rows = []

    for digest, items in sorted(
        duplicate_groups.items()
    ):

        for item in items:

            duplicate_rows.append({
                "sha256": digest,
                "group_size": len(items),
                "cross_split":
                    len(
                        set(
                            x["split"]
                            for x in items
                        )
                    ) > 1,
                "split":
                    item["split"],
                "data_folder":
                    item["data_folder"],
                "image_id":
                    item["image_id"],
                "key":
                    item["key"],
                "image_path":
                    item["image_path"],
            })

    duplicate_fields = [
        "sha256",
        "group_size",
        "cross_split",
        "split",
        "data_folder",
        "image_id",
        "key",
        "image_path",
    ]

    with open(
        duplicate_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=duplicate_fields
        )

        writer.writeheader()

        writer.writerows(duplicate_rows)

    print(
        f"Saved: {duplicate_csv}"
    )

    # ========================================================
    # SAVE CROSS-SPLIT REPORT
    # ========================================================

    leakage_csv = (
        OUTPUT_DIR /
        "stage_0_9_cross_split_leakage.csv"
    )

    leakage_rows = []

    for digest, items in sorted(
        cross_split_groups.items()
    ):

        for item in items:

            leakage_rows.append({
                "sha256": digest,
                "group_size": len(items),
                "split": item["split"],
                "data_folder": item["data_folder"],
                "image_id": item["image_id"],
                "key": item["key"],
                "image_path": item["image_path"],
            })

    leakage_fields = [
        "sha256",
        "group_size",
        "split",
        "data_folder",
        "image_id",
        "key",
        "image_path",
    ]

    with open(
        leakage_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=leakage_fields
        )

        writer.writeheader()

        writer.writerows(leakage_rows)

    print(
        f"Saved: {leakage_csv}"
    )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================

    summary = {

        "stage": "0.9",

        "package": PACKAGE,

        "total_annotations":
            len(records),

        "split_counts":
            dict(split_counts),

        "missing_images":
            len(missing_images),

        "hash_errors":
            len(hash_errors),

        "unique_image_hashes":
            len(hash_to_records),

        "duplicate_groups":
            len(duplicate_groups),

        "duplicate_images":
            duplicate_image_count,

        "redundant_copies":
            duplicate_extra_count,

        "cross_split_duplicate_groups":
            len(cross_split_groups),

        "cross_split_duplicate_images":
            cross_split_images,

        "internal_sequence_duplicate_groups":
            internal_sequence_groups,

        "cross_sequence_duplicate_groups":
            cross_sequence_groups,

        "cross_split_pairs":
            {
                f"{a}_vs_{b}": count
                for (a, b), count
                in split_pair_counter.items()
            },
    }

    summary_path = (
        OUTPUT_DIR /
        "stage_0_9_canonical_dataset_audit.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"Saved: {summary_path}"
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 0.9 AUDIT COMPLETE")
    print("=" * 80)

    print()
    print("IMPORTANT:")
    print()
    print("NO DATASET FILES WERE COPIED OR DELETED.")
    print()
    print(
        "We will use this audit to determine the "
        "leakage-safe canonical dataset."
    )

    print()
    print(
        "Do NOT train yet."
    )


if __name__ == "__main__":
    main()