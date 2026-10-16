import os
import csv
import json
import hashlib
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# STAGE 0.9F
# FINAL CANONICAL DATASET VERIFICATION
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

DATASET_INFO_PATH = os.path.join(
    CANONICAL_DIR,
    "canonical_dataset_info.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage_0_9f_final_verification.json"
)


# ============================================================
# EXPECTED DATASET VALUES
# ============================================================

EXPECTED_TOTAL = 14087

EXPECTED_COUNTS = {
    "train": 9819,
    "val": 2102,
    "test": 2166
}

EXPECTED_SEQUENCES = 29
EXPECTED_COMPONENTS = 20


# ============================================================
# HELPERS
# ============================================================

def normalize_path(path):

    return os.path.normcase(
        os.path.normpath(path)
    )


def sha256_file(path, chunk_size=1024 * 1024):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


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

        if os.path.exists(path):
            return path

    # Relative to raw dataset
    candidate = os.path.normpath(
        os.path.join(
            RAW_DATASET,
            source_value
        )
    )

    if os.path.exists(candidate):
        return candidate

    # Relative to project
    candidate = os.path.normpath(
        os.path.join(
            PROJECT_ROOT,
            source_value
        )
    )

    if os.path.exists(candidate):
        return candidate

    return None


def resolve_canonical_json(
    split,
    filename
):

    if not filename:
        return None

    filename = str(
        filename
    ).strip()

    split = str(
        split
    ).strip().lower()

    # ========================================================
    # PRIMARY EXPECTED LOCATION
    # ========================================================

    candidate = os.path.join(
        CANONICAL_DIR,
        split,
        "labels",
        filename
    )

    if os.path.exists(candidate):

        return os.path.normpath(
            candidate
        )

    # ========================================================
    # Fallback: search within the split labels directory
    # ========================================================

    labels_dir = os.path.join(
        CANONICAL_DIR,
        split,
        "labels"
    )

    if os.path.isdir(labels_dir):

        candidate_name = os.path.basename(
            filename
        )

        for root, dirs, files in os.walk(
            labels_dir
        ):

            if candidate_name in files:

                return os.path.join(
                    root,
                    candidate_name
                )

    return None


def get_image_dimensions(path):

    if Image is None:
        return None

    try:

        with Image.open(path) as img:

            return (
                int(img.width),
                int(img.height)
            )

    except Exception:

        return None


def get_json_dimensions(data):

    width = data.get(
        "imageWidth"
    )

    height = data.get(
        "imageHeight"
    )

    if width is None or height is None:
        return None

    try:

        return (
            int(width),
            int(height)
        )

    except Exception:

        return None


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STAGE 0.9F - FINAL CANONICAL DATASET VERIFICATION")
print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results = {

    "stage": "0.9F",

    "status": "PASS",

    "manifest_records": 0,

    "expected_records": EXPECTED_TOTAL,

    "split_counts": {},

    "expected_split_counts": EXPECTED_COUNTS,

    "missing_canonical_json": 0,

    "invalid_json": 0,

    "non_crack_labels": 0,

    "missing_source_images": 0,

    "duplicate_source_paths": 0,

    "dimension_checked": 0,

    "dimension_match": 0,

    "dimension_mismatch": 0,

    "dimension_read_errors": 0,

    "filename_pairing_checked": 0,

    "filename_pairing_errors": 0,

    "sequence_count": 0,

    "sequence_split_conflicts": 0,

    "component_count": 0,

    "component_split_conflicts": 0,

    "unique_image_hashes": 0,

    "duplicate_hash_groups": 0,

    "cross_split_hash_groups": 0,

    "hash_errors": 0,

    "checks": {},

    "warnings": [],

    "errors": []
}


# ============================================================
# 1. BASIC PATH CHECK
# ============================================================

print("\n[1/12] Checking canonical dataset paths...")


if os.path.isdir(
    CANONICAL_DIR
):

    print(
        "Canonical dataset: FOUND"
    )

else:

    print(
        "Canonical dataset: MISSING"
    )

    results["status"] = "FAIL"

    results["errors"].append(
        "canonical_dataset directory is missing."
    )


if os.path.isfile(
    MANIFEST_PATH
):

    print(
        "Manifest: FOUND"
    )

else:

    print(
        "Manifest: MISSING"
    )

    results["status"] = "FAIL"

    results["errors"].append(
        "canonical_manifest.csv is missing."
    )


if os.path.isfile(
    DATASET_INFO_PATH
):

    print(
        "Dataset info: FOUND"
    )

else:

    print(
        "Dataset info: MISSING"
    )

    results["warnings"].append(
        "canonical_dataset_info.json is missing."
    )


results["checks"]["basic_paths"] = (
    os.path.isdir(CANONICAL_DIR)
    and os.path.isfile(MANIFEST_PATH)
)


if not os.path.isfile(
    MANIFEST_PATH
):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    raise SystemExit(1)


# ============================================================
# 2. LOAD MANIFEST
# ============================================================

print("\n[2/12] Loading canonical manifest...")


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


results[
    "manifest_records"
] = len(rows)


print(
    "Manifest records:",
    len(rows)
)


if len(rows) != EXPECTED_TOTAL:

    results["status"] = "FAIL"

    results["errors"].append(
        "Manifest does not contain exactly 14087 records."
    )


results["checks"][
    "manifest_count"
] = (
    len(rows) == EXPECTED_TOTAL
)


if not rows:

    results["status"] = "FAIL"

    results["errors"].append(
        "Manifest is empty."
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    raise SystemExit(1)


# ============================================================
# 3. VERIFY REQUIRED MANIFEST COLUMNS
# ============================================================

print("\n[3/12] Checking manifest columns...")


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


print(
    "Manifest columns:"
)


for column in actual_columns:

    print(
        "  -",
        column
    )


missing_columns = [

    column

    for column in required_columns

    if column not in actual_columns
]


if missing_columns:

    results["status"] = "FAIL"

    results["errors"].append(
        "Missing manifest columns: "
        + ", ".join(
            missing_columns
        )
    )


results[
    "checks"
][
    "required_columns"
] = (
    len(missing_columns) == 0
)


if missing_columns:

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    raise SystemExit(1)


# ============================================================
# 4. SPLIT COUNTS
# ============================================================

print("\n[4/12] Verifying split counts...")


split_counts = Counter()


for row in rows:

    split = str(
        row["split"]
    ).strip().lower()

    split_counts[
        split
    ] += 1


results[
    "split_counts"
] = dict(
    split_counts
)


for split in [
    "train",
    "val",
    "test"
]:

    print(
        split.upper() + ":",
        split_counts.get(
            split,
            0
        )
    )


split_ok = True


for split in [
    "train",
    "val",
    "test"
]:

    if (
        split_counts.get(
            split,
            0
        )
        != EXPECTED_COUNTS[
            split
        ]
    ):

        split_ok = False


if set(
    split_counts.keys()
) != {
    "train",
    "val",
    "test"
}:

    split_ok = False


results[
    "checks"
][
    "split_counts"
] = split_ok


if not split_ok:

    results["status"] = "FAIL"


# ============================================================
# 5. CANONICAL JSON VERIFICATION
# ============================================================

print(
    "\n[5/12] Verifying canonical annotation JSON files..."
)


sequence_splits = defaultdict(
    set
)

component_splits = defaultdict(
    set
)

label_examples = []
filename_pairing_examples = []


for index, row in enumerate(
    rows,
    start=1
):

    split = str(
        row["split"]
    ).strip().lower()


    canonical_filename = str(
        row["canonical_json_filename"]
    ).strip()


    # --------------------------------------------------------
    # Resolve canonical JSON correctly
    # --------------------------------------------------------

    label_path = resolve_canonical_json(
        split,
        canonical_filename
    )


    if (
        label_path is None
        or not os.path.isfile(
            label_path
        )
    ):

        results[
            "missing_canonical_json"
        ] += 1

        if len(
            label_examples
        ) < 20:

            label_examples.append(
                {
                    "split": split,
                    "filename": canonical_filename,
                    "expected_directory": os.path.join(
                        CANONICAL_DIR,
                        split,
                        "labels"
                    )
                }
            )

        continue


    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        results[
            "invalid_json"
        ] += 1

        if len(
            label_examples
        ) < 20:

            label_examples.append(
                {
                    "path": label_path,
                    "error": str(e)
                }
            )

        continue


    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    shapes = data.get(
        "shapes",
        []
    )


    labels = set()


    for shape in shapes:

        label = str(
            shape.get(
                "label",
                ""
            )
        ).strip().lower()

        if label:

            labels.add(
                label
            )


    if labels != {
        "crack"
    }:

        results[
            "non_crack_labels"
        ] += 1


    # --------------------------------------------------------
    # Verify canonical JSON filename
    # against manifest
    # --------------------------------------------------------

    actual_json_filename = os.path.basename(
        label_path
    )


    if (
        actual_json_filename
        != canonical_filename
    ):

        results[
            "filename_pairing_errors"
        ] += 1

        if len(
            filename_pairing_examples
        ) < 20:

            filename_pairing_examples.append(
                {
                    "manifest_filename":
                        canonical_filename,
                    "actual_filename":
                        actual_json_filename,
                    "path":
                        label_path
                }
            )


    results[
        "filename_pairing_checked"
    ] += 1


    # --------------------------------------------------------
    # Sequence split tracking
    # --------------------------------------------------------

    sequence = str(
        row["sequence"]
    ).strip()


    if sequence:

        sequence_splits[
            sequence
        ].add(
            split
        )


    # --------------------------------------------------------
    # Component split tracking
    # --------------------------------------------------------

    component = str(
        row["component"]
    ).strip()


    if component:

        component_splits[
            component
        ].add(
            split
        )


    if index % 1000 == 0:

        print(
            "  Verified JSON:",
            index,
            "/",
            len(rows)
        )


print(
    "Missing canonical JSON:",
    results[
        "missing_canonical_json"
    ]
)


print(
    "Invalid JSON:",
    results[
        "invalid_json"
    ]
)


print(
    "Non-crack label files:",
    results[
        "non_crack_labels"
    ]
)


print(
    "Filename pairing errors:",
    results[
        "filename_pairing_errors"
    ]
)


results[
    "checks"
][
    "canonical_json_existence"
] = (
    results[
        "missing_canonical_json"
    ] == 0
)


results[
    "checks"
][
    "json_parsing"
] = (
    results[
        "invalid_json"
    ] == 0
)


results[
    "checks"
][
    "labels"
] = (
    results[
        "non_crack_labels"
    ] == 0
)


results[
    "checks"
][
    "json_filename_pairing"
] = (
    results[
        "filename_pairing_errors"
    ] == 0
)


if (
    results[
        "missing_canonical_json"
    ] > 0
):

    results["status"] = "FAIL"


if (
    results[
        "invalid_json"
    ] > 0
):

    results["status"] = "FAIL"


if (
    results[
        "non_crack_labels"
    ] > 0
):

    results["status"] = "FAIL"


if (
    results[
        "filename_pairing_errors"
    ] > 0
):

    results["status"] = "FAIL"


if label_examples:

    results[
        "json_error_examples"
    ] = label_examples


if filename_pairing_examples:

    results[
        "filename_pairing_examples"
    ] = filename_pairing_examples


# ============================================================
# 6. SOURCE IMAGE EXISTENCE
# ============================================================

print(
    "\n[6/12] Verifying source images..."
)


source_paths = []

missing_source_examples = []


for row in rows:

    source_value = str(
        row["source_image"]
    ).strip()


    source_path = resolve_source_image(
        source_value
    )


    if (
        source_path is None
        or not os.path.isfile(
            source_path
        )
    ):

        results[
            "missing_source_images"
        ] += 1

        source_paths.append(
            ""
        )

        if len(
            missing_source_examples
        ) < 20:

            missing_source_examples.append(
                source_value
            )

    else:

        source_paths.append(
            normalize_path(
                source_path
            )
        )


print(
    "Missing source images:",
    results[
        "missing_source_images"
    ]
)


results[
    "checks"
][
    "source_images"
] = (
    results[
        "missing_source_images"
    ] == 0
)


if (
    results[
        "missing_source_images"
    ] > 0
):

    results["status"] = "FAIL"

    results[
        "missing_source_examples"
    ] = missing_source_examples


# ============================================================
# 7. SOURCE PATH UNIQUENESS
# ============================================================

print(
    "\n[7/12] Checking duplicate source-image references..."
)


source_counter = Counter(
    path
    for path in source_paths
    if path
)


duplicate_source_paths = {

    path: count

    for path, count
    in source_counter.items()

    if count > 1
}


results[
    "duplicate_source_paths"
] = len(
    duplicate_source_paths
)


print(
    "Duplicate source paths:",
    len(
        duplicate_source_paths
    )
)


results[
    "checks"
][
    "unique_source_paths"
] = (
    len(
        duplicate_source_paths
    ) == 0
)


if duplicate_source_paths:

    results["status"] = "FAIL"


# ============================================================
# 8. DIMENSION VERIFICATION
# ============================================================

print(
    "\n[8/12] Verifying annotation dimensions against source images..."
)


if Image is None:

    print(
        "WARNING: Pillow is not installed."
    )

    results[
        "checks"
    ][
        "dimensions"
    ] = "SKIPPED"

    results[
        "warnings"
    ].append(
        "Pillow is not installed."
    )

else:

    for index, row in enumerate(
        rows,
        start=1
    ):

        split = str(
            row["split"]
        ).strip().lower()


        canonical_filename = str(
            row["canonical_json_filename"]
        ).strip()


        label_path = resolve_canonical_json(
            split,
            canonical_filename
        )


        source_path = resolve_source_image(
            str(
                row["source_image"]
            ).strip()
        )


        if (
            label_path is None
            or source_path is None
        ):

            continue


        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception:

            continue


        json_dimensions = get_json_dimensions(
            data
        )


        actual_dimensions = get_image_dimensions(
            source_path
        )


        if (
            json_dimensions is None
            or actual_dimensions is None
        ):

            results[
                "dimension_read_errors"
            ] += 1

            continue


        results[
            "dimension_checked"
        ] += 1


        if (
            json_dimensions
            == actual_dimensions
        ):

            results[
                "dimension_match"
            ] += 1

        else:

            results[
                "dimension_mismatch"
            ] += 1


        if index % 1000 == 0:

            print(
                "  Dimensions checked:",
                index,
                "/",
                len(rows)
            )


print(
    "Dimensions checked:",
    results[
        "dimension_checked"
    ]
)


print(
    "Dimension matches:",
    results[
        "dimension_match"
    ]
)


print(
    "Dimension mismatches:",
    results[
        "dimension_mismatch"
    ]
)


print(
    "Dimension read errors:",
    results[
        "dimension_read_errors"
    ]
)


if Image is not None:

    results[
        "checks"
    ][
        "dimensions"
    ] = (
        results[
            "dimension_mismatch"
        ] == 0
        and results[
            "dimension_checked"
        ] == EXPECTED_TOTAL
    )


    if not results[
        "checks"
    ][
        "dimensions"
    ]:

        results["status"] = "FAIL"


# ============================================================
# 9. SEQUENCE INTEGRITY
# ============================================================

print(
    "\n[9/12] Verifying sequence integrity..."
)


results[
    "sequence_count"
] = len(
    sequence_splits
)


sequence_conflicts = {

    sequence: sorted(
        splits
    )

    for sequence, splits
    in sequence_splits.items()

    if len(splits) > 1
}


results[
    "sequence_split_conflicts"
] = len(
    sequence_conflicts
)


print(
    "Sequences:",
    results[
        "sequence_count"
    ]
)


print(
    "Expected sequences:",
    EXPECTED_SEQUENCES
)


print(
    "Sequence split conflicts:",
    results[
        "sequence_split_conflicts"
    ]
)


results[
    "checks"
][
    "sequence_count"
] = (
    results[
        "sequence_count"
    ] == EXPECTED_SEQUENCES
)


results[
    "checks"
][
    "sequence_integrity"
] = (
    results[
        "sequence_split_conflicts"
    ] == 0
)


if (
    results[
        "sequence_count"
    ] != EXPECTED_SEQUENCES
):

    results[
        "warnings"
    ].append(
        "Expected 29 sequences."
    )


if sequence_conflicts:

    results["status"] = "FAIL"


# ============================================================
# 10. COMPONENT INTEGRITY
# ============================================================

print(
    "\n[10/12] Verifying component integrity..."
)


results[
    "component_count"
] = len(
    component_splits
)


component_conflicts = {

    component: sorted(
        splits
    )

    for component, splits
    in component_splits.items()

    if len(splits) > 1
}


results[
    "component_split_conflicts"
] = len(
    component_conflicts
)


print(
    "Components:",
    results[
        "component_count"
    ]
)


print(
    "Expected components:",
    EXPECTED_COMPONENTS
)


print(
    "Component split conflicts:",
    results[
        "component_split_conflicts"
    ]
)


results[
    "checks"
][
    "component_count"
] = (
    results[
        "component_count"
    ] == EXPECTED_COMPONENTS
)


results[
    "checks"
][
    "component_integrity"
] = (
    results[
        "component_split_conflicts"
    ] == 0
)


if component_conflicts:

    results["status"] = "FAIL"


# ============================================================
# 11. IMAGE CONTENT DUPLICATE / LEAKAGE CHECK
# ============================================================

print(
    "\n[11/12] Checking image-content duplicates across splits..."
)


hash_to_splits = defaultdict(
    set
)

hash_to_count = Counter()


for index, row in enumerate(
    rows,
    start=1
):

    source_path = resolve_source_image(
        str(
            row["source_image"]
        ).strip()
    )


    if (
        source_path is None
        or not os.path.isfile(
            source_path
        )
    ):

        continue


    try:

        image_hash = sha256_file(
            source_path
        )

    except Exception:

        results[
            "hash_errors"
        ] += 1

        continue


    split = str(
        row["split"]
    ).strip().lower()


    hash_to_splits[
        image_hash
    ].add(
        split
    )


    hash_to_count[
        image_hash
    ] += 1


    if index % 500 == 0:

        print(
            "  Hashed:",
            index,
            "/",
            len(rows)
        )


results[
    "unique_image_hashes"
] = len(
    hash_to_count
)


duplicate_hash_groups = {

    image_hash: count

    for image_hash, count
    in hash_to_count.items()

    if count > 1
}


cross_split_hash_groups = {

    image_hash: sorted(
        hash_to_splits[
            image_hash
        ]
    )

    for image_hash
    in hash_to_splits

    if len(
        hash_to_splits[
            image_hash
        ]
    ) > 1
}


results[
    "duplicate_hash_groups"
] = len(
    duplicate_hash_groups
)


results[
    "cross_split_hash_groups"
] = len(
    cross_split_hash_groups
)


print(
    "Unique image hashes:",
    results[
        "unique_image_hashes"
    ]
)


print(
    "Duplicate hash groups:",
    results[
        "duplicate_hash_groups"
    ]
)


print(
    "Cross-split hash groups:",
    results[
        "cross_split_hash_groups"
    ]
)


print(
    "Hash errors:",
    results[
        "hash_errors"
    ]
)


results[
    "checks"
][
    "duplicate_isolation"
] = (
    results[
        "cross_split_hash_groups"
    ] == 0
)


if (
    results[
        "cross_split_hash_groups"
    ] > 0
):

    results["status"] = "FAIL"


# ============================================================
# 12. METADATA CHECK
# ============================================================

print(
    "\n[12/12] Checking canonical dataset metadata..."
)


if os.path.isfile(
    DATASET_INFO_PATH
):

    try:

        with open(
            DATASET_INFO_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            dataset_info = json.load(
                f
            )


        print(
            "Dataset metadata: readable"
        )


        results[
            "checks"
        ][
            "dataset_metadata"
        ] = True


    except Exception as e:

        print(
            "Dataset metadata: ERROR"
        )


        results[
            "checks"
        ][
            "dataset_metadata"
        ] = False


        results[
            "warnings"
        ].append(
            str(e)
        )

else:

    results[
        "checks"
    ][
        "dataset_metadata"
    ] = False


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 0.9F FINAL VERIFICATION SUMMARY"
)

print(
    "=" * 70
)


for name, value in results[
    "checks"
].items():

    print(
        f"{name}: {value}"
    )


# ============================================================
# HARD FAILURE CONDITIONS
# ============================================================

hard_fail_checks = [

    "basic_paths",

    "manifest_count",

    "required_columns",

    "split_counts",

    "canonical_json_existence",

    "json_parsing",

    "labels",

    "json_filename_pairing",

    "source_images",

    "unique_source_paths",

    "dimensions",

    "sequence_integrity",

    "component_integrity",

    "duplicate_isolation"
]


for check in hard_fail_checks:

    value = results[
        "checks"
    ].get(
        check
    )


    if value is False:

        results[
            "status"
        ] = "FAIL"


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    "\nFinal status:",
    results[
        "status"
    ]
)


print(
    "\nVerification report:"
)

print(
    OUTPUT_PATH
)


print(
    "\n" + "=" * 70
)


if results[
    "status"
] == "PASS":

    print(
        "STAGE 0.9F COMPLETE - FINAL VERIFICATION PASSED"
    )

else:

    print(
        "STAGE 0.9F FAILED - DO NOT START STAGE 1.0"
    )


print(
    "=" * 70
)