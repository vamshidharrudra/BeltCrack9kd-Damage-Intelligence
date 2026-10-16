import os
import json
import csv
import shutil
from collections import defaultdict

# ============================================================
# STAGE 0.9E
# CANONICAL DATASET CONSTRUCTION
# ============================================================
#
# DISK-SAFE VERSION
#
# IMPORTANT:
#
# The previous implementation attempted to duplicate all
# full-resolution images. This can require a very large amount
# of additional disk space.
#
# This version creates a MANIFEST-BASED canonical dataset:
#
#   - JSON annotations are copied into canonical_dataset
#   - Original images remain in the raw dataset
#   - Exact source image paths are stored in the manifest
#   - No raw images are duplicated
#   - No raw files are modified/deleted/moved
#
# This is still a reproducible canonical dataset because the
# manifest defines exactly which image + annotation belongs to
# train/val/test.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd_Damage_Intelligence"
)

RAW_DATASET = (
    r"C:\Users\vamsh\Downloads"
    r"\BeltCrack9kd"
)

PACKAGE_1 = os.path.join(
    RAW_DATASET,
    "BeltCrack9kd1"
)

AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "audit_outputs"
)

CANONICAL_DIR = os.path.join(
    PROJECT_ROOT,
    "canonical_dataset"
)

INVENTORY_CSV = os.path.join(
    AUDIT_DIR,
    "stage_0_9_canonical_image_inventory.csv"
)

COMPONENT_JSON = os.path.join(
    AUDIT_DIR,
    "stage_0_9b_corrected_components.json"
)

OPTIMIZED_PLAN_JSON = os.path.join(
    AUDIT_DIR,
    "stage_0_9d1_optimized_split_plan.json"
)

VERIFICATION_JSON = os.path.join(
    AUDIT_DIR,
    "stage_0_9d2_verification.json"
)

MANIFEST_CSV = os.path.join(
    CANONICAL_DIR,
    "canonical_manifest.csv"
)

DATASET_INFO_JSON = os.path.join(
    CANONICAL_DIR,
    "canonical_dataset_info.json"
)

CONSTRUCTION_REPORT_JSON = os.path.join(
    AUDIT_DIR,
    "stage_0_9e_construction_report.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("STAGE 0.9E - CANONICAL DATASET CONSTRUCTION")
print("=" * 72)

print()
print("DISK-SAFE MANIFEST-BASED VERSION")
print()
print("The raw images will NOT be duplicated.")
print("Only annotation JSON files will be copied.")
print("The original BeltCrack9kd dataset will NOT be modified.")
print()


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

print("Performing safety checks...")

required_paths = [
    RAW_DATASET,
    PACKAGE_1,
    INVENTORY_CSV,
    COMPONENT_JSON,
    OPTIMIZED_PLAN_JSON,
    VERIFICATION_JSON
]

for path in required_paths:

    if not os.path.exists(path):

        print()
        print("ERROR: Required path/file not found:")
        print("  " + path)

        raise SystemExit(1)


print("  Raw dataset found.")
print("  Package 1 found.")
print("  Stage 0.9 inventory found.")
print("  Stage 0.9B component audit found.")
print("  Stage 0.9D.1 optimized plan found.")
print("  Stage 0.9D.2 verification found.")


# ============================================================
# VERIFY STAGE 0.9D.2
# ============================================================

print()
print("Checking Stage 0.9D.2 verification...")

with open(
    VERIFICATION_JSON,
    "r",
    encoding="utf-8"
) as f:

    verification = json.load(f)


if verification.get("overall_pass") is not True:

    print()
    print("ERROR:")
    print("Stage 0.9D.2 did not PASS.")
    print("Canonical construction stopped.")

    raise SystemExit(1)


print("  Stage 0.9D.2 status: PASS")


# ============================================================
# LOAD OPTIMIZED PLAN
# ============================================================

print()
print("Loading optimized component assignments...")

with open(
    OPTIMIZED_PLAN_JSON,
    "r",
    encoding="utf-8"
) as f:

    optimized_plan = json.load(f)


train_components = [
    int(x)
    for x in optimized_plan.get(
        "train_components",
        []
    )
]

val_components = [
    int(x)
    for x in optimized_plan.get(
        "val_components",
        []
    )
]

test_components = [
    int(x)
    for x in optimized_plan.get(
        "test_components",
        []
    )
]


component_to_split = {}

for component_id in train_components:

    component_to_split[
        component_id
    ] = "train"


for component_id in val_components:

    component_to_split[
        component_id
    ] = "val"


for component_id in test_components:

    component_to_split[
        component_id
    ] = "test"


print()
print("TRAIN components:")
print(" ", train_components)

print("VAL components:")
print(" ", val_components)

print("TEST components:")
print(" ", test_components)


# ============================================================
# NORMALIZE SEQUENCE
# ============================================================

def normalize_sequence_name(value):

    if isinstance(value, dict):

        if "sequence" in value:

            return normalize_sequence_name(
                value["sequence"]
            )

        split = str(
            value.get(
                "split",
                ""
            )
        ).strip()

        data_folder = str(
            value.get(
                "data_folder",
                ""
            )
        ).strip()

        if split and data_folder:

            return (
                f"{split}/{data_folder}"
            )

        return ""

    if value is None:

        return ""

    value = str(
        value
    ).strip()

    value = value.replace(
        "\\",
        "/"
    )

    if value.startswith("./"):

        value = value[2:]

    return value


# ============================================================
# LOAD COMPONENT MAPPING
# ============================================================

print()
print("Loading sequence/component mapping...")

with open(
    COMPONENT_JSON,
    "r",
    encoding="utf-8"
) as f:

    component_data = json.load(f)


components = {}

raw_components = component_data.get(
    "components",
    component_data.get(
        "connected_components",
        []
    )
)


if isinstance(
    raw_components,
    dict
):

    for component_id, value in (
        raw_components.items()
    ):

        try:

            cid = int(
                component_id
            )

        except Exception:

            continue

        if isinstance(
            value,
            dict
        ):

            raw_members = value.get(
                "sequences",
                value.get(
                    "members",
                    []
                )
            )

        else:

            raw_members = value

        members = []

        if isinstance(
            raw_members,
            list
        ):

            for member in raw_members:

                sequence = (
                    normalize_sequence_name(
                        member
                    )
                )

                if sequence:

                    members.append(
                        sequence
                    )

        components[
            cid
        ] = members


elif isinstance(
    raw_components,
    list
):

    for index, value in enumerate(
        raw_components,
        start=1
    ):

        cid = index

        if isinstance(
            value,
            dict
        ):

            if "component_id" in value:

                try:

                    cid = int(
                        value[
                            "component_id"
                        ]
                    )

                except Exception:

                    pass

            raw_members = value.get(
                "sequences",
                value.get(
                    "members",
                    []
                )
            )

        else:

            raw_members = value

        members = []

        if isinstance(
            raw_members,
            list
        ):

            for member in raw_members:

                sequence = (
                    normalize_sequence_name(
                        member
                    )
                )

                if sequence:

                    members.append(
                        sequence
                    )

        components[
            cid
        ] = members


sequence_to_component = {}


for component_id, members in (
    components.items()
):

    for sequence in members:

        if sequence in (
            sequence_to_component
        ):

            print()
            print(
                "ERROR: Sequence appears in "
                "multiple components:"
            )
            print(
                "  " + sequence
            )

            raise SystemExit(1)

        sequence_to_component[
            sequence
        ] = component_id


print(
    f"  Components: "
    f"{len(components)}"
)

print(
    f"  Sequences: "
    f"{len(sequence_to_component)}"
)


# ============================================================
# LOAD INVENTORY
# ============================================================

print()
print("Loading Stage 0.9 inventory...")

inventory = []

with open(
    INVENTORY_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(
        f
    )

    fieldnames = (
        reader.fieldnames or []
    )

    for row in reader:

        inventory.append(
            row
        )


print(
    f"  Inventory records: "
    f"{len(inventory):,}"
)


# ============================================================
# INVENTORY HELPERS
# ============================================================

def get_sequence_from_row(row):

    for key in [
        "sequence_id",
        "sequence",
        "sequence_name"
    ]:

        if (
            key in row
            and row[key]
        ):

            value = (
                normalize_sequence_name(
                    row[key]
                )
            )

            if "/" in value:

                return value

    split = ""

    for key in [
        "split",
        "original_split"
    ]:

        if (
            key in row
            and row[key]
        ):

            split = str(
                row[key]
            ).strip()

            break

    data_folder = ""

    for key in [
        "data_folder",
        "data",
        "folder"
    ]:

        if (
            key in row
            and row[key]
        ):

            data_folder = str(
                row[key]
            ).strip()

            break

    if (
        split
        and data_folder
    ):

        return (
            f"{split}/{data_folder}"
        )

    image_path = ""

    for key in [
        "image_path",
        "image",
        "path"
    ]:

        if (
            key in row
            and row[key]
        ):

            image_path = str(
                row[key]
            ).replace(
                "\\",
                "/"
            )

            break

    if image_path:

        parts = image_path.split(
            "/"
        )

        for i, part in enumerate(
            parts
        ):

            if part.lower() in {
                "train",
                "val",
                "test"
            }:

                if (
                    i + 1
                    < len(parts)
                ):

                    return (
                        f"{part.lower()}/"
                        f"{parts[i + 1]}"
                    )

    return ""


def get_image_filename_from_row(row):

    for key in [
        "image_path",
        "image",
        "path"
    ]:

        if (
            key in row
            and row[key]
        ):

            value = str(
                row[key]
            ).replace(
                "\\",
                "/"
            )

            return os.path.basename(
                value
            )

    return ""


# ============================================================
# BUILD RECORDS
# ============================================================

print()
print("Building canonical records...")

canonical_records = []

mapping_errors = []


for row_index, row in enumerate(
    inventory
):

    sequence = (
        get_sequence_from_row(
            row
        )
    )

    if not sequence:

        mapping_errors.append(
            f"Row {row_index + 2}: "
            "sequence could not be determined"
        )

        continue

    if (
        sequence
        not in sequence_to_component
    ):

        mapping_errors.append(
            f"Row {row_index + 2}: "
            f"unknown sequence {sequence}"
        )

        continue

    component_id = (
        sequence_to_component[
            sequence
        ]
    )

    if (
        component_id
        not in component_to_split
    ):

        mapping_errors.append(
            f"Row {row_index + 2}: "
            f"component {component_id} "
            "has no split"
        )

        continue

    split = (
        component_to_split[
            component_id
        ]
    )

    original_filename = (
        get_image_filename_from_row(
            row
        )
    )

    if not original_filename:

        mapping_errors.append(
            f"Row {row_index + 2}: "
            "image filename could not "
            "be determined"
        )

        continue

    canonical_records.append({

        "sequence":
            sequence,

        "component":
            component_id,

        "split":
            split,

        "original_filename":
            original_filename,

        "inventory_row":
            row
    })


if mapping_errors:

    print()
    print(
        "ERROR: Inventory mapping failed."
    )

    for error in mapping_errors[:20]:

        print(
            "  " + error
        )

    raise SystemExit(1)


print(
    f"  Canonical records: "
    f"{len(canonical_records):,}"
)


if len(canonical_records) != (
    len(inventory)
):

    print()
    print(
        "ERROR: Record count mismatch."
    )

    raise SystemExit(1)


# ============================================================
# RESOLVE SOURCE FILES
# ============================================================

print()
print(
    "Resolving Package 1 image/JSON paths..."
)


def resolve_package1_files(
    record
):

    sequence = record[
        "sequence"
    ]

    split, data_folder = (
        sequence.split(
            "/",
            1
        )
    )

    original_filename = (
        record[
            "original_filename"
        ]
    )

    stem = os.path.splitext(
        original_filename
    )[0]

    image_dir = os.path.join(
        PACKAGE_1,
        "images",
        split,
        data_folder
    )

    json_dir = os.path.join(
        PACKAGE_1,
        "json",
        split,
        data_folder
    )

    source_image = os.path.join(
        image_dir,
        original_filename
    )

    source_json = os.path.join(
        json_dir,
        stem + ".json"
    )

    if (
        os.path.isfile(
            source_image
        )
        and
        os.path.isfile(
            source_json
        )
    ):

        return (
            source_image,
            source_json
        )

    # Case-insensitive fallback
    image_match = None

    if os.path.isdir(
        image_dir
    ):

        target = (
            original_filename.lower()
        )

        for filename in os.listdir(
            image_dir
        ):

            if filename.lower() == target:

                image_match = os.path.join(
                    image_dir,
                    filename
                )

                break

    json_match = None

    if os.path.isdir(
        json_dir
    ):

        target = (
            stem + ".json"
        ).lower()

        for filename in os.listdir(
            json_dir
        ):

            if filename.lower() == target:

                json_match = os.path.join(
                    json_dir,
                    filename
                )

                break

    if (
        image_match
        and json_match
    ):

        return (
            image_match,
            json_match
        )

    return (
        None,
        None
    )


# ============================================================
# PRE-COPY VALIDATION
# ============================================================

print()
print(
    "Performing pre-copy validation..."
)

print(
    "No output files will be created "
    "until validation passes."
)


resolved_records = []

missing_images = []

missing_json = []


for record in canonical_records:

    source_image, source_json = (
        resolve_package1_files(
            record
        )
    )

    if source_image is None:

        missing_images.append(
            (
                record[
                    "sequence"
                ],
                record[
                    "original_filename"
                ]
            )
        )

        continue

    if source_json is None:

        missing_json.append(
            (
                record[
                    "sequence"
                ],
                record[
                    "original_filename"
                ]
            )
        )

        continue

    resolved_records.append({

        **record,

        "source_image":
            source_image,

        "source_json":
            source_json
    })


print(
    f"  Resolved pairs: "
    f"{len(resolved_records):,}"
)

print(
    f"  Missing images: "
    f"{len(missing_images):,}"
)

print(
    f"  Missing JSON: "
    f"{len(missing_json):,}"
)


if (
    missing_images
    or missing_json
):

    print()
    print(
        "ERROR: Pre-copy validation FAILED."
    )

    print(
        "Nothing will be constructed."
    )

    if missing_images:

        print()
        print(
            "Example missing images:"
        )

        for item in missing_images[:10]:

            print(
                "  " + str(item)
            )

    if missing_json:

        print()
        print(
            "Example missing JSON:"
        )

        for item in missing_json[:10]:

            print(
                "  " + str(item)
            )

    raise SystemExit(1)


if len(resolved_records) != 14087:

    print()
    print(
        "ERROR: Expected 14,087 "
        "resolved records."
    )

    raise SystemExit(1)


print()
print(
    "  PRE-COPY VALIDATION PASS"
)


# ============================================================
# REMOVE INCOMPLETE PREVIOUS OUTPUT
# ============================================================

if os.path.exists(
    CANONICAL_DIR
):

    print()
    print(
        "Existing canonical_dataset detected."
    )

    print(
        "This is expected because the previous "
        "Stage 0.9E run ran out of disk space."
    )

    print()
    print(
        "Only this incomplete canonical_dataset "
        "will be removed."
    )

    print(
        "The raw BeltCrack9kd dataset will NOT "
        "be modified."
    )

    # Absolute path safety
    expected_path = os.path.normcase(
        os.path.abspath(
            os.path.join(
                PROJECT_ROOT,
                "canonical_dataset"
            )
        )
    )

    actual_path = os.path.normcase(
        os.path.abspath(
            CANONICAL_DIR
        )
    )

    if actual_path != expected_path:

        print()
        print(
            "ERROR: Safety path check failed."
        )

        raise SystemExit(1)

    if not actual_path.startswith(
        os.path.normcase(
            os.path.abspath(
                PROJECT_ROOT
            )
        )
    ):

        print()
        print(
            "ERROR: Canonical directory is "
            "outside project."
        )

        raise SystemExit(1)

    print()
    print(
        "Removing incomplete canonical_dataset..."
    )

    shutil.rmtree(
        CANONICAL_DIR
    )

    print(
        "  Incomplete output removed."
    )


# ============================================================
# CREATE CANONICAL STRUCTURE
# ============================================================

print()
print(
    "Creating canonical dataset structure..."
)


for split in [
    "train",
    "val",
    "test"
]:

    os.makedirs(
        os.path.join(
            CANONICAL_DIR,
            split,
            "labels"
        ),
        exist_ok=True
    )


# ============================================================
# COPY ONLY JSON ANNOTATIONS
# ============================================================

print()
print(
    "Copying annotation JSON files..."
)

print()
print(
    "FULL-RESOLUTION IMAGES ARE NOT DUPLICATED."
)

print(
    "Original image paths are stored in "
    "canonical_manifest.csv."
)

print()


split_counts = defaultdict(
    int
)

sequence_counts = defaultdict(
    int
)

component_counts = defaultdict(
    int
)

manifest_records = []

destination_json_names = set()


for index, record in enumerate(
    resolved_records,
    start=1
):

    split = record[
        "split"
    ]

    sequence = record[
        "sequence"
    ]

    component_id = record[
        "component"
    ]

    original_filename = record[
        "original_filename"
    ]

    source_image = record[
        "source_image"
    ]

    source_json = record[
        "source_json"
    ]

    # --------------------------------------------------------
    # Sequence-safe prefix
    # --------------------------------------------------------

    sequence_prefix = (
        sequence
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            " ",
            "_"
        )
    )

    # --------------------------------------------------------
    # Unique canonical filenames
    # --------------------------------------------------------

    canonical_image_filename = (
        sequence_prefix
        + "_"
        + original_filename
    )

    original_json_filename = (
        os.path.basename(
            source_json
        )
    )

    canonical_json_filename = (
        sequence_prefix
        + "_"
        + original_json_filename
    )

    destination_json = os.path.join(
        CANONICAL_DIR,
        split,
        "labels",
        canonical_json_filename
    )

    # --------------------------------------------------------
    # Collision protection
    # --------------------------------------------------------

    if canonical_json_filename in (
        destination_json_names
    ):

        print()
        print(
            "ERROR: JSON filename collision:"
        )

        print(
            "  "
            + canonical_json_filename
        )

        raise SystemExit(1)

    destination_json_names.add(
        canonical_json_filename
    )

    # --------------------------------------------------------
    # Copy ONLY JSON
    # --------------------------------------------------------

    shutil.copy2(
        source_json,
        destination_json
    )

    # --------------------------------------------------------
    # Record manifest
    # --------------------------------------------------------

    manifest_records.append({

        "split":
            split,

        "component":
            component_id,

        "sequence":
            sequence,

        "original_image_filename":
            original_filename,

        "canonical_image_filename":
            canonical_image_filename,

        "canonical_image_path":
            os.path.relpath(
                source_image,
                PROJECT_ROOT
            ).replace(
                "\\",
                "/"
            ),

        "original_json_filename":
            original_json_filename,

        "canonical_json_filename":
            canonical_json_filename,

        "source_image":
            os.path.relpath(
                source_image,
                RAW_DATASET
            ).replace(
                "\\",
                "/"
            ),

        "source_json":
            os.path.relpath(
                source_json,
                RAW_DATASET
            ).replace(
                "\\",
                "/"
            )
    })

    split_counts[
        split
    ] += 1

    sequence_counts[
        sequence
    ] += 1

    component_counts[
        f"{split}:{component_id}"
    ] += 1

    if (
        index == 1
        or index % 500 == 0
        or index == len(
            resolved_records
        )
    ):

        print(
            f"  Processed "
            f"{index:,} / "
            f"{len(resolved_records):,}"
        )


# ============================================================
# WRITE MANIFEST
# ============================================================

print()
print(
    "Writing canonical manifest..."
)


manifest_fields = [

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


with open(
    MANIFEST_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_fields
    )

    writer.writeheader()

    writer.writerows(
        manifest_records
    )


# ============================================================
# WRITE DATASET INFO
# ============================================================

print()
print(
    "Writing canonical dataset metadata..."
)


dataset_info = {

    "stage":
        "0.9E",

    "dataset_name":
        "BeltCrack9kd_Damage_Intelligence",

    "source_dataset":
        "BeltCrack9kd",

    "annotation_authority":
        "BeltCrack9kd1",

    "construction_type":
        "Manifest-based canonical dataset",

    "image_storage":
        "Original raw dataset",

    "image_duplication":
        False,

    "annotation_storage":
        "Copied JSON files",

    "original_dataset_modified":
        False,

    "raw_files_deleted":
        False,

    "raw_files_moved":
        False,

    "total_records":
        len(manifest_records),

    "split_counts":
        dict(
            split_counts
        ),

    "component_assignments": {

        "train":
            train_components,

        "val":
            val_components,

        "test":
            test_components
    },

    "sequence_count":
        len(
            sequence_counts
        ),

    "sequence_counts":
        dict(
            sorted(
                sequence_counts.items()
            )
        ),

    "component_counts":
        dict(
            sorted(
                component_counts.items()
            )
        ),

    "annotation_format":
        "LabelMe JSON",

    "label":
        "crack",

    "filename_policy":
        "split_datafolder_original_filename",

    "canonical_structure": {

        "train":
            "train/labels",

        "val":
            "val/labels",

        "test":
            "test/labels"
    },

    "image_access":
        "Use canonical_manifest.csv"
}


with open(
    DATASET_INFO_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset_info,
        f,
        indent=2
    )


# ============================================================
# POST-CONSTRUCTION VERIFICATION
# ============================================================

print()
print(
    "Performing post-construction verification..."
)


post_counts = {}


for split in [
    "train",
    "val",
    "test"
]:

    label_dir = os.path.join(
        CANONICAL_DIR,
        split,
        "labels"
    )

    json_files = [

        x

        for x in os.listdir(
            label_dir
        )

        if x.lower().endswith(
            ".json"
        )
    ]

    post_counts[
        split
    ] = len(
        json_files
    )

    print()
    print(
        f"  {split.upper()}:"
    )

    print(
        f"    Annotation JSON: "
        f"{len(json_files):,}"
    )


post_total_json = sum(
    post_counts.values()
)


post_copy_pass = (

    post_total_json == 14087

    and

    post_counts[
        "train"
    ] == 9819

    and

    post_counts[
        "val"
    ] == 2102

    and

    post_counts[
        "test"
    ] == 2166

    and

    len(
        manifest_records
    ) == 14087
)


# ============================================================
# WRITE CONSTRUCTION REPORT
# ============================================================

construction_report = {

    "stage":
        "0.9E",

    "status":
        "PASS"
        if post_copy_pass
        else "FAIL",

    "construction_mode":
        "manifest_based",

    "source_dataset":
        RAW_DATASET,

    "annotation_package":
        PACKAGE_1,

    "canonical_dataset":
        CANONICAL_DIR,

    "input_records":
        len(inventory),

    "resolved_records":
        len(resolved_records),

    "annotation_json_copied":
        post_total_json,

    "raw_images_copied":
        0,

    "raw_dataset_modified":
        False,

    "raw_files_deleted":
        False,

    "raw_files_moved":
        False,

    "split_counts":
        dict(
            split_counts
        ),

    "post_construction_counts":
        post_counts,

    "post_copy_pass":
        post_copy_pass,

    "mapping_errors":
        mapping_errors,

    "missing_images":
        missing_images,

    "missing_json":
        missing_json
}


with open(
    CONSTRUCTION_REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        construction_report,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)


if post_copy_pass:

    print(
        "STAGE 0.9E COMPLETE"
    )

    print("=" * 72)

    print()
    print(
        "Canonical manifest successfully constructed."
    )

    print()
    print(
        "TRAIN: "
        f"{post_counts['train']:,} annotations"
    )

    print(
        "VAL:   "
        f"{post_counts['val']:,} annotations"
    )

    print(
        "TEST:  "
        f"{post_counts['test']:,} annotations"
    )

    print()
    print(
        "TOTAL: "
        f"{post_total_json:,} annotations"
    )

    print()
    print(
        "Raw images duplicated:"
    )

    print(
        "  0"
    )

    print()
    print(
        "Canonical dataset:"
    )

    print(
        "  " + CANONICAL_DIR
    )

    print()
    print(
        "Manifest:"
    )

    print(
        "  " + MANIFEST_CSV
    )

    print()
    print(
        "Dataset info:"
    )

    print(
        "  " + DATASET_INFO_JSON
    )

    print()
    print(
        "Construction report:"
    )

    print(
        "  " + CONSTRUCTION_REPORT_JSON
    )

    print()
    print(
        "NEXT: STAGE 0.9F - "
        "FINAL CANONICAL DATASET VERIFICATION"
    )

else:

    print(
        "STAGE 0.9E FAILED"
    )

    print("=" * 72)

    print()
    print(
        "Canonical dataset verification failed."
    )

    print(
        "Do NOT continue to Stage 0.9F."
    )