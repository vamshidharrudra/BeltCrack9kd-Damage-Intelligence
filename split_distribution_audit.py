from pathlib import Path
import json
import csv
from collections import Counter, defaultdict
import statistics


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

OUTPUT_DIR = PROJECT_ROOT / "audit_outputs"

INVENTORY_CSV = (
    OUTPUT_DIR /
    "stage_0_9_canonical_image_inventory.csv"
)

SPLIT_PLAN_JSON = (
    OUTPUT_DIR /
    "stage_0_9c_split_plan.json"
)

COMPONENT_JSON = (
    OUTPUT_DIR /
    "stage_0_9b_corrected_components.json"
)

OUTPUT_JSON = (
    OUTPUT_DIR /
    "stage_0_9d_distribution_validation.json"
)


# ============================================================
# EXPECTED SPLIT
# ============================================================

EXPECTED_TOTAL = 14087

EXPECTED_COUNTS = {
    "train": 9861,
    "val": 2113,
    "test": 2113
}


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        len(values) - 1
    ) * p

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        +
        (
            values[upper]
            -
            values[lower]
        )
        *
        fraction
    )


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    name,
    values
):

    if not values:

        print(
            f"  {name}: no data"
        )

        return

    print(
        f"  {name}:"
    )

    print(
        f"    mean   : "
        f"{statistics.mean(values):.4f}"
    )

    print(
        f"    median : "
        f"{statistics.median(values):.4f}"
    )

    print(
        f"    min    : "
        f"{min(values):.4f}"
    )

    print(
        f"    max    : "
        f"{max(values):.4f}"
    )

    print(
        f"    p25    : "
        f"{percentile(values, 0.25):.4f}"
    )

    print(
        f"    p75    : "
        f"{percentile(values, 0.75):.4f}"
    )


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory():

    print()
    print(
        "Loading canonical image inventory..."
    )

    if not INVENTORY_CSV.exists():

        raise FileNotFoundError(
            f"Inventory file not found:\n"
            f"{INVENTORY_CSV}"
        )

    records = []

    with open(
        INVENTORY_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            # Shape count
            row["shape_count"] = int(
                row["shape_count"]
            )

            # Width
            if row.get("width"):

                row["width"] = int(
                    row["width"]
                )

            else:

                row["width"] = None

            # Height
            if row.get("height"):

                row["height"] = int(
                    row["height"]
                )

            else:

                row["height"] = None

            records.append(row)

    print(
        f"Loaded {len(records):,} "
        f"inventory records."
    )

    return records


# ============================================================
# LOAD STAGE 0.9B COMPONENT MEMBERSHIP
# ============================================================

def load_component_membership():

    print()
    print(
        "Loading Stage 0.9B component membership..."
    )

    if not COMPONENT_JSON.exists():

        raise FileNotFoundError(
            f"Component file not found:\n"
            f"{COMPONENT_JSON}"
        )

    with open(
        COMPONENT_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "components" not in data:

        raise KeyError(
            "Stage 0.9B JSON does not contain "
            "'components'."
        )

    membership = {}

    for component in data["components"]:

        component_id = int(
            component["component_id"]
        )

        sequences = component.get(
            "sequences",
            []
        )

        for sequence_info in sequences:

            sequence_name = (
                sequence_info.get(
                    "sequence"
                )
            )

            if not sequence_name:

                continue

            membership[
                sequence_name
            ] = component_id

    print(
        f"Mapped {len(membership)} "
        f"sequences to components."
    )

    return membership


# ============================================================
# LOAD ACTUAL STAGE 0.9C SPLIT PLAN
# ============================================================

def load_split_plan():

    print()
    print(
        "Loading Stage 0.9C split plan..."
    )

    if not SPLIT_PLAN_JSON.exists():

        raise FileNotFoundError(
            f"Split plan not found:\n"
            f"{SPLIT_PLAN_JSON}"
        )

    with open(
        SPLIT_PLAN_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Verify expected fields
    # --------------------------------------------------------

    required_fields = [
        "train_components",
        "val_components",
        "test_components"
    ]

    missing_fields = []

    for field in required_fields:

        if field not in data:

            missing_fields.append(
                field
            )

    if missing_fields:

        raise KeyError(
            "Stage 0.9C split plan is missing: "
            +
            ", ".join(missing_fields)
        )

    # --------------------------------------------------------
    # Read component lists
    # --------------------------------------------------------

    train_components = [
        int(x)
        for x in data[
            "train_components"
        ]
    ]

    val_components = [
        int(x)
        for x in data[
            "val_components"
        ]
    ]

    test_components = [
        int(x)
        for x in data[
            "test_components"
        ]
    ]

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    train_set = set(
        train_components
    )

    val_set = set(
        val_components
    )

    test_set = set(
        test_components
    )

    train_val_overlap = (
        train_set
        &
        val_set
    )

    train_test_overlap = (
        train_set
        &
        test_set
    )

    val_test_overlap = (
        val_set
        &
        test_set
    )

    if train_val_overlap:

        raise RuntimeError(
            "Components appear in both "
            "TRAIN and VAL: "
            +
            str(
                sorted(
                    train_val_overlap
                )
            )
        )

    if train_test_overlap:

        raise RuntimeError(
            "Components appear in both "
            "TRAIN and TEST: "
            +
            str(
                sorted(
                    train_test_overlap
                )
            )
        )

    if val_test_overlap:

        raise RuntimeError(
            "Components appear in both "
            "VAL and TEST: "
            +
            str(
                sorted(
                    val_test_overlap
                )
            )
        )

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    assignments = {}

    for component_id in train_components:

        assignments[
            component_id
        ] = "train"

    for component_id in val_components:

        assignments[
            component_id
        ] = "val"

    for component_id in test_components:

        assignments[
            component_id
        ] = "test"

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "Stage 0.9C component assignments:"
    )

    print(
        f"  TRAIN: "
        f"{len(train_components)} components"
    )

    print(
        f"  VAL  : "
        f"{len(val_components)} components"
    )

    print(
        f"  TEST : "
        f"{len(test_components)} components"
    )

    print()

    print(
        "TRAIN components:"
    )

    print(
        "  "
        +
        ", ".join(
            str(x)
            for x in sorted(
                train_components
            )
        )
    )

    print()

    print(
        "VAL components:"
    )

    print(
        "  "
        +
        ", ".join(
            str(x)
            for x in sorted(
                val_components
            )
        )
    )

    print()

    print(
        "TEST components:"
    )

    print(
        "  "
        +
        ", ".join(
            str(x)
            for x in sorted(
                test_components
            )
        )
    )

    # --------------------------------------------------------
    # Verify actual count information if present
    # --------------------------------------------------------

    if "actual_images" in data:

        print()
        print(
            "Stage 0.9C recorded image counts:"
        )

        actual_images = data[
            "actual_images"
        ]

        for split in [
            "train",
            "val",
            "test"
        ]:

            if split in actual_images:

                print(
                    f"  {split.upper():5s}: "
                    f"{actual_images[split]:,}"
                )

    return assignments


# ============================================================
# BUILD NEW SPLIT RECORDS
# ============================================================

def build_split_records(
    records,
    membership,
    assignments
):

    print()
    print(
        "Assigning inventory records "
        "to proposed ML splits..."
    )

    split_records = defaultdict(list)

    missing_sequences = []

    missing_components = []

    for record in records:

        sequence = (
            f"{record['split']}/"
            f"{record['data_folder']}"
        )

        # ----------------------------------------------------
        # Find component
        # ----------------------------------------------------

        if sequence not in membership:

            missing_sequences.append(
                sequence
            )

            continue

        component_id = membership[
            sequence
        ]

        # ----------------------------------------------------
        # Find split assignment
        # ----------------------------------------------------

        if component_id not in assignments:

            missing_components.append(
                component_id
            )

            continue

        assigned_split = assignments[
            component_id
        ]

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        record_copy = dict(
            record
        )

        record_copy[
            "component_id"
        ] = component_id

        record_copy[
            "assigned_split"
        ] = assigned_split

        split_records[
            assigned_split
        ].append(
            record_copy
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if missing_sequences:

        print()
        print(
            "ERROR: Missing sequences:"
        )

        for sequence in sorted(
            set(missing_sequences)
        ):

            print(
                f"  {sequence}"
            )

        raise RuntimeError(
            "Some inventory sequences were "
            "not found in Stage 0.9B."
        )

    if missing_components:

        print()
        print(
            "ERROR: Missing component assignments:"
        )

        for component_id in sorted(
            set(missing_components)
        ):

            print(
                f"  Component {component_id}"
            )

        raise RuntimeError(
            "Some components were not assigned "
            "by Stage 0.9C."
        )

    total_assigned = sum(
        len(items)
        for items
        in split_records.values()
    )

    if total_assigned != len(records):

        raise RuntimeError(
            f"Assignment mismatch: "
            f"{total_assigned:,} assigned vs "
            f"{len(records):,} records."
        )

    print()
    print(
        f"Successfully assigned "
        f"{total_assigned:,} records."
    )

    return split_records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STAGE 0.9D - SPLIT DISTRIBUTION VALIDATION"
    )
    print("=" * 80)

    # ========================================================
    # FILE CHECK
    # ========================================================

    print()
    print(
        "Checking required files..."
    )

    required_files = [
        INVENTORY_CSV,
        SPLIT_PLAN_JSON,
        COMPONENT_JSON
    ]

    for path in required_files:

        if not path.exists():

            print()
            print(
                "ERROR: Required file not found:"
            )

            print(
                path
            )

            return

        print(
            f"  OK: {path.name}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    records = load_inventory()

    assignments = load_split_plan()

    membership = load_component_membership()

    split_records = build_split_records(
        records,
        membership,
        assignments
    )

    total_images = len(
        records
    )

    # ========================================================
    # 1. BASIC SPLIT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "1. BASIC SPLIT DISTRIBUTION"
    )
    print("=" * 80)

    split_counts = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        count = len(
            split_records[split]
        )

        split_counts[
            split
        ] = count

        percentage = (
            count
            /
            total_images
            *
            100
        )

        print(
            f"{split.upper():5s}: "
            f"{count:,} images "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # 2. CRACK/POLYGON COUNT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "2. CRACK/POLYGON COUNT DISTRIBUTION"
    )
    print("=" * 80)

    shape_distributions = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        counter = Counter(
            record["shape_count"]
            for record
            in split_records[split]
        )

        shape_distributions[
            split
        ] = counter

        print()
        print(
            f"{split.upper()}:"
        )

        split_total = len(
            split_records[split]
        )

        for (
            shape_count,
            image_count
        ) in sorted(
            counter.items()
        ):

            percentage = (
                image_count
                /
                split_total
                *
                100
            )

            print(
                f"  {shape_count:2d} polygon(s): "
                f"{image_count:5,d} "
                f"({percentage:6.2f}%)"
            )

    # ========================================================
    # 3. CRACKS PER IMAGE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "3. CRACKS PER IMAGE"
    )
    print("=" * 80)

    cracks_per_image = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        values = [
            record["shape_count"]
            for record
            in split_records[split]
        ]

        cracks_per_image[
            split
        ] = values

        print()
        print(
            split.upper()
        )

        print_statistics(
            "cracks per image",
            values
        )

    # ========================================================
    # 4. IMAGE RESOLUTION DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "4. IMAGE RESOLUTION DISTRIBUTION"
    )
    print("=" * 80)

    resolution_distributions = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        counter = Counter(
            (
                record["width"],
                record["height"]
            )
            for record
            in split_records[split]
        )

        resolution_distributions[
            split
        ] = counter

        print()
        print(
            f"{split.upper()}:"
        )

        split_total = len(
            split_records[split]
        )

        for (
            resolution,
            count
        ) in counter.most_common():

            width, height = resolution

            percentage = (
                count
                /
                split_total
                *
                100
            )

            print(
                f"  {width} x {height}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    # ========================================================
    # 5. SEQUENCE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "5. SEQUENCE DISTRIBUTION"
    )
    print("=" * 80)

    sequence_summary = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        sequence_counts = Counter()

        for record in split_records[split]:

            sequence = (
                f"{record['split']}/"
                f"{record['data_folder']}"
            )

            sequence_counts[
                sequence
            ] += 1

        sequence_summary[
            split
        ] = sequence_counts

        print()
        print(
            f"{split.upper()}: "
            f"{len(sequence_counts)} sequences"
        )

        for (
            sequence,
            count
        ) in sorted(
            sequence_counts.items()
        ):

            print(
                f"  {sequence:18s}: "
                f"{count:,} images"
            )

    # ========================================================
    # 6. LABEL DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "6. LABEL DISTRIBUTION"
    )
    print("=" * 80)

    label_distributions = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        counter = Counter()

        for record in split_records[split]:

            labels = record.get(
                "labels",
                ""
            )

            if labels:

                for label in labels.split("|"):

                    label = label.strip()

                    if label:

                        counter[
                            label
                        ] += 1

        label_distributions[
            split
        ] = counter

        print()
        print(
            f"{split.upper()}:"
        )

        for (
            label,
            count
        ) in counter.most_common():

            print(
                f"  {label}: {count:,}"
            )

    # ========================================================
    # 7. COMPONENT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "7. COMPONENT DISTRIBUTION"
    )
    print("=" * 80)

    component_summary = {}

    for split in [
        "train",
        "val",
        "test"
    ]:

        component_counts = Counter(
            record["component_id"]
            for record
            in split_records[split]
        )

        component_summary[
            split
        ] = component_counts

        print()
        print(
            f"{split.upper()}: "
            f"{len(component_counts)} components"
        )

        for (
            component_id,
            count
        ) in sorted(
            component_counts.items()
        ):

            print(
                f"  Component "
                f"{component_id:2d}: "
                f"{count:,} images"
            )

    # ========================================================
    # 8. DISTRIBUTION DIFFERENCE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "8. DISTRIBUTION DIFFERENCE CHECK"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Shape-count difference
    # --------------------------------------------------------

    all_shape_counts = sorted(
        set(
            shape_count
            for counter
            in shape_distributions.values()
            for shape_count
            in counter.keys()
        )
    )

    max_shape_difference = 0.0

    for shape_count in all_shape_counts:

        proportions = []

        for split in [
            "train",
            "val",
            "test"
        ]:

            count = (
                shape_distributions[
                    split
                ].get(
                    shape_count,
                    0
                )
            )

            split_total = len(
                split_records[split]
            )

            proportions.append(
                count
                /
                split_total
            )

        difference = (
            max(proportions)
            -
            min(proportions)
        )

        max_shape_difference = max(
            max_shape_difference,
            difference
        )

    print()
    print(
        "Maximum absolute difference in "
        "shape-count proportions:"
    )

    print(
        f"  {max_shape_difference * 100:.2f} "
        f"percentage points"
    )

    # --------------------------------------------------------
    # Resolution difference
    # --------------------------------------------------------

    all_resolutions = sorted(
        set(
            resolution
            for counter
            in resolution_distributions.values()
            for resolution
            in counter.keys()
        )
    )

    max_resolution_difference = 0.0

    for resolution in all_resolutions:

        proportions = []

        for split in [
            "train",
            "val",
            "test"
        ]:

            count = (
                resolution_distributions[
                    split
                ].get(
                    resolution,
                    0
                )
            )

            split_total = len(
                split_records[split]
            )

            proportions.append(
                count
                /
                split_total
            )

        difference = (
            max(proportions)
            -
            min(proportions)
        )

        max_resolution_difference = max(
            max_resolution_difference,
            difference
        )

    print()
    print(
        "Maximum absolute difference in "
        "resolution proportions:"
    )

    print(
        f"  {max_resolution_difference * 100:.2f} "
        f"percentage points"
    )

    # ========================================================
    # 9. SEQUENCE INTEGRITY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "9. SEQUENCE INTEGRITY VALIDATION"
    )
    print("=" * 80)

    sequence_to_split = {}

    sequence_conflicts = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        for sequence in (
            sequence_summary[split]
        ):

            if sequence in sequence_to_split:

                previous_split = (
                    sequence_to_split[
                        sequence
                    ]
                )

                if previous_split != split:

                    sequence_conflicts.append({
                        "sequence":
                            sequence,

                        "split_a":
                            previous_split,

                        "split_b":
                            split
                    })

            else:

                sequence_to_split[
                    sequence
                ] = split

    print(
        f"Sequences assigned to splits : "
        f"{len(sequence_to_split)}"
    )

    print(
        f"Sequence split conflicts      : "
        f"{len(sequence_conflicts)}"
    )

    if sequence_conflicts:

        print(
            "FAIL: Sequence appears in "
            "multiple ML splits."
        )

        for conflict in sequence_conflicts:

            print(
                f"  {conflict['sequence']} : "
                f"{conflict['split_a']} / "
                f"{conflict['split_b']}"
            )

    else:

        print(
            "PASS: No sequence appears in "
            "multiple ML splits."
        )

    # ========================================================
    # 10. EXACT DUPLICATE VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "10. EXACT DUPLICATE SPLIT VALIDATION"
    )
    print("=" * 80)

    hash_to_splits = defaultdict(set)

    for split in [
        "train",
        "val",
        "test"
    ]:

        for record in split_records[split]:

            digest = record.get(
                "sha256",
                ""
            )

            if digest:

                hash_to_splits[
                    digest
                ].add(
                    split
                )

    cross_split_hashes = {

        digest: splits

        for (
            digest,
            splits
        ) in hash_to_splits.items()

        if len(splits) > 1
    }

    print(
        f"Unique image hashes : "
        f"{len(hash_to_splits):,}"
    )

    print(
        f"Hashes crossing new splits : "
        f"{len(cross_split_hashes):,}"
    )

    if cross_split_hashes:

        print(
            "FAIL: Exact duplicate images "
            "still cross train/val/test."
        )

    else:

        print(
            "PASS: No exact image hash crosses "
            "train/val/test."
        )

    # ========================================================
    # 11. FINAL COUNT VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "11. FINAL VALIDATION"
    )
    print("=" * 80)

    train_count_check = (
        split_counts["train"]
        ==
        EXPECTED_COUNTS["train"]
    )

    val_count_check = (
        split_counts["val"]
        ==
        EXPECTED_COUNTS["val"]
    )

    test_count_check = (
        split_counts["test"]
        ==
        EXPECTED_COUNTS["test"]
    )

    total_check = (
        total_images
        ==
        EXPECTED_TOTAL
        and
        sum(
            split_counts.values()
        )
        ==
        EXPECTED_TOTAL
    )

    sequence_check = (
        len(sequence_conflicts)
        ==
        0
    )

    duplicate_check = (
        len(cross_split_hashes)
        ==
        0
    )

    print()

    print(
        f"Train count check   : "
        f"{'PASS' if train_count_check else 'FAIL'}"
    )

    print(
        f"Val count check     : "
        f"{'PASS' if val_count_check else 'FAIL'}"
    )

    print(
        f"Test count check    : "
        f"{'PASS' if test_count_check else 'FAIL'}"
    )

    print(
        f"Total image check   : "
        f"{'PASS' if total_check else 'FAIL'}"
    )

    print(
        f"Sequence integrity  : "
        f"{'PASS' if sequence_check else 'FAIL'}"
    )

    print(
        f"Duplicate isolation : "
        f"{'PASS' if duplicate_check else 'FAIL'}"
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "stage": "0.9D",

        "total_images": total_images,

        "split_counts": {
            "train":
                split_counts["train"],

            "val":
                split_counts["val"],

            "test":
                split_counts["test"]
        },

        "split_percentages": {},

        "cracks_per_image": {},

        "shape_count_distribution": {},

        "resolution_distribution": {},

        "sequence_counts": {},

        "component_counts": {},

        "label_distribution": {},

        "maximum_shape_distribution_difference":
            max_shape_difference,

        "maximum_resolution_distribution_difference":
            max_resolution_difference,

        "sequence_conflicts":
            len(sequence_conflicts),

        "cross_split_duplicate_hashes":
            len(cross_split_hashes),

        "validation": {
            "train_count":
                train_count_check,

            "val_count":
                val_count_check,

            "test_count":
                test_count_check,

            "total_images":
                total_check,

            "sequence_integrity":
                sequence_check,

            "duplicate_isolation":
                duplicate_check
        }
    }

    # --------------------------------------------------------
    # Split percentages
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "split_percentages"
        ][split] = (
            split_counts[split]
            /
            total_images
            *
            100
        )

    # --------------------------------------------------------
    # Crack statistics
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        values = cracks_per_image[
            split
        ]

        summary[
            "cracks_per_image"
        ][split] = {

            "mean":
                statistics.mean(values),

            "median":
                statistics.median(values),

            "min":
                min(values),

            "max":
                max(values),

            "p25":
                percentile(
                    values,
                    0.25
                ),

            "p75":
                percentile(
                    values,
                    0.75
                )
        }

    # --------------------------------------------------------
    # Shape distribution
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "shape_count_distribution"
        ][split] = {

            str(shape_count):
                count

            for (
                shape_count,
                count
            ) in shape_distributions[
                split
            ].items()
        }

    # --------------------------------------------------------
    # Resolution distribution
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "resolution_distribution"
        ][split] = {

            f"{width}x{height}":
                count

            for (
                width,
                height
            ), count in resolution_distributions[
                split
            ].items()
        }

    # --------------------------------------------------------
    # Sequence counts
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "sequence_counts"
        ][split] = len(
            sequence_summary[
                split
            ]
        )

    # --------------------------------------------------------
    # Component counts
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "component_counts"
        ][split] = len(
            component_summary[
                split
            ]
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        summary[
            "label_distribution"
        ][split] = {

            label:
                count

            for (
                label,
                count
            ) in label_distributions[
                split
            ].items()
        }

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print(
        "Saved Stage 0.9D summary:"
    )

    print(
        OUTPUT_JSON
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 0.9D COMPLETE"
    )
    print("=" * 80)

    all_structural_checks_pass = (
        train_count_check
        and
        val_count_check
        and
        test_count_check
        and
        total_check
        and
        sequence_check
        and
        duplicate_check
    )

    if all_structural_checks_pass:

        print()
        print(
            "STATUS: PASS"
        )

        print()
        print(
            "The proposed split passed "
            "all structural leakage checks."
        )

    else:

        print()
        print(
            "STATUS: REVIEW REQUIRED"
        )

        print()
        print(
            "Do not construct the canonical "
            "dataset yet."
        )

    print()
    print(
        "No images were copied."
    )

    print(
        "No images were deleted."
    )

    print(
        "No original dataset files were modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()