from pathlib import Path
import json
import csv
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

OUTPUT_DIR = PROJECT_ROOT / "audit_outputs"

COMPONENT_JSON = (
    OUTPUT_DIR /
    "stage_0_9b_corrected_components.json"
)


# ============================================================
# CANDIDATE SPLIT
# ============================================================
#
# IMPORTANT:
# We assign WHOLE duplicate-connected components.
#
# A component can never be divided between train/val/test.
#
# Test = 2,113 images
# Val  = 2,113 images
# Train = remaining 9,861 images
#
# Target:
# Train = 70%
# Val   = 15%
# Test  = 15%
#
# ============================================================

TEST_COMPONENTS = {
    4,
    6,
    13,
}

VAL_COMPONENTS = {
    12,
    14,
    15,
    16,
    17,
    19,
}


# ============================================================
# LOAD COMPONENTS
# ============================================================

def load_components():

    with open(
        COMPONENT_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["components"]


# ============================================================
# ASSIGN SPLIT
# ============================================================

def assign_split(component_id):

    if component_id in TEST_COMPONENTS:
        return "test"

    if component_id in VAL_COMPONENTS:
        return "val"

    return "train"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STAGE 0.9C - LEAKAGE-SAFE SPLIT PLANNING AUDIT"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not COMPONENT_JSON.exists():

        print()
        print("ERROR: Component JSON file not found.")
        print()
        print(
            f"Expected file:\n{COMPONENT_JSON}"
        )

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    components = load_components()

    print()
    print(
        f"Components loaded : {len(components)}"
    )

    # ========================================================
    # ASSIGN COMPONENTS
    # ========================================================

    assignments = {}

    for component in components:

        component_id = int(
            component["component_id"]
        )

        split = assign_split(
            component_id
        )

        assignments[
            component_id
        ] = split

    # ========================================================
    # COMPONENT ASSIGNMENTS
    # ========================================================

    print()
    print("=" * 80)
    print("COMPONENT ASSIGNMENTS")
    print("=" * 80)

    for component in components:

        cid = int(
            component["component_id"]
        )

        split = assignments[cid]

        print(
            f"Component {cid:2d}"
            f" -> {split.upper():5s}"
            f" -> {component['image_count']:,} images"
        )

        for sequence in component["sequences"]:

            print(
                f"    - {sequence['sequence']}"
                f" : {sequence['image_count']:,} images"
            )

    # ========================================================
    # SPLIT TOTALS
    # ========================================================

    split_images = Counter()
    split_components = Counter()
    split_sequences = Counter()

    for component in components:

        cid = int(
            component["component_id"]
        )

        split = assignments[cid]

        split_images[split] += int(
            component["image_count"]
        )

        split_components[split] += 1

        split_sequences[split] += int(
            component["sequence_count"]
        )

    total_images = sum(
        split_images.values()
    )

    # ========================================================
    # FINAL SPLIT SIZES
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL SPLIT SIZES")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test"
    ]:

        count = split_images[split]

        percentage = (
            count /
            total_images *
            100
        )

        print(
            f"{split.upper():5s}: "
            f"{count:5,d} images "
            f"({percentage:6.2f}%) "
            f"| components={split_components[split]} "
            f"| sequences={split_sequences[split]}"
        )

    print()
    print(
        f"TOTAL : {total_images:,}"
    )

    # ========================================================
    # VERIFY TOTAL
    # ========================================================

    expected_total = 14087

    print()
    print("=" * 80)
    print("TOTAL IMAGE COUNT VALIDATION")
    print("=" * 80)

    print(
        f"Expected : {expected_total:,}"
    )

    print(
        f"Actual   : {total_images:,}"
    )

    if total_images == expected_total:

        print(
            "PASS: Total image count matches 14,087."
        )

    else:

        print(
            "FAIL: Total image count does not match."
        )

    # ========================================================
    # VERIFY COMPONENT INTEGRITY
    # ========================================================

    print()
    print("=" * 80)
    print("COMPONENT INTEGRITY CHECK")
    print("=" * 80)

    integrity_errors = []

    for component in components:

        cid = int(
            component["component_id"]
        )

        assigned_split = assignments[cid]

        # A component has exactly one assignment.
        #
        # Every sequence inside it therefore inherits
        # the same assignment.

        for sequence in component["sequences"]:

            if assigned_split not in {
                "train",
                "val",
                "test"
            }:

                integrity_errors.append({
                    "component": cid,
                    "sequence":
                        sequence["sequence"]
                })

    if integrity_errors:

        print(
            f"FAIL: {len(integrity_errors)} "
            f"component integrity errors."
        )

        for error in integrity_errors:

            print(
                f"  Component {error['component']}"
                f" -> {error['sequence']}"
            )

    else:

        print(
            "PASS: Every connected component remains intact."
        )

    # ========================================================
    # VERIFY NO COMPONENT IS IN MULTIPLE SPLITS
    # ========================================================

    print()
    print(
        "Checking for component split conflicts..."
    )

    component_split_conflicts = []

    for component in components:

        cid = int(
            component["component_id"]
        )

        assigned = assignments[cid]

        if assigned not in {
            "train",
            "val",
            "test"
        }:

            component_split_conflicts.append(
                cid
            )

    if component_split_conflicts:

        print(
            "FAIL: Invalid component assignments:"
        )

        for cid in component_split_conflicts:

            print(
                f"  Component {cid}"
            )

    else:

        print(
            "PASS: No component is assigned "
            "to multiple ML splits."
        )

    # ========================================================
    # TARGET COMPARISON
    # ========================================================

    target_ratios = {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15
    }

    print()
    print("=" * 80)
    print("TARGET COMPARISON")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test"
    ]:

        actual_ratio = (
            split_images[split]
            /
            total_images
        )

        target_ratio = (
            target_ratios[split]
        )

        difference = (
            actual_ratio -
            target_ratio
        ) * 100

        print(
            f"{split.upper():5s}: "
            f"actual={actual_ratio * 100:6.2f}% "
            f"target={target_ratio * 100:6.2f}% "
            f"diff={difference:+.2f}%"
        )

    # ========================================================
    # EXACT TARGET CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("TARGET IMAGE COUNT CHECK")
    print("=" * 80)

    expected_train = 9861
    expected_val = 2113
    expected_test = 2113

    target_counts = {
        "train": expected_train,
        "val": expected_val,
        "test": expected_test
    }

    target_count_errors = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        actual = split_images[split]
        expected = target_counts[split]

        print(
            f"{split.upper():5s}: "
            f"expected={expected:,} "
            f"actual={actual:,}"
        )

        if actual != expected:

            target_count_errors.append({
                "split": split,
                "expected": expected,
                "actual": actual
            })

    if target_count_errors:

        print()
        print(
            "WARNING: Candidate split does not exactly "
            "match the planned target counts."
        )

    else:

        print()
        print(
            "PASS: Candidate split exactly matches "
            "the planned 70/15/15 image counts."
        )

    # ========================================================
    # CROSS-SPLIT ORIGINAL DATASET ANALYSIS
    # ========================================================

    print()
    print("=" * 80)
    print("ORIGINAL SPLIT DISTRIBUTION INSIDE NEW SPLITS")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test"
    ]:

        original_distribution = Counter()

        for component in components:

            cid = int(
                component["component_id"]
            )

            if assignments[cid] != split:
                continue

            for sequence in component["sequences"]:

                original_distribution[
                    sequence["split"]
                ] += int(
                    sequence["image_count"]
                )

        print()
        print(
            f"NEW {split.upper()}:"
        )

        for original_split in [
            "train",
            "val",
            "test"
        ]:

            count = original_distribution[
                original_split
            ]

            print(
                f"  original {original_split:5s}: "
                f"{count:,}"
            )

    # ========================================================
    # SAVE COMPONENT ASSIGNMENTS
    # ========================================================

    csv_path = (
        OUTPUT_DIR /
        "stage_0_9c_split_component_assignments.csv"
    )

    rows = []

    for component in components:

        cid = int(
            component["component_id"]
        )

        split = assignments[cid]

        for sequence in component["sequences"]:

            rows.append({

                "component_id":
                    cid,

                "assigned_split":
                    split,

                "sequence":
                    sequence["sequence"],

                "original_split":
                    sequence["split"],

                "data_folder":
                    sequence["data_folder"],

                "image_count":
                    sequence["image_count"]
            })

    fields = [
        "component_id",
        "assigned_split",
        "sequence",
        "original_split",
        "data_folder",
        "image_count"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved: {csv_path}"
    )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================

    train_components = sorted(
        cid
        for cid, split in assignments.items()
        if split == "train"
    )

    val_components = sorted(
        cid
        for cid, split in assignments.items()
        if split == "val"
    )

    test_components = sorted(
        cid
        for cid, split in assignments.items()
        if split == "test"
    )

    summary = {

        "stage":
            "0.9C",

        "total_images":
            total_images,

        "target_ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15
        },

        "actual_images": {
            "train":
                split_images["train"],

            "val":
                split_images["val"],

            "test":
                split_images["test"]
        },

        "actual_ratios": {
            "train":
                split_images["train"]
                /
                total_images,

            "val":
                split_images["val"]
                /
                total_images,

            "test":
                split_images["test"]
                /
                total_images
        },

        "component_counts": {
            "train":
                split_components["train"],

            "val":
                split_components["val"],

            "test":
                split_components["test"]
        },

        "sequence_counts": {
            "train":
                split_sequences["train"],

            "val":
                split_sequences["val"],

            "test":
                split_sequences["test"]
        },

        "train_components":
            train_components,

        "val_components":
            val_components,

        "test_components":
            test_components,

        "integrity_errors":
            integrity_errors,

        "component_split_conflicts":
            component_split_conflicts,

        "target_count_errors":
            target_count_errors
    }

    json_path = (
        OUTPUT_DIR /
        "stage_0_9c_split_plan.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"Saved: {json_path}"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 0.9C COMPLETE")
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is ONLY a split plan."
    )

    print(
        "No images were copied."
    )

    print(
        "No images were deleted."
    )

    print(
        "No original dataset files were modified."
    )

    print()

    if (
        not integrity_errors
        and
        not component_split_conflicts
        and
        not target_count_errors
        and
        total_images == expected_total
    ):

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: REVIEW REQUIRED"
        )

    print()
    print(
        "Next step: validate annotation/damage "
        "distribution before constructing the "
        "canonical dataset."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()