import csv
import json
import random
from pathlib import Path
from collections import defaultdict


# ============================================================
# STAGE 0.9D.1
# LEAKAGE-SAFE BALANCED SPLIT OPTIMIZATION
#
# Purpose:
# Optimize the assignment of the 20 duplicate-connected
# components into TRAIN / VAL / TEST.
#
# IMPORTANT:
# - Package 1 is the canonical annotation authority.
# - Components remain intact.
# - Sequences remain intact.
# - Exact duplicate-connected components remain intact.
# - No images are copied.
# - No images are moved.
# - No images are deleted.
# - Raw dataset is not modified.
#
# This stage ONLY creates a proposed split plan.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

AUDIT_ROOT = PROJECT_ROOT / "audit_outputs"

INVENTORY_FILE = (
    AUDIT_ROOT /
    "stage_0_9_canonical_image_inventory.csv"
)

COMPONENT_FILE = (
    AUDIT_ROOT /
    "stage_0_9b_corrected_components.json"
)

BASELINE_FILE = (
    AUDIT_ROOT /
    "stage_0_9c_split_plan.json"
)

OUTPUT_PLAN = (
    AUDIT_ROOT /
    "stage_0_9d1_optimized_split_plan.json"
)

OUTPUT_COMPONENTS = (
    AUDIT_ROOT /
    "stage_0_9d1_component_assignments.csv"
)

OUTPUT_SUMMARY = (
    AUDIT_ROOT /
    "stage_0_9d1_optimization_summary.json"
)


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

SPLITS = [
    "train",
    "val",
    "test"
]

TARGET_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

RANDOM_SEED = 42

N_RESTARTS = 500

LOCAL_SEARCH_ITERATIONS = 1000


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

WEIGHT_COUNT = 8.0

WEIGHT_SHAPE = 5.0

WEIGHT_MEAN_CRACKS = 5.0

WEIGHT_RESOLUTION = 5.0

WEIGHT_ORIENTATION = 2.0


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_distribution(counter, categories):

    total = sum(counter.values())

    if total <= 0:

        return {
            category: 0.0
            for category in categories
        }

    return {
        category:
            counter.get(category, 0) / total
        for category in categories
    }


def squared_distribution_error(
    actual,
    target,
    categories
):

    return sum(
        (
            actual.get(category, 0.0)
            - target.get(category, 0.0)
        ) ** 2
        for category in categories
    )


def absolute_distribution_error(
    actual,
    target,
    categories
):

    return sum(
        abs(
            actual.get(category, 0.0)
            - target.get(category, 0.0)
        )
        for category in categories
    )


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory():

    if not INVENTORY_FILE.exists():

        raise FileNotFoundError(
            "Missing Stage 0.9 inventory:\n"
            + str(INVENTORY_FILE)
        )

    records = []

    with open(
        INVENTORY_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            records.append(row)

    if not records:

        raise ValueError(
            "Stage 0.9 inventory is empty."
        )

    return records


# ============================================================
# LOAD COMPONENT JSON
# ============================================================

def load_components():

    if not COMPONENT_FILE.exists():

        raise FileNotFoundError(
            "Missing Stage 0.9B component file:\n"
            + str(COMPONENT_FILE)
        )

    with open(
        COMPONENT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# NORMALIZE SEQUENCE NAME
# ============================================================

def normalize_sequence_name(value):

    """
    Convert a sequence representation into:

        train/data1
        train/data10
        val/data19
        test/data24

    Handles:
        string
        dictionary containing "sequence"
        dictionary containing split + data_folder
    """

    if value is None:
        return ""

    # --------------------------------------------------------
    # ACTUAL STAGE 0.9B STRUCTURE
    # --------------------------------------------------------

    if isinstance(value, dict):

        if value.get("sequence"):

            return normalize_sequence_name(
                value.get("sequence")
            )

        split = value.get("split")

        data_folder = value.get(
            "data_folder"
        )

        if split and data_folder:

            return (
                str(split).strip()
                + "/"
                + str(data_folder).strip()
            )

        return ""

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    text = str(value).strip()

    text = text.replace(
        "\\",
        "/"
    )

    while "//" in text:

        text = text.replace(
            "//",
            "/"
        )

    text = text.strip("/")

    parts = [
        p for p in text.split("/")
        if p
    ]

    # Already canonical.
    for split in SPLITS:

        if split in parts:

            idx = parts.index(
                split
            )

            if idx + 1 < len(parts):

                return (
                    split
                    + "/"
                    + parts[idx + 1]
                )

    return text


# ============================================================
# EXTRACT COMPONENT MAPPING
# ============================================================

def extract_component_mapping(
    component_data
):

    """
    Extract:

        component_id -> sequence list

    from Stage 0.9B.

    IMPORTANT:
    Stage 0.9B stores sequence members as dictionaries,
    e.g.

        {
            "sequence": "test/data24",
            "split": "test",
            "data_folder": "data24",
            "image_count": 623
        }

    We explicitly extract the "sequence" field.
    """

    mapping = {}

    # --------------------------------------------------------
    # Case 1:
    # components is a list
    # --------------------------------------------------------

    components = component_data.get(
        "components"
    )

    if isinstance(components, list):

        for component in components:

            if not isinstance(
                component,
                dict
            ):
                continue

            cid = (
                component.get(
                    "component_id"
                )
                or component.get(
                    "component"
                )
                or component.get(
                    "id"
                )
            )

            members = (
                component.get(
                    "sequences"
                )
                or component.get(
                    "sequence_ids"
                )
                or component.get(
                    "members"
                )
            )

            if cid is None:
                continue

            if not isinstance(
                members,
                list
            ):
                continue

            try:
                cid = int(cid)

            except Exception:
                continue

            sequence_list = []

            for member in members:

                sequence = normalize_sequence_name(
                    member
                )

                if sequence:

                    sequence_list.append(
                        sequence
                    )

            mapping[cid] = (
                sequence_list
            )

    # --------------------------------------------------------
    # Case 2:
    # dictionary of components
    # --------------------------------------------------------

    if not mapping:

        possible = (
            component_data.get(
                "component_members"
            )
            or component_data.get(
                "components_by_id"
            )
            or component_data.get(
                "connected_components"
            )
        )

        if isinstance(
            possible,
            dict
        ):

            for cid, members in (
                possible.items()
            ):

                if not isinstance(
                    members,
                    list
                ):
                    continue

                try:
                    cid = int(cid)

                except Exception:
                    continue

                sequence_list = []

                for member in members:

                    sequence = normalize_sequence_name(
                        member
                    )

                    if sequence:

                        sequence_list.append(
                            sequence
                        )

                mapping[cid] = (
                    sequence_list
                )

    # --------------------------------------------------------
    # Case 3:
    # sequence -> component
    # --------------------------------------------------------

    if not mapping:

        possible = (
            component_data.get(
                "sequence_to_component"
            )
            or component_data.get(
                "sequence_component_map"
            )
        )

        if isinstance(
            possible,
            dict
        ):

            reverse = defaultdict(
                list
            )

            for sequence, cid in (
                possible.items()
            ):

                normalized = (
                    normalize_sequence_name(
                        sequence
                    )
                )

                if not normalized:
                    continue

                try:
                    cid = int(cid)

                except Exception:
                    continue

                reverse[cid].append(
                    normalized
                )

            mapping = dict(
                reverse
            )

    if not mapping:

        raise ValueError(
            "Could not extract connected "
            "components from:\n"
            + str(COMPONENT_FILE)
        )

    return mapping


# ============================================================
# BUILD SEQUENCE -> COMPONENT
# ============================================================

def build_sequence_to_component(
    component_mapping
):

    sequence_to_component = {}

    for cid, sequences in (
        component_mapping.items()
    ):

        for sequence in sequences:

            sequence = normalize_sequence_name(
                sequence
            )

            if not sequence:
                continue

            if sequence in (
                sequence_to_component
            ):

                previous_component = (
                    sequence_to_component[
                        sequence
                    ]
                )

                if (
                    previous_component
                    != cid
                ):

                    raise ValueError(
                        "Sequence belongs "
                        "to multiple components: "
                        + sequence
                    )

            sequence_to_component[
                sequence
            ] = cid

    return sequence_to_component


# ============================================================
# EXTRACT SEQUENCE FROM INVENTORY
# ============================================================

def get_sequence_from_row(row):

    """
    Recover exact sequence identity from
    Stage 0.9 inventory.

    Expected:
        test/data24
        train/data10
        val/data19
    """

    # --------------------------------------------------------
    # Explicit sequence_id
    # --------------------------------------------------------

    for key in [
        "sequence_id",
        "sequence",
        "sequence_name"
    ]:

        value = row.get(
            key
        )

        if value:

            sequence = (
                normalize_sequence_name(
                    value
                )
            )

            if (
                sequence
                and "/" in sequence
            ):

                return sequence

    # --------------------------------------------------------
    # Derive from image_path
    # --------------------------------------------------------

    image_path = (
        row.get(
            "image_path"
        )
        or row.get(
            "relative_path"
        )
        or row.get(
            "path"
        )
        or ""
    )

    image_path = image_path.replace(
        "\\",
        "/"
    )

    parts = [
        p for p in image_path.split("/")
        if p
    ]

    for split in SPLITS:

        if split not in parts:
            continue

        split_index = parts.index(
            split
        )

        after_split = parts[
            split_index + 1:
        ]

        # Search for dataN.
        for candidate in after_split:

            lower = candidate.lower()

            if lower in {
                "images",
                "annotations",
                "annotation",
                "json",
                "labels"
            }:

                continue

            if "." in candidate:

                continue

            return (
                split
                + "/"
                + candidate
            )

    return ""


# ============================================================
# BUILD COMPONENT STATISTICS
# ============================================================

def build_component_statistics(
    inventory,
    sequence_to_component
):

    component_records = (
        defaultdict(list)
    )

    unknown_sequences = (
        defaultdict(int)
    )

    # --------------------------------------------------------
    # Map every inventory record
    # --------------------------------------------------------

    for row in inventory:

        sequence = (
            get_sequence_from_row(
                row
            )
        )

        if not sequence:

            unknown_sequences[
                "<EMPTY_SEQUENCE>"
            ] += 1

            continue

        sequence = (
            normalize_sequence_name(
                sequence
            )
        )

        if sequence not in (
            sequence_to_component
        ):

            unknown_sequences[
                sequence
            ] += 1

            continue

        cid = (
            sequence_to_component[
                sequence
            ]
        )

        component_records[
            cid
        ].append(row)

    # --------------------------------------------------------
    # Verify mapping
    # --------------------------------------------------------

    if unknown_sequences:

        print()
        print(
            "ERROR: Inventory-to-component "
            "mapping is incomplete."
        )

        print()

        print(
            "Unmapped sequence records:"
        )

        for sequence, count in sorted(
            unknown_sequences.items()
        ):

            print(
                f"  {sequence}: "
                f"{count:,} records"
            )

        print()

        print(
            "Known component sequences:"
        )

        for sequence in sorted(
            sequence_to_component
        ):

            print(
                f"  {sequence} -> "
                f"component "
                f"{sequence_to_component[sequence]}"
            )

        raise ValueError(
            "Inventory-to-component "
            "mapping is incomplete."
        )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    all_shape_counts = (
        defaultdict(int)
    )

    all_resolution_counts = (
        defaultdict(int)
    )

    all_orientation_counts = (
        defaultdict(int)
    )

    total_images = 0

    total_cracks = 0

    for row in inventory:

        crack_count = safe_int(
            row.get(
                "shape_count"
            )
            or row.get(
                "crack_count"
            )
            or row.get(
                "num_shapes"
            )
        )

        width = safe_int(
            row.get(
                "width"
            )
        )

        height = safe_int(
            row.get(
                "height"
            )
        )

        resolution = (
            f"{width}x{height}"
        )

        if (
            width > 0
            and height > 0
        ):

            if width >= height:

                orientation = (
                    "landscape"
                )

            else:

                orientation = (
                    "portrait"
                )

        else:

            orientation = "unknown"

        all_shape_counts[
            crack_count
        ] += 1

        all_resolution_counts[
            resolution
        ] += 1

        all_orientation_counts[
            orientation
        ] += 1

        total_images += 1

        total_cracks += (
            crack_count
        )

    shape_categories = sorted(
        all_shape_counts.keys()
    )

    resolution_categories = sorted(
        all_resolution_counts.keys()
    )

    orientation_categories = sorted(
        all_orientation_counts.keys()
    )

    global_stats = {

        "image_count":
            total_images,

        "crack_total":
            total_cracks,

        "mean_cracks":
            (
                total_cracks
                / total_images
                if total_images > 0
                else 0.0
            ),

        "shape_distribution":
            normalize_distribution(
                all_shape_counts,
                shape_categories
            ),

        "resolution_distribution":
            normalize_distribution(
                all_resolution_counts,
                resolution_categories
            ),

        "orientation_distribution":
            normalize_distribution(
                all_orientation_counts,
                orientation_categories
            ),

        "shape_categories":
            shape_categories,

        "resolution_categories":
            resolution_categories,

        "orientation_categories":
            orientation_categories
    }

    # --------------------------------------------------------
    # Per-component statistics
    # --------------------------------------------------------

    component_stats = {}

    for cid, rows in (
        component_records.items()
    ):

        shape_counts = (
            defaultdict(int)
        )

        resolution_counts = (
            defaultdict(int)
        )

        orientation_counts = (
            defaultdict(int)
        )

        crack_total = 0

        for row in rows:

            crack_count = safe_int(
                row.get(
                    "shape_count"
                )
                or row.get(
                    "crack_count"
                )
                or row.get(
                    "num_shapes"
                )
            )

            width = safe_int(
                row.get(
                    "width"
                )
            )

            height = safe_int(
                row.get(
                    "height"
                )
            )

            resolution = (
                f"{width}x{height}"
            )

            if (
                width > 0
                and height > 0
            ):

                if width >= height:

                    orientation = (
                        "landscape"
                    )

                else:

                    orientation = (
                        "portrait"
                    )

            else:

                orientation = "unknown"

            shape_counts[
                crack_count
            ] += 1

            resolution_counts[
                resolution
            ] += 1

            orientation_counts[
                orientation
            ] += 1

            crack_total += (
                crack_count
            )

        image_count = len(rows)

        component_stats[cid] = {

            "image_count":
                image_count,

            "crack_total":
                crack_total,

            "mean_cracks":
                (
                    crack_total
                    / image_count
                    if image_count > 0
                    else 0.0
                ),

            "shape_counts":
                dict(shape_counts),

            "resolution_counts":
                dict(resolution_counts),

            "orientation_counts":
                dict(orientation_counts)
        }

    return (
        component_stats,
        global_stats
    )


# ============================================================
# SPLIT STATISTICS
# ============================================================

def calculate_split_statistics(
    assignment,
    component_stats,
    global_stats
):

    split_stats = {}

    for split in SPLITS:

        component_ids = [
            cid
            for cid, assigned_split
            in assignment.items()
            if assigned_split == split
        ]

        image_count = 0

        crack_total = 0

        shape_counts = (
            defaultdict(int)
        )

        resolution_counts = (
            defaultdict(int)
        )

        orientation_counts = (
            defaultdict(int)
        )

        for cid in component_ids:

            stat = (
                component_stats[cid]
            )

            image_count += (
                stat["image_count"]
            )

            crack_total += (
                stat["crack_total"]
            )

            for key, value in (
                stat[
                    "shape_counts"
                ].items()
            ):

                shape_counts[
                    key
                ] += value

            for key, value in (
                stat[
                    "resolution_counts"
                ].items()
            ):

                resolution_counts[
                    key
                ] += value

            for key, value in (
                stat[
                    "orientation_counts"
                ].items()
            ):

                orientation_counts[
                    key
                ] += value

        mean_cracks = (
            crack_total
            / image_count
            if image_count > 0
            else 0.0
        )

        split_stats[split] = {

            "components":
                sorted(
                    component_ids
                ),

            "image_count":
                image_count,

            "ratio":
                (
                    image_count
                    / global_stats[
                        "image_count"
                    ]
                ),

            "crack_total":
                crack_total,

            "mean_cracks":
                mean_cracks,

            "shape_counts":
                dict(
                    shape_counts
                ),

            "shape_distribution":
                normalize_distribution(
                    shape_counts,
                    global_stats[
                        "shape_categories"
                    ]
                ),

            "resolution_counts":
                dict(
                    resolution_counts
                ),

            "resolution_distribution":
                normalize_distribution(
                    resolution_counts,
                    global_stats[
                        "resolution_categories"
                    ]
                ),

            "orientation_counts":
                dict(
                    orientation_counts
                ),

            "orientation_distribution":
                normalize_distribution(
                    orientation_counts,
                    global_stats[
                        "orientation_categories"
                    ]
                )
        }

    return split_stats


# ============================================================
# OBJECTIVE
# ============================================================

def calculate_objective(
    assignment,
    component_stats,
    global_stats
):

    split_stats = (
        calculate_split_statistics(
            assignment,
            component_stats,
            global_stats
        )
    )

    score = 0.0

    total_images = (
        global_stats[
            "image_count"
        ]
    )

    global_mean = (
        global_stats[
            "mean_cracks"
        ]
    )

    for split in SPLITS:

        stats = (
            split_stats[split]
        )

        # ----------------------------------------------------
        # Split size
        # ----------------------------------------------------

        target_count = (
            TARGET_RATIOS[split]
            * total_images
        )

        count_error = (
            abs(
                stats[
                    "image_count"
                ]
                - target_count
            )
            / total_images
        )

        score += (
            WEIGHT_COUNT
            * count_error
        )

        # ----------------------------------------------------
        # Shape distribution
        # ----------------------------------------------------

        shape_error = (
            squared_distribution_error(
                stats[
                    "shape_distribution"
                ],
                global_stats[
                    "shape_distribution"
                ],
                global_stats[
                    "shape_categories"
                ]
            )
        )

        score += (
            WEIGHT_SHAPE
            * shape_error
        )

        # ----------------------------------------------------
        # Mean cracks
        # ----------------------------------------------------

        mean_error = (
            abs(
                stats[
                    "mean_cracks"
                ]
                - global_mean
            )
            / max(
                global_mean,
                1e-9
            )
        )

        score += (
            WEIGHT_MEAN_CRACKS
            * mean_error
        )

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        resolution_error = (
            squared_distribution_error(
                stats[
                    "resolution_distribution"
                ],
                global_stats[
                    "resolution_distribution"
                ],
                global_stats[
                    "resolution_categories"
                ]
            )
        )

        score += (
            WEIGHT_RESOLUTION
            * resolution_error
        )

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        orientation_error = (
            squared_distribution_error(
                stats[
                    "orientation_distribution"
                ],
                global_stats[
                    "orientation_distribution"
                ],
                global_stats[
                    "orientation_categories"
                ]
            )
        )

        score += (
            WEIGHT_ORIENTATION
            * orientation_error
        )

    return (
        score,
        split_stats
    )


# ============================================================
# INITIAL ASSIGNMENT
# ============================================================

def create_initial_assignment(
    component_ids,
    component_stats,
    rng
):

    ids = list(
        component_ids
    )

    rng.shuffle(
        ids
    )

    ids.sort(
        key=lambda cid:
            component_stats[cid][
                "image_count"
            ],
        reverse=True
    )

    total_images = sum(
        component_stats[cid][
            "image_count"
        ]
        for cid in component_ids
    )

    targets = {

        split:
            TARGET_RATIOS[split]
            * total_images

        for split in SPLITS
    }

    current_counts = {
        split: 0
        for split in SPLITS
    }

    assignment = {}

    for cid in ids:

        size = (
            component_stats[cid][
                "image_count"
            ]
        )

        best_split = None

        best_cost = float(
            "inf"
        )

        for split in SPLITS:

            projected = (
                current_counts[
                    split
                ]
                + size
            )

            cost = abs(
                projected
                - targets[
                    split
                ]
            )

            if cost < best_cost:

                best_cost = cost

                best_split = split

        assignment[
            cid
        ] = best_split

        current_counts[
            best_split
        ] += size

    return assignment


# ============================================================
# LOCAL OPTIMIZATION
# ============================================================

def optimize_assignment(
    initial_assignment,
    component_stats,
    global_stats,
    rng
):

    current = dict(
        initial_assignment
    )

    current_score, _ = (
        calculate_objective(
            current,
            component_stats,
            global_stats
        )
    )

    component_ids = list(
        component_stats.keys()
    )

    # --------------------------------------------------------
    # Single component moves
    # --------------------------------------------------------

    for _ in range(
        LOCAL_SEARCH_ITERATIONS
    ):

        cid = rng.choice(
            component_ids
        )

        old_split = (
            current[cid]
        )

        alternatives = [
            split
            for split in SPLITS
            if split != old_split
        ]

        new_split = rng.choice(
            alternatives
        )

        current[cid] = (
            new_split
        )

        new_score, _ = (
            calculate_objective(
                current,
                component_stats,
                global_stats
            )
        )

        if new_score <= current_score:

            current_score = (
                new_score
            )

        else:

            current[cid] = (
                old_split
            )

    # --------------------------------------------------------
    # Pairwise swaps
    # --------------------------------------------------------

    improved = True

    while improved:

        improved = False

        for i in range(
            len(component_ids)
        ):

            for j in range(
                i + 1,
                len(component_ids)
            ):

                cid_a = (
                    component_ids[i]
                )

                cid_b = (
                    component_ids[j]
                )

                split_a = (
                    current[cid_a]
                )

                split_b = (
                    current[cid_b]
                )

                if split_a == split_b:
                    continue

                current[cid_a] = (
                    split_b
                )

                current[cid_b] = (
                    split_a
                )

                new_score, _ = (
                    calculate_objective(
                        current,
                        component_stats,
                        global_stats
                    )
                )

                if new_score < current_score:

                    current_score = (
                        new_score
                    )

                    improved = True

                else:

                    current[cid_a] = (
                        split_a
                    )

                    current[cid_b] = (
                        split_b
                    )

    return (
        current,
        current_score
    )


# ============================================================
# LOAD BASELINE
# ============================================================

def load_baseline():

    if not BASELINE_FILE.exists():
        return None

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# RECONSTRUCT BASELINE
# ============================================================

def reconstruct_baseline(
    baseline,
    component_ids
):

    if baseline is None:
        return None

    assignment = {}

    for split in SPLITS:

        key = (
            split
            + "_components"
        )

        components = (
            baseline.get(
                key,
                []
            )
        )

        for cid in components:

            try:
                cid = int(cid)

            except Exception:
                continue

            assignment[
                cid
            ] = split

    if set(
        assignment.keys()
    ) != set(
        component_ids
    ):

        return None

    return assignment


# ============================================================
# WRITE COMPONENT CSV
# ============================================================

def write_component_csv(
    assignment,
    component_stats
):

    fieldnames = [
        "component_id",
        "assigned_split",
        "image_count",
        "crack_total",
        "mean_cracks",
        "shape_counts",
        "resolution_counts",
        "orientation_counts"
    ]

    with open(
        OUTPUT_COMPONENTS,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for cid in sorted(
            component_stats
        ):

            stat = (
                component_stats[cid]
            )

            writer.writerow({

                "component_id":
                    cid,

                "assigned_split":
                    assignment[cid],

                "image_count":
                    stat[
                        "image_count"
                    ],

                "crack_total":
                    stat[
                        "crack_total"
                    ],

                "mean_cracks":
                    (
                        f"{stat['mean_cracks']:.6f}"
                    ),

                "shape_counts":
                    json.dumps(
                        stat[
                            "shape_counts"
                        ],
                        sort_keys=True
                    ),

                "resolution_counts":
                    json.dumps(
                        stat[
                            "resolution_counts"
                        ],
                        sort_keys=True
                    ),

                "orientation_counts":
                    json.dumps(
                        stat[
                            "orientation_counts"
                        ],
                        sort_keys=True
                    )
            })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STAGE 0.9D.1 - "
        "BALANCED SPLIT OPTIMIZATION"
    )
    print("=" * 72)

    print()
    print(
        "This stage ONLY optimizes "
        "component assignments."
    )

    print(
        "No images will be copied, moved, "
        "or deleted."
    )

    # --------------------------------------------------------
    # LOAD INVENTORY
    # --------------------------------------------------------

    print()
    print(
        "Loading Stage 0.9 inventory..."
    )

    inventory = load_inventory()

    print(
        f"  Inventory records: "
        f"{len(inventory):,}"
    )

    # --------------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------------

    print()
    print(
        "Loading Stage 0.9B component audit..."
    )

    component_data = (
        load_components()
    )

    component_mapping = (
        extract_component_mapping(
            component_data
        )
    )

    print(
        f"  Components found: "
        f"{len(component_mapping):,}"
    )

    # --------------------------------------------------------
    # SEQUENCE -> COMPONENT
    # --------------------------------------------------------

    sequence_to_component = (
        build_sequence_to_component(
            component_mapping
        )
    )

    print(
        f"  Sequences mapped: "
        f"{len(sequence_to_component):,}"
    )

    # --------------------------------------------------------
    # VERIFY EXPECTED STRUCTURE
    # --------------------------------------------------------

    if len(component_mapping) != 20:

        raise ValueError(
            "Expected 20 components, found "
            + str(
                len(component_mapping)
            )
        )

    if len(sequence_to_component) != 29:

        raise ValueError(
            "Expected 29 sequences, found "
            + str(
                len(sequence_to_component)
            )
        )

    # --------------------------------------------------------
    # PRINT CLEAN SEQUENCE MAPPING
    # --------------------------------------------------------

    print()
    print(
        "Verified sequence identities:"
    )

    for sequence in sorted(
        sequence_to_component
    ):

        print(
            f"  {sequence} -> "
            f"component "
            f"{sequence_to_component[sequence]}"
        )

    # --------------------------------------------------------
    # BUILD STATS
    # --------------------------------------------------------

    print()
    print(
        "Building component statistics..."
    )

    (
        component_stats,
        global_stats
    ) = build_component_statistics(
        inventory,
        sequence_to_component
    )

    component_ids = sorted(
        component_stats.keys()
    )

    print(
        f"  Components with inventory: "
        f"{len(component_ids):,}"
    )

    if len(component_ids) != 20:

        raise ValueError(
            "Expected statistics for 20 "
            "components, found "
            + str(
                len(component_ids)
            )
        )

    # --------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "GLOBAL DATASET"
    )

    print(
        f"  Images: "
        f"{global_stats['image_count']:,}"
    )

    print(
        f"  Total crack polygons: "
        f"{global_stats['crack_total']:,}"
    )

    print(
        f"  Mean cracks/image: "
        f"{global_stats['mean_cracks']:.4f}"
    )

    # --------------------------------------------------------
    # GLOBAL RESOLUTION
    # --------------------------------------------------------

    print()
    print(
        "GLOBAL RESOLUTION DISTRIBUTION"
    )

    for category in (
        global_stats[
            "resolution_categories"
        ]
    ):

        percentage = (
            global_stats[
                "resolution_distribution"
            ][category]
            * 100
        )

        print(
            f"  {category}: "
            f"{percentage:.2f}%"
        )

    # --------------------------------------------------------
    # COMPONENT TABLE
    # --------------------------------------------------------

    print()
    print(
        "COMPONENT STATISTICS"
    )

    print(
        f"{'ID':>4} "
        f"{'Images':>8} "
        f"{'MeanCracks':>11}"
    )

    print(
        "-" * 32
    )

    for cid in component_ids:

        stat = (
            component_stats[cid]
        )

        print(
            f"{cid:>4} "
            f"{stat['image_count']:>8,} "
            f"{stat['mean_cracks']:>11.4f}"
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = load_baseline()

    baseline_assignment = (
        reconstruct_baseline(
            baseline,
            component_ids
        )
    )

    baseline_score = None

    if baseline_assignment is not None:

        (
            baseline_score,
            _
        ) = calculate_objective(
            baseline_assignment,
            component_stats,
            global_stats
        )

        print()
        print(
            "Stage 0.9C baseline objective:"
        )

        print(
            f"  {baseline_score:.8f}"
        )

    else:

        print()
        print(
            "Stage 0.9C baseline could not "
            "be reconstructed."
        )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    print()
    print(
        f"Running {N_RESTARTS} "
        f"optimization restarts..."
    )

    master_rng = random.Random(
        RANDOM_SEED
    )

    best_assignment = None

    best_score = float(
        "inf"
    )

    best_stats = None

    for restart in range(
        N_RESTARTS
    ):

        seed = (
            master_rng.randint(
                0,
                2_000_000_000
            )
        )

        rng = random.Random(
            seed
        )

        initial_assignment = (
            create_initial_assignment(
                component_ids,
                component_stats,
                rng
            )
        )

        (
            candidate_assignment,
            candidate_score
        ) = optimize_assignment(
            initial_assignment,
            component_stats,
            global_stats,
            rng
        )

        if (
            candidate_score
            < best_score
        ):

            best_score = (
                candidate_score
            )

            best_assignment = (
                candidate_assignment
            )

            (
                _,
                best_stats
            ) = calculate_objective(
                best_assignment,
                component_stats,
                global_stats
            )

        if (
            restart == 0
            or (restart + 1) % 50 == 0
        ):

            print(
                f"  Restart "
                f"{restart + 1}/"
                f"{N_RESTARTS} "
                f"best score = "
                f"{best_score:.8f}"
            )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if best_assignment is None:

        raise RuntimeError(
            "Optimization produced no "
            "valid assignment."
        )

    if set(
        best_assignment.keys()
    ) != set(
        component_ids
    ):

        raise RuntimeError(
            "Not every component was "
            "assigned."
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "OPTIMIZATION RESULT"
    )
    print("=" * 72)

    print()
    print(
        f"Optimized objective: "
        f"{best_score:.8f}"
    )

    if baseline_score is not None:

        if baseline_score != 0:

            improvement = (
                (
                    baseline_score
                    - best_score
                )
                / baseline_score
                * 100
            )

        else:

            improvement = 0.0

        print(
            f"Baseline objective:  "
            f"{baseline_score:.8f}"
        )

        print(
            f"Objective improvement: "
            f"{improvement:.2f}%"
        )

    # --------------------------------------------------------
    # SPLIT RESULTS
    # --------------------------------------------------------

    print()

    for split in SPLITS:

        stat = (
            best_stats[split]
        )

        target_count = (
            TARGET_RATIOS[split]
            * global_stats[
                "image_count"
            ]
        )

        print(
            split.upper()
            + ":"
        )

        print(
            f"  Components: "
            f"{stat['components']}"
        )

        print(
            f"  Images: "
            f"{stat['image_count']:,} "
            f"/ target "
            f"{target_count:.1f}"
        )

        print(
            f"  Ratio: "
            f"{stat['ratio'] * 100:.2f}%"
        )

        print(
            f"  Total cracks: "
            f"{stat['crack_total']:,}"
        )

        print(
            f"  Mean cracks/image: "
            f"{stat['mean_cracks']:.4f}"
        )

    # --------------------------------------------------------
    # SHAPE DISTRIBUTION
    # --------------------------------------------------------

    print()
    print(
        "SHAPE DISTRIBUTION ERROR"
    )

    for split in SPLITS:

        error = (
            absolute_distribution_error(
                best_stats[
                    split
                ][
                    "shape_distribution"
                ],
                global_stats[
                    "shape_distribution"
                ],
                global_stats[
                    "shape_categories"
                ]
            )
        )

        print(
            f"  {split}: "
            f"{error:.6f}"
        )

    # --------------------------------------------------------
    # RESOLUTION DISTRIBUTION
    # --------------------------------------------------------

    print()
    print(
        "RESOLUTION DISTRIBUTION ERROR"
    )

    for split in SPLITS:

        error = (
            absolute_distribution_error(
                best_stats[
                    split
                ][
                    "resolution_distribution"
                ],
                global_stats[
                    "resolution_distribution"
                ],
                global_stats[
                    "resolution_categories"
                ]
            )
        )

        print(
            f"  {split}: "
            f"{error:.6f}"
        )

    # --------------------------------------------------------
    # ORIENTATION DISTRIBUTION
    # --------------------------------------------------------

    print()
    print(
        "ORIENTATION DISTRIBUTION"
    )

    for split in SPLITS:

        print(
            f"  {split}: "
            f"{best_stats[split]['orientation_distribution']}"
        )

    # --------------------------------------------------------
    # SAVE PLAN
    # --------------------------------------------------------

    output_plan = {

        "stage":
            "0.9D.1",

        "status":
            "PROPOSED",

        "description":
            (
                "Leakage-safe component-level "
                "split optimization."
            ),

        "random_seed":
            RANDOM_SEED,

        "n_restarts":
            N_RESTARTS,

        "local_search_iterations":
            LOCAL_SEARCH_ITERATIONS,

        "target_ratios":
            TARGET_RATIOS,

        "total_images":
            global_stats[
                "image_count"
            ],

        "optimized_objective":
            best_score,

        "baseline_objective":
            baseline_score,

        "train_components":
            best_stats[
                "train"
            ][
                "components"
            ],

        "val_components":
            best_stats[
                "val"
            ][
                "components"
            ],

        "test_components":
            best_stats[
                "test"
            ][
                "components"
            ],

        "actual_images": {

            split:
                best_stats[
                    split
                ][
                    "image_count"
                ]

            for split in SPLITS
        },

        "actual_ratios": {

            split:
                best_stats[
                    split
                ][
                    "ratio"
                ]

            for split in SPLITS
        },

        "split_statistics":
            best_stats,

        "global_statistics":
            global_stats,

        "constraints": {

            "component_integrity":
                True,

            "sequence_integrity":
                True,

            "duplicate_isolation":
                True,

            "raw_dataset_modified":
                False,

            "images_copied":
                False,

            "images_moved":
                False,

            "images_deleted":
                False
        }
    }

    with open(
        OUTPUT_PLAN,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_plan,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # COMPONENT CSV
    # --------------------------------------------------------

    write_component_csv(
        best_assignment,
        component_stats
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if (
        baseline_score is not None
        and baseline_score != 0
    ):

        improvement_percent = (
            (
                baseline_score
                - best_score
            )
            / baseline_score
            * 100
        )

    else:

        improvement_percent = None

    summary = {

        "stage":
            "0.9D.1",

        "status":
            "PROPOSED",

        "optimized_objective":
            best_score,

        "baseline_objective":
            baseline_score,

        "objective_improvement_percent":
            improvement_percent,

        "component_count":
            len(component_ids),

        "sequence_count":
            len(
                sequence_to_component
            ),

        "image_count":
            global_stats[
                "image_count"
            ],

        "optimized_split_counts": {

            split:
                best_stats[
                    split
                ][
                    "image_count"
                ]

            for split in SPLITS
        },

        "optimized_split_ratios": {

            split:
                best_stats[
                    split
                ][
                    "ratio"
                ]

            for split in SPLITS
        },

        "optimized_mean_cracks": {

            split:
                best_stats[
                    split
                ][
                    "mean_cracks"
                ]

            for split in SPLITS
        },

        "files_modified":
            False,

        "dataset_copied":
            False,

        "dataset_deleted":
            False
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"  {OUTPUT_PLAN}"
    )

    print(
        f"  {OUTPUT_COMPONENTS}"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print()
    print(
        "No raw dataset files were modified."
    )

    print(
        "No images were copied."
    )

    print(
        "No images were moved."
    )

    print(
        "No images were deleted."
    )

    print()
    print("=" * 72)
    print(
        "STAGE 0.9D.1 COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":

    main()