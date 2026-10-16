import os
import json
import csv
import hashlib
from collections import defaultdict, Counter

# ============================================================
# STAGE 0.9D.2
# OPTIMIZED SPLIT INTEGRITY VERIFICATION
# ============================================================

PROJECT_ROOT = r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"

AUDIT_DIR = os.path.join(PROJECT_ROOT, "audit_outputs")

INVENTORY_CSV = os.path.join(
    AUDIT_DIR,
    "stage_0_9_canonical_image_inventory.csv"
)

# Fallback if the actual inventory filename differs
if not os.path.exists(INVENTORY_CSV):
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

OPTIMIZED_ASSIGNMENTS_CSV = os.path.join(
    AUDIT_DIR,
    "stage_0_9d1_component_assignments.csv"
)

OUTPUT_JSON = os.path.join(
    AUDIT_DIR,
    "stage_0_9d2_verification.json"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_sequence_name(value):
    """
    Converts sequence representations into:

        split/dataN

    Handles:
    - strings
    - dictionaries from Stage 0.9B
    """
    if isinstance(value, dict):
        if "sequence" in value:
            return normalize_sequence_name(value["sequence"])

        split = str(value.get("split", "")).strip()
        data_folder = str(value.get("data_folder", "")).strip()

        if split and data_folder:
            return f"{split}/{data_folder}"

        return ""

    if value is None:
        return ""

    value = str(value).strip().replace("\\", "/")

    if value.startswith("./"):
        value = value[2:]

    return value


def find_column(fieldnames, candidates):
    lower_map = {
        str(x).strip().lower(): x
        for x in fieldnames
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def get_sequence_from_row(row):
    """
    Recover exact sequence identity from inventory row.
    """

    for key in ["sequence_id", "sequence", "sequence_name"]:
        if key in row and row[key]:
            seq = normalize_sequence_name(row[key])
            if "/" in seq:
                return seq

    split = ""
    data_folder = ""

    for key in ["split", "original_split"]:
        if key in row and row[key]:
            split = str(row[key]).strip()
            break

    for key in ["data_folder", "data", "folder"]:
        if key in row and row[key]:
            data_folder = str(row[key]).strip()
            break

    if split and data_folder:
        return f"{split}/{data_folder}"

    image_path = ""

    for key in ["image_path", "image", "path"]:
        if key in row and row[key]:
            image_path = str(row[key]).replace("\\", "/")
            break

    if image_path:
        parts = image_path.split("/")

        split_names = {"train", "val", "test"}

        for i, part in enumerate(parts):
            if part.lower() in split_names and i + 1 < len(parts):
                return f"{part.lower()}/{parts[i + 1]}"

    return ""


def get_int(row, candidates, default=0):
    for key in candidates:
        if key in row:
            try:
                return int(float(row[key]))
            except Exception:
                pass

    return default


def get_float(row, candidates, default=0.0):
    for key in candidates:
        if key in row:
            try:
                return float(row[key])
            except Exception:
                pass

    return default


def calculate_file_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("STAGE 0.9D.2 - OPTIMIZED SPLIT INTEGRITY VERIFICATION")
print("=" * 72)

print()
print("This stage ONLY verifies the Stage 0.9D.1 assignment.")
print("No images will be copied, moved, modified, or deleted.")
print()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    INVENTORY_CSV,
    COMPONENT_JSON,
    OPTIMIZED_PLAN_JSON,
    OPTIMIZED_ASSIGNMENTS_CSV
]

for path in required_files:
    if not os.path.exists(path):
        print("ERROR: Required file not found:")
        print("  " + path)
        raise SystemExit(1)


# ============================================================
# LOAD INVENTORY
# ============================================================

print("Loading Stage 0.9 inventory...")

inventory = []

with open(
    INVENTORY_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames or []

    for row in reader:
        inventory.append(row)

print(f"  Inventory records: {len(inventory):,}")

if len(inventory) != 14087:
    print("WARNING: Expected 14,087 inventory records.")


# ============================================================
# LOAD COMPONENT AUDIT
# ============================================================

print()
print("Loading Stage 0.9B component audit...")

with open(
    COMPONENT_JSON,
    "r",
    encoding="utf-8"
) as f:
    component_data = json.load(f)


# ------------------------------------------------------------
# Extract component members
# ------------------------------------------------------------

components = {}

raw_components = component_data.get(
    "components",
    component_data.get("connected_components", [])
)

if isinstance(raw_components, dict):

    for component_id, value in raw_components.items():

        try:
            cid = int(component_id)
        except Exception:
            continue

        members = []

        if isinstance(value, dict):
            raw_members = value.get(
                "sequences",
                value.get("members", [])
            )
        else:
            raw_members = value

        if isinstance(raw_members, list):
            for member in raw_members:
                seq = normalize_sequence_name(member)

                if seq:
                    members.append(seq)

        components[cid] = members

elif isinstance(raw_components, list):

    for index, value in enumerate(raw_components, start=1):

        cid = index
        members = []

        if isinstance(value, dict):

            if "component_id" in value:
                try:
                    cid = int(value["component_id"])
                except Exception:
                    pass

            raw_members = value.get(
                "sequences",
                value.get("members", [])
            )

        else:
            raw_members = value

        if isinstance(raw_members, list):

            for member in raw_members:

                seq = normalize_sequence_name(member)

                if seq:
                    members.append(seq)

        components[cid] = members


print(f"  Components found: {len(components)}")


# ============================================================
# BUILD SEQUENCE -> COMPONENT MAP
# ============================================================

sequence_to_component = {}

for component_id, members in components.items():

    for sequence in members:

        if sequence in sequence_to_component:

            print()
            print("ERROR: Sequence appears in multiple components:")
            print("  " + sequence)
            print(
                f"  Component {sequence_to_component[sequence]}"
                f" and Component {component_id}"
            )

            raise SystemExit(1)

        sequence_to_component[sequence] = component_id


print(
    f"  Sequences mapped: "
    f"{len(sequence_to_component)}"
)


# ============================================================
# LOAD OPTIMIZED PLAN
# ============================================================

print()
print("Loading Stage 0.9D.1 optimized plan...")

with open(
    OPTIMIZED_PLAN_JSON,
    "r",
    encoding="utf-8"
) as f:
    optimized_plan = json.load(f)


train_components = [
    int(x)
    for x in optimized_plan.get("train_components", [])
]

val_components = [
    int(x)
    for x in optimized_plan.get("val_components", [])
]

test_components = [
    int(x)
    for x in optimized_plan.get("test_components", [])
]


print("  TRAIN components:", train_components)
print("  VAL components  :", val_components)
print("  TEST components :", test_components)


# ============================================================
# COMPONENT SPLIT MAP
# ============================================================

component_to_split = {}

for cid in train_components:

    if cid in component_to_split:
        print("ERROR: Component assigned twice:", cid)
        raise SystemExit(1)

    component_to_split[cid] = "train"


for cid in val_components:

    if cid in component_to_split:
        print("ERROR: Component assigned twice:", cid)
        raise SystemExit(1)

    component_to_split[cid] = "val"


for cid in test_components:

    if cid in component_to_split:
        print("ERROR: Component assigned twice:", cid)
        raise SystemExit(1)

    component_to_split[cid] = "test"


# ============================================================
# CHECK COMPONENT COVERAGE
# ============================================================

print()
print("Checking component coverage...")

all_component_ids = set(components.keys())
assigned_component_ids = set(component_to_split.keys())

missing_components = (
    all_component_ids - assigned_component_ids
)

extra_components = (
    assigned_component_ids - all_component_ids
)

if missing_components:
    print("  ERROR: Unassigned components:", sorted(missing_components))
else:
    print("  All components assigned.")

if extra_components:
    print("  ERROR: Unknown components:", sorted(extra_components))


component_coverage_pass = (
    not missing_components
    and not extra_components
    and len(component_to_split) == len(all_component_ids)
)


# ============================================================
# MAP INVENTORY RECORDS
# ============================================================

print()
print("Mapping inventory records to optimized splits...")

sequence_records = defaultdict(list)

mapping_errors = []

for index, row in enumerate(inventory):

    sequence = get_sequence_from_row(row)

    if not sequence:
        mapping_errors.append(
            f"Row {index + 2}: sequence could not be determined"
        )
        continue

    if sequence not in sequence_to_component:

        mapping_errors.append(
            f"Row {index + 2}: unknown sequence {sequence}"
        )
        continue

    component_id = sequence_to_component[sequence]

    if component_id not in component_to_split:

        mapping_errors.append(
            f"Row {index + 2}: component {component_id} "
            f"has no split assignment"
        )
        continue

    split = component_to_split[component_id]

    sequence_records[sequence].append({
        "row": row,
        "split": split,
        "component": component_id
    })


print(
    f"  Successfully mapped records: "
    f"{sum(len(v) for v in sequence_records.values()):,}"
)

if mapping_errors:

    print()
    print("MAPPING ERRORS:")
    for error in mapping_errors[:20]:
        print("  " + error)

    if len(mapping_errors) > 20:
        print(
            f"  ... and "
            f"{len(mapping_errors) - 20} more"
        )


mapping_pass = (
    len(mapping_errors) == 0
    and sum(len(v) for v in sequence_records.values())
    == len(inventory)
)


# ============================================================
# SEQUENCE SPLIT INTEGRITY
# ============================================================

print()
print("Checking sequence integrity...")

sequence_split_sets = {}

sequence_split_conflicts = []

for sequence, records in sequence_records.items():

    splits = set(
        record["split"]
        for record in records
    )

    sequence_split_sets[sequence] = splits

    if len(splits) != 1:

        sequence_split_conflicts.append({
            "sequence": sequence,
            "splits": sorted(splits)
        })


print(
    f"  Sequences analyzed: "
    f"{len(sequence_split_sets)}"
)

print(
    f"  Sequence conflicts: "
    f"{len(sequence_split_conflicts)}"
)

sequence_integrity_pass = (
    len(sequence_split_conflicts) == 0
    and len(sequence_split_sets) == len(sequence_to_component)
)


# ============================================================
# COMPONENT SPLIT INTEGRITY
# ============================================================

print()
print("Checking component integrity...")

component_sequence_splits = defaultdict(set)

for sequence, component_id in sequence_to_component.items():

    if sequence in sequence_split_sets:

        for split in sequence_split_sets[sequence]:
            component_sequence_splits[component_id].add(split)


component_conflicts = []

for component_id, splits in component_sequence_splits.items():

    if len(splits) != 1:

        component_conflicts.append({
            "component": component_id,
            "splits": sorted(splits)
        })


print(
    f"  Component conflicts: "
    f"{len(component_conflicts)}"
)

component_integrity_pass = (
    len(component_conflicts) == 0
)


# ============================================================
# SPLIT COUNTS
# ============================================================

print()
print("CALCULATING OPTIMIZED SPLIT STATISTICS")

split_records = defaultdict(list)

for sequence, records in sequence_records.items():

    for record in records:
        split_records[record["split"]].append(record)


split_counts = {
    split: len(records)
    for split, records in split_records.items()
}

total_records = sum(split_counts.values())


print()
print("IMAGE COUNTS")

for split in ["train", "val", "test"]:

    count = split_counts.get(split, 0)

    ratio = (
        count / total_records
        if total_records
        else 0
    )

    print(
        f"  {split.upper():5s}: "
        f"{count:,} "
        f"({ratio * 100:.2f}%)"
    )

print(f"  TOTAL: {total_records:,}")


count_pass = (
    total_records == len(inventory)
    and split_counts.get("train", 0) == 9819
    and split_counts.get("val", 0) == 2102
    and split_counts.get("test", 0) == 2166
)


# ============================================================
# DUPLICATE HASH CHECK
# ============================================================

print()
print("Checking image-hash leakage using inventory...")

hash_column = find_column(
    fieldnames,
    [
        "sha256",
        "image_hash",
        "hash",
        "content_hash"
    ]
)

hash_split_map = defaultdict(set)

if hash_column:

    for records in split_records.values():

        for record in records:

            value = record["row"].get(hash_column, "").strip()

            if value:
                hash_split_map[value].add(
                    record["split"]
                )

    cross_split_hashes = {
        h: sorted(splits)
        for h, splits in hash_split_map.items()
        if len(splits) > 1
    }

    print(
        f"  Hashes analyzed: "
        f"{len(hash_split_map):,}"
    )

    print(
        f"  Cross-split hashes: "
        f"{len(cross_split_hashes):,}"
    )

else:

    print("  WARNING: No hash column found in inventory.")

    cross_split_hashes = {}

duplicate_isolation_pass = (
    hash_column is not None
    and len(cross_split_hashes) == 0
)


# ============================================================
# LABEL STATISTICS
# ============================================================

print()
print("LABEL STATISTICS")

shape_column = find_column(
    fieldnames,
    [
        "shape_count",
        "crack_count",
        "num_shapes",
        "polygon_count"
    ]
)

if shape_column:

    for split in ["train", "val", "test"]:

        values = []

        for record in split_records.get(split, []):

            values.append(
                get_int(
                    record["row"],
                    [shape_column],
                    0
                )
            )

        if values:

            mean_value = sum(values) / len(values)

            print(
                f"  {split.upper():5s}: "
                f"total polygons = {sum(values):,}, "
                f"mean = {mean_value:.4f}"
            )

else:

    print("  WARNING: Shape-count column not found.")


# ============================================================
# RESOLUTION STATISTICS
# ============================================================

print()
print("RESOLUTION DISTRIBUTION")

width_column = find_column(
    fieldnames,
    ["width", "image_width"]
)

height_column = find_column(
    fieldnames,
    ["height", "image_height"]
)

resolution_stats = {}

if width_column and height_column:

    for split in ["train", "val", "test"]:

        counter = Counter()

        for record in split_records.get(split, []):

            width = get_int(
                record["row"],
                [width_column]
            )

            height = get_int(
                record["row"],
                [height_column]
            )

            if width and height:

                counter[
                    f"{width}x{height}"
                ] += 1

        total = sum(counter.values())

        resolution_stats[split] = dict(counter)

        print()
        print(f"  {split.upper()}:")

        for resolution, count in sorted(
            counter.items(),
            key=lambda x: (-x[1], x[0])
        ):

            ratio = (
                count / total * 100
                if total
                else 0
            )

            print(
                f"    {resolution}: "
                f"{count:,} "
                f"({ratio:.2f}%)"
            )

else:

    print("  WARNING: Width/height columns not found.")


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 72)
print("FINAL VERIFICATION")
print("=" * 72)

checks = {
    "Component coverage": component_coverage_pass,
    "Inventory mapping": mapping_pass,
    "Sequence integrity": sequence_integrity_pass,
    "Component integrity": component_integrity_pass,
    "Split counts": count_pass,
    "Duplicate isolation": duplicate_isolation_pass
}

for name, passed in checks.items():

    print(
        f"  {name:28s}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


overall_pass = all(checks.values())


# ============================================================
# SAVE REPORT
# ============================================================

verification_report = {
    "stage": "0.9D.2",
    "status": "PASS" if overall_pass else "FAIL",

    "inventory_records": len(inventory),

    "component_count": len(components),

    "sequence_count": len(sequence_to_component),

    "optimized_components": {
        "train": train_components,
        "val": val_components,
        "test": test_components
    },

    "split_counts": split_counts,

    "mapping_errors": mapping_errors,

    "sequence_conflicts": sequence_split_conflicts,

    "component_conflicts": component_conflicts,

    "hash_column": hash_column,

    "cross_split_hash_count": len(
        cross_split_hashes
    ),

    "resolution_statistics": resolution_stats,

    "checks": checks,

    "overall_pass": overall_pass
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        verification_report,
        f,
        indent=2
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("Output:")
print("  " + OUTPUT_JSON)

print()

if overall_pass:

    print("=" * 72)
    print("STAGE 0.9D.2 PASS")
    print("=" * 72)
    print()
    print("The optimized split is structurally verified.")
    print("No sequence crosses train/val/test.")
    print("No component crosses train/val/test.")
    print("No exact image hash crosses train/val/test.")
    print()
    print("NEXT: STAGE 0.9E - CANONICAL DATASET CONSTRUCTION")
    print()

else:

    print("=" * 72)
    print("STAGE 0.9D.2 FAILED")
    print("=" * 72)
    print()
    print("Do NOT construct the canonical dataset yet.")
    print("Review the failed checks above.")
    print()