from pathlib import Path
import csv
import json
from collections import defaultdict, Counter


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


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory():

    records = []

    with open(
        INVENTORY_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            records.append(row)

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STAGE 0.9B - CORRECTED DUPLICATE-CONNECTED SEQUENCE AUDIT")
    print("=" * 80)

    if not INVENTORY_CSV.exists():

        print()
        print("ERROR:")
        print(
            f"Inventory not found:\n{INVENTORY_CSV}"
        )
        return

    records = load_inventory()

    print()
    print(
        f"Inventory records : {len(records):,}"
    )

    # ========================================================
    # BUILD SEQUENCE IDENTITIES
    # ========================================================

    sequences = set()

    sequence_images = Counter()

    hash_records = defaultdict(list)

    for record in records:

        split = record["split"].strip()

        data_folder = record[
            "data_folder"
        ].strip()

        sequence = (
            f"{split}/{data_folder}"
        )

        sequences.add(sequence)

        sequence_images[
            sequence
        ] += 1

        digest = record[
            "sha256"
        ].strip()

        if digest:

            hash_records[digest].append(
                record
            )

    print(
        f"Unique sequences   : "
        f"{len(sequences):,}"
    )

    # ========================================================
    # UNION-FIND
    # ========================================================

    uf = UnionFind()

    for sequence in sequences:
        uf.add(sequence)

    duplicate_groups = 0
    sequence_edges = 0

    for digest, items in hash_records.items():

        if len(items) <= 1:
            continue

        involved = sorted(
            set(
                f"{item['split'].strip()}/"
                f"{item['data_folder'].strip()}"
                for item in items
            )
        )

        if len(involved) <= 1:
            continue

        duplicate_groups += 1

        base = involved[0]

        for other in involved[1:]:

            uf.union(
                base,
                other
            )

            sequence_edges += 1

    # ========================================================
    # COMPONENTS
    # ========================================================

    components = defaultdict(list)

    for sequence in sorted(sequences):

        root = uf.find(sequence)

        components[root].append(
            sequence
        )

    component_list = sorted(
        components.values(),
        key=lambda component: (
            -len(component),
            component
        )
    )

    print()
    print(
        f"Connected components : "
        f"{len(component_list):,}"
    )

    # ========================================================
    # BUILD DETAILED COMPONENT DATA
    # ========================================================

    component_rows = []

    for component_id, component in enumerate(
        component_list,
        start=1
    ):

        # -----------------------------------------------
        # Sort sequences deterministically
        # -----------------------------------------------

        component = sorted(component)

        # -----------------------------------------------
        # Sequence details
        # -----------------------------------------------

        sequence_details = []

        total_images = 0

        split_image_counts = Counter()

        for sequence in component:

            split, data_folder = sequence.split(
                "/",
                1
            )

            image_count = sequence_images[
                sequence
            ]

            total_images += image_count

            split_image_counts[
                split
            ] += image_count

            sequence_details.append({
                "sequence": sequence,
                "split": split,
                "data_folder": data_folder,
                "image_count": image_count
            })

        # -----------------------------------------------
        # Verification
        # -----------------------------------------------

        calculated_total = sum(
            item["image_count"]
            for item in sequence_details
        )

        if calculated_total != total_images:

            raise RuntimeError(
                f"Component {component_id} "
                f"image-count mismatch."
            )

        component_rows.append({
            "component_id": component_id,
            "sequence_count": len(component),
            "image_count": total_images,
            "train_images":
                split_image_counts["train"],
            "val_images":
                split_image_counts["val"],
            "test_images":
                split_image_counts["test"],
            "sequences": sequence_details
        })

    # ========================================================
    # PRINT COMPONENTS
    # ========================================================

    print()
    print("=" * 80)
    print("CONNECTED COMPONENT DETAILS")
    print("=" * 80)

    for component in component_rows:

        print()
        print(
            f"Component {component['component_id']}"
        )

        print(
            f"  sequences : "
            f"{component['sequence_count']}"
        )

        print(
            f"  images    : "
            f"{component['image_count']:,}"
        )

        print(
            f"  train     : "
            f"{component['train_images']:,}"
        )

        print(
            f"  val       : "
            f"{component['val_images']:,}"
        )

        print(
            f"  test      : "
            f"{component['test_images']:,}"
        )

        print(
            "  members:"
        )

        for detail in component["sequences"]:

            print(
                f"    - "
                f"{detail['sequence']}"
                f" : "
                f"{detail['image_count']:,} images"
            )

    # ========================================================
    # CROSS-SPLIT COMPONENTS
    # ========================================================

    cross_split = []

    for component in component_rows:

        splits = set(
            detail["split"]
            for detail in component["sequences"]
        )

        if len(splits) > 1:

            cross_split.append(
                component
            )

    print()
    print("=" * 80)
    print("CROSS-SPLIT COMPONENTS")
    print("=" * 80)

    print()
    print(
        f"Components spanning multiple splits: "
        f"{len(cross_split):,}"
    )

    for component in cross_split:

        print()
        print(
            f"Component "
            f"{component['component_id']}"
        )

        for detail in component["sequences"]:

            print(
                f"  {detail['sequence']}"
                f" : "
                f"{detail['image_count']:,}"
            )

    # ========================================================
    # SAVE SEQUENCE-LEVEL CSV
    # ========================================================

    sequence_csv = (
        OUTPUT_DIR /
        "stage_0_9b_sequence_component_members.csv"
    )

    rows = []

    for component in component_rows:

        for detail in component["sequences"]:

            rows.append({
                "component_id":
                    component["component_id"],

                "sequence":
                    detail["sequence"],

                "split":
                    detail["split"],

                "data_folder":
                    detail["data_folder"],

                "image_count":
                    detail["image_count"],

                "component_total_images":
                    component["image_count"]
            })

    fields = [
        "component_id",
        "sequence",
        "split",
        "data_folder",
        "image_count",
        "component_total_images"
    ]

    with open(
        sequence_csv,
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
        f"Saved: {sequence_csv}"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    summary = {
        "inventory_records": len(records),
        "unique_sequences": len(sequences),
        "duplicate_hash_groups":
            duplicate_groups,
        "sequence_connection_edges":
            sequence_edges,
        "connected_components":
            len(component_rows),
        "cross_split_components":
            len(cross_split),
        "components":
            component_rows
    }

    json_path = (
        OUTPUT_DIR /
        "stage_0_9b_corrected_components.json"
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
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 0.9B CORRECTED AUDIT COMPLETE")
    print("=" * 80)

    print()
    print(
        "No images were copied."
    )

    print(
        "No images were deleted."
    )

    print(
        "No split assignments were changed."
    )


if __name__ == "__main__":
    main()