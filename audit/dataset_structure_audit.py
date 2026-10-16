from pathlib import Path
from collections import Counter, defaultdict
import json
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)

PACKAGE = 1

PACKAGE_ROOT = (
    DATASET_ROOT / f"BeltCrack9kd{PACKAGE}"
)

IMAGE_ROOT = PACKAGE_ROOT / "images"
JSON_ROOT = PACKAGE_ROOT / "json"


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def get_image_dimensions(path):
    """
    Read JPEG dimensions without loading the full image.

    Returns:
        (width, height)
    """

    try:
        with open(path, "rb") as f:

            data = f.read(24)

            if len(data) != 24:
                return None

            if data[0:2] != b"\xff\xd8":
                return None

            f.seek(0)

            size = 2

            ftype = 0

            while True:

                byte = f.read(1)

                while byte == b"\xff":
                    byte = f.read(1)

                if not byte:
                    return None

                marker = byte[0]

                # SOF markers
                if (
                    0xC0 <= marker <= 0xC3
                    or
                    0xC5 <= marker <= 0xC7
                    or
                    0xC9 <= marker <= 0xCB
                    or
                    0xCD <= marker <= 0xCF
                ):

                    f.read(3)

                    height = int.from_bytes(
                        f.read(2),
                        "big"
                    )

                    width = int.from_bytes(
                        f.read(2),
                        "big"
                    )

                    return width, height

                else:

                    length_bytes = f.read(2)

                    if len(length_bytes) != 2:
                        return None

                    length = int.from_bytes(
                        length_bytes,
                        "big"
                    )

                    if length < 2:
                        return None

                    f.seek(
                        length - 2,
                        1
                    )

    except Exception:
        return None


def load_json(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "BELTCRACK9KD - STAGE 0.6"
    )

    print(
        "DATASET STRUCTURE & SEQUENCE AUDIT"
    )

    print("=" * 70)

    print()

    print(
        "Dataset:"
    )

    print(
        DATASET_ROOT
    )

    print()

    print(
        "Package:"
    )

    print(
        f"BeltCrack9kd{PACKAGE}"
    )

    print()

    print(
        "READ-ONLY AUDIT"
    )

    print(
        "No files will be modified."
    )


    # ========================================================
    # DIRECTORY CHECK
    # ========================================================

    if not IMAGE_ROOT.exists():

        print()
        print(
            f"ERROR: Image directory not found:"
        )
        print(IMAGE_ROOT)
        return

    if not JSON_ROOT.exists():

        print()
        print(
            f"ERROR: JSON directory not found:"
        )
        print(JSON_ROOT)
        return


    # ========================================================
    # DISCOVER DATA FOLDERS
    # ========================================================

    print()
    print("=" * 70)
    print("DATA FOLDER STRUCTURE")
    print("=" * 70)

    image_data_folders = {}

    json_data_folders = {}

    for split in ["train", "val", "test"]:

        split_image_root = (
            IMAGE_ROOT / split
        )

        split_json_root = (
            JSON_ROOT / split
        )

        image_folders = []

        json_folders = []

        if split_image_root.exists():

            image_folders = sorted(
                [
                    p.name
                    for p in split_image_root.iterdir()
                    if p.is_dir()
                ]
            )

        if split_json_root.exists():

            json_folders = sorted(
                [
                    p.name
                    for p in split_json_root.iterdir()
                    if p.is_dir()
                ]
            )

        image_data_folders[split] = (
            image_folders
        )

        json_data_folders[split] = (
            json_folders
        )

        print()
        print(
            f"{split.upper()}"
        )

        print(
            f"  Image data folders: "
            f"{len(image_folders)}"
        )

        print(
            f"  JSON data folders  : "
            f"{len(json_folders)}"
        )

        print(
            f"  Image folders: "
            f"{image_folders}"
        )

        print(
            f"  JSON folders : "
            f"{json_folders}"
        )


    # ========================================================
    # COUNT IMAGES BY DATA FOLDER
    # ========================================================

    print()
    print("=" * 70)
    print("IMAGE COUNTS BY DATA FOLDER")
    print("=" * 70)

    folder_statistics = {}

    for split in ["train", "val", "test"]:

        split_root = IMAGE_ROOT / split

        if not split_root.exists():
            continue

        for data_folder in sorted(
            [
                p for p in split_root.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.name
        ):

            images = sorted(
                data_folder.glob("*.jpg")
            )

            folder_key = (
                f"{split}/{data_folder.name}"
            )

            folder_statistics[
                folder_key
            ] = {
                "count": len(images),
                "files": images
            }

            print(
                f"{folder_key:25s} "
                f"{len(images):6,d}"
            )


    # ========================================================
    # FILENAME ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("FILENAME / ID STRUCTURE ANALYSIS")
    print("=" * 70)

    numeric_ids = 0
    non_numeric_ids = 0

    min_id = None
    max_id = None

    ids_by_folder = defaultdict(list)

    for folder_key, info in (
        folder_statistics.items()
    ):

        for image_path in info["files"]:

            stem = image_path.stem

            try:

                numeric_id = int(stem)

                numeric_ids += 1

                ids_by_folder[
                    folder_key
                ].append(numeric_id)

                if min_id is None:
                    min_id = numeric_id

                else:
                    min_id = min(
                        min_id,
                        numeric_id
                    )

                if max_id is None:
                    max_id = numeric_id

                else:
                    max_id = max(
                        max_id,
                        numeric_id
                    )

            except ValueError:

                non_numeric_ids += 1

    print()

    print(
        f"Numeric filenames     : "
        f"{numeric_ids:,}"
    )

    print(
        f"Non-numeric filenames : "
        f"{non_numeric_ids:,}"
    )

    print()

    if min_id is not None:

        print(
            f"Global minimum numeric ID: "
            f"{min_id}"
        )

        print(
            f"Global maximum numeric ID: "
            f"{max_id}"
        )


    # ========================================================
    # ID CONTIGUITY
    # ========================================================

    print()
    print("=" * 70)
    print("ID CONTIGUITY BY DATA FOLDER")
    print("=" * 70)

    for folder_key, ids in sorted(
        ids_by_folder.items()
    ):

        if not ids:
            continue

        unique_ids = sorted(
            set(ids)
        )

        min_value = min(unique_ids)
        max_value = max(unique_ids)

        expected = (
            max_value -
            min_value +
            1
        )

        missing = expected - len(
            unique_ids
        )

        print()

        print(
            f"{folder_key}"
        )

        print(
            f"  Count       : "
            f"{len(unique_ids):,}"
        )

        print(
            f"  Min ID      : "
            f"{min_value}"
        )

        print(
            f"  Max ID      : "
            f"{max_value}"
        )

        print(
            f"  Missing IDs : "
            f"{missing:,}"
        )

        if len(unique_ids) <= 30:

            print(
                f"  IDs         : "
                f"{unique_ids}"
            )

        else:

            print(
                f"  First IDs   : "
                f"{unique_ids[:15]}"
            )

            print(
                f"  Last IDs    : "
                f"{unique_ids[-15:]}"
            )


    # ========================================================
    # JSON / IMAGE KEY ALIGNMENT
    # ========================================================

    print()
    print("=" * 70)
    print("IMAGE / JSON STRUCTURE ALIGNMENT")
    print("=" * 70)

    alignment_errors = []

    for split in ["train", "val", "test"]:

        image_split = (
            IMAGE_ROOT / split
        )

        json_split = (
            JSON_ROOT / split
        )

        if not image_split.exists():
            continue

        for data_folder in sorted(
            [
                p for p in image_split.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.name
        ):

            json_folder = (
                json_split /
                data_folder.name
            )

            image_ids = {
                p.stem
                for p in data_folder.glob(
                    "*.jpg"
                )
            }

            json_ids = set()

            if json_folder.exists():

                json_ids = {
                    p.stem
                    for p in json_folder.glob(
                        "*.json"
                    )
                }

            only_images = (
                image_ids - json_ids
            )

            only_json = (
                json_ids - image_ids
            )

            if (
                only_images
                or
                only_json
            ):

                alignment_errors.append(
                    {
                        "folder":
                            f"{split}/{data_folder.name}",
                        "only_images":
                            len(only_images),
                        "only_json":
                            len(only_json)
                    }
                )

    print()

    if not alignment_errors:

        print(
            "All image/JSON IDs align "
            "within their split/data folders."
        )

    else:

        print(
            "Alignment problems found:"
        )

        for error in alignment_errors:

            print(
                f"  {error['folder']}: "
                f"images-only="
                f"{error['only_images']}, "
                f"json-only="
                f"{error['only_json']}"
            )


    # ========================================================
    # IMAGE DIMENSION ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("IMAGE DIMENSION ANALYSIS")
    print("=" * 70)

    dimension_counts = Counter()

    dimension_errors = 0

    dimension_samples = []

    # Analyze up to 100 images per data folder
    # for a fast structural audit.

    for folder_key, info in sorted(
        folder_statistics.items()
    ):

        sample_files = info["files"][:100]

        for image_path in sample_files:

            dimensions = (
                get_image_dimensions(
                    image_path
                )
            )

            if dimensions is None:

                dimension_errors += 1

                continue

            dimension_counts[
                dimensions
            ] += 1

            dimension_samples.append(
                (
                    folder_key,
                    dimensions
                )
            )

    print()

    print(
        "Sampled up to 100 images "
        "per data folder."
    )

    print()

    print(
        "Dimensions found:"
    )

    for dimensions, count in (
        dimension_counts.most_common()
    ):

        print(
            f"  {dimensions[0]} x "
            f"{dimensions[1]} : "
            f"{count:,}"
        )

    print()

    print(
        f"Dimension read errors: "
        f"{dimension_errors}"
    )


    # ========================================================
    # JSON IMAGE METADATA
    # ========================================================

    print()
    print("=" * 70)
    print("JSON IMAGE METADATA ANALYSIS")
    print("=" * 70)

    json_dimension_counts = Counter()

    json_image_paths = Counter()

    json_metadata_samples = []

    json_files = sorted(
        JSON_ROOT.rglob("*.json")
    )

    for json_path in json_files:

        data = load_json(
            json_path
        )

        if data is None:
            continue

        height = data.get(
            "imageHeight"
        )

        width = data.get(
            "imageWidth"
        )

        if (
            width is not None
            and
            height is not None
        ):

            json_dimension_counts[
                (width, height)
            ] += 1

        image_path_value = data.get(
            "imagePath"
        )

        if image_path_value:

            json_image_paths[
                str(image_path_value)
            ] += 1

        if len(json_metadata_samples) < 10:

            json_metadata_samples.append(
                {
                    "json":
                        str(json_path),
                    "imagePath":
                        image_path_value,
                    "width":
                        width,
                    "height":
                        height
                }
            )

    print()

    print(
        "JSON-reported dimensions:"
    )

    for dimensions, count in (
        json_dimension_counts.most_common()
    ):

        print(
            f"  {dimensions[0]} x "
            f"{dimensions[1]} : "
            f"{count:,}"
        )


    # ========================================================
    # IMAGEPATH PATTERN ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("JSON imagePath PATTERN ANALYSIS")
    print("=" * 70)

    print()

    print(
        "Most common imagePath values:"
    )

    for value, count in (
        json_image_paths.most_common(15)
    ):

        print(
            f"  {count:6,d} : "
            f"{value}"
        )


    # ========================================================
    # DATA FOLDER ID OVERLAP
    # ========================================================

    print()
    print("=" * 70)
    print("DATA-FOLDER ID OVERLAP ANALYSIS")
    print("=" * 70)

    # Check whether the same filename ID appears
    # in multiple data folders.

    id_locations = defaultdict(list)

    for folder_key, ids in (
        ids_by_folder.items()
    ):

        for numeric_id in set(ids):

            id_locations[
                numeric_id
            ].append(folder_key)

    repeated_ids = {

        numeric_id: locations

        for numeric_id, locations
        in id_locations.items()

        if len(locations) > 1
    }

    print()

    print(
        f"Unique numeric IDs: "
        f"{len(id_locations):,}"
    )

    print(
        f"IDs appearing in multiple "
        f"locations: "
        f"{len(repeated_ids):,}"
    )

    if repeated_ids:

        print()

        print(
            "Examples:"
        )

        shown = 0

        for numeric_id, locations in (
            sorted(repeated_ids.items())
        ):

            print(
                f"  ID {numeric_id}: "
                f"{locations}"
            )

            shown += 1

            if shown >= 20:
                break


    # ========================================================
    # SEQUENTIAL NEIGHBOR ANALYSIS
    # ========================================================

    print()
    print("=" * 70)
    print("SEQUENTIAL NEIGHBOR ANALYSIS")
    print("=" * 70)

    print()

    print(
        "This does NOT assume that numeric IDs "
        "represent time."
    )

    print(
        "It only checks whether consecutive IDs "
        "exist inside each data folder."
    )

    sequential_results = []

    for folder_key, ids in sorted(
        ids_by_folder.items()
    ):

        unique_ids = sorted(
            set(ids)
        )

        if len(unique_ids) < 2:
            continue

        consecutive_pairs = 0

        for i in range(
            len(unique_ids) - 1
        ):

            if (
                unique_ids[i + 1]
                ==
                unique_ids[i] + 1
            ):

                consecutive_pairs += 1

        possible_pairs = (
            len(unique_ids) - 1
        )

        percentage = (
            consecutive_pairs
            /
            possible_pairs
            *
            100
        )

        sequential_results.append(
            (
                folder_key,
                len(unique_ids),
                consecutive_pairs,
                percentage
            )
        )

    for (
        folder_key,
        count,
        consecutive,
        percentage
    ) in sequential_results:

        print()

        print(
            f"{folder_key}"
        )

        print(
            f"  IDs: "
            f"{count:,}"
        )

        print(
            f"  Consecutive ID pairs: "
            f"{consecutive:,}"
        )

        print(
            f"  Consecutive coverage: "
            f"{percentage:.2f}%"
        )


    # ========================================================
    # FINAL STRUCTURAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 0.6 AUDIT COMPLETE")
    print("=" * 70)

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Numeric filename order alone does NOT prove "
        "temporal sequence."
    )

    print()

    print(
        "We will use the results above to decide "
        "whether genuine sequence-level modeling "
        "is supported by the dataset."
    )

    print()

    print(
        "DO NOT create sequences yet."
    )

    print(
        "DO NOT create the final train/val/test split yet."
    )

    print(
        "DO NOT delete duplicate images."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()