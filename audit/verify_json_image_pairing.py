from pathlib import Path
import json
from collections import defaultdict


# ============================================================
# STAGE 0.2.3
# VERIFY JSON FILENAME <-> IMAGE FILENAME PAIRING
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


# ============================================================
# COUNTERS
# ============================================================

total_json = 0

paired = 0
missing = 0

dimension_match = 0
dimension_mismatch = 0

package_stats = defaultdict(
    lambda: {
        "json": 0,
        "paired": 0,
        "missing": 0,
        "dimension_match": 0,
        "dimension_mismatch": 0,
    }
)

missing_examples = []
dimension_mismatch_examples = []
successful_examples = []


# ============================================================
# GET PACKAGE NAME
# ============================================================

def get_package_name(path):

    for part in path.parts:

        if (
            part.startswith("BeltCrack9kd")
            and part != "BeltCrack9kd"
        ):

            return part

    return "UNKNOWN"


# ============================================================
# GET IMAGE SIZE
# ============================================================

def get_image_size(image_path):

    try:

        from PIL import Image

        with Image.open(image_path) as img:

            return img.width, img.height

    except Exception:

        return None


# ============================================================
# FIND IMAGE FOR JSON
# ============================================================

def find_corresponding_image(
    json_path
):

    package_name = get_package_name(
        json_path
    )

    # --------------------------------------------------------
    # Relative location inside package
    #
    # json/.../<split>/<data_folder>/<id>.json
    #
    # becomes
    #
    # images/.../<split>/<data_folder>/<id>.jpg
    # --------------------------------------------------------

    try:

        relative_to_package = (
            json_path
            .relative_to(
                DATASET_ROOT
                / package_name
                / "json"
            )
        )

    except ValueError:

        return None

    # Remove .json
    image_stem = relative_to_package.stem

    # Keep split/data folder
    image_parent = (
        relative_to_package.parent
    )

    image_directory = (
        DATASET_ROOT
        / package_name
        / "images"
        / image_parent
    )

    # --------------------------------------------------------
    # Try extensions
    # --------------------------------------------------------

    for extension in IMAGE_EXTENSIONS:

        candidate = (
            image_directory
            / (image_stem + extension)
        )

        if (
            candidate.exists()
            and candidate.is_file()
        ):

            return candidate

    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "STAGE 0.2.3 - VERIFY JSON <-> IMAGE PAIRING"
    )
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():

        print()
        print(
            "[ERROR] Dataset root does not exist."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # FIND JSON FILES
    # --------------------------------------------------------

    print()
    print(
        "Searching for JSON files..."
    )

    json_files = list(
        DATASET_ROOT.rglob("*.json")
    )

    print(
        f"JSON files found: "
        f"{len(json_files):,}"
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        total_json += 1

        package_name = get_package_name(
            json_path
        )

        stats = package_stats[
            package_name
        ]

        stats["json"] += 1

        # ----------------------------------------------------
        # FIND IMAGE
        # ----------------------------------------------------

        image_path = find_corresponding_image(
            json_path
        )

        if image_path is None:

            missing += 1

            stats["missing"] += 1

            if len(missing_examples) < 30:

                missing_examples.append(
                    json_path
                )

        else:

            paired += 1

            stats["paired"] += 1

            if len(successful_examples) < 30:

                successful_examples.append(
                    (
                        json_path,
                        image_path
                    )
                )

            # ------------------------------------------------
            # CHECK DIMENSIONS
            # ------------------------------------------------

            try:

                with open(
                    json_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                json_width = data.get(
                    "imageWidth"
                )

                json_height = data.get(
                    "imageHeight"
                )

                actual_size = get_image_size(
                    image_path
                )

                if (
                    actual_size is not None
                    and
                    json_width is not None
                    and
                    json_height is not None
                ):

                    actual_width, actual_height = (
                        actual_size
                    )

                    if (
                        actual_width
                        == json_width
                        and
                        actual_height
                        == json_height
                    ):

                        dimension_match += 1

                        stats[
                            "dimension_match"
                        ] += 1

                    else:

                        dimension_mismatch += 1

                        stats[
                            "dimension_mismatch"
                        ] += 1

                        if (
                            len(
                                dimension_mismatch_examples
                            )
                            < 30
                        ):

                            dimension_mismatch_examples.append(
                                (
                                    json_path,
                                    image_path,
                                    (
                                        json_width,
                                        json_height
                                    ),
                                    (
                                        actual_width,
                                        actual_height
                                    ),
                                )
                            )

            except Exception:

                pass

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % 5000 == 0:

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(json_files):,}"
            )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Total JSON files:       "
        f"{total_json:,}"
    )

    print(
        f"JSON -> image paired:   "
        f"{paired:,}"
    )

    print(
        f"JSON -> image missing:  "
        f"{missing:,}"
    )

    if total_json > 0:

        pairing_rate = (
            paired
            / total_json
            * 100
        )

        print(
            f"Pairing success rate:   "
            f"{pairing_rate:.2f}%"
        )

    print()
    print(
        f"Dimension matches:      "
        f"{dimension_match:,}"
    )

    print(
        f"Dimension mismatches:   "
        f"{dimension_mismatch:,}"
    )

    if paired > 0:

        dimension_rate = (
            dimension_match
            / paired
            * 100
        )

        print(
            f"Dimension match rate:   "
            f"{dimension_rate:.2f}%"
        )

    # ========================================================
    # PACKAGE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PACKAGE SUMMARY")
    print("=" * 70)

    for package_name in sorted(
        package_stats.keys()
    ):

        stats = package_stats[
            package_name
        ]

        total = stats["json"]

        paired_count = stats[
            "paired"
        ]

        missing_count = stats[
            "missing"
        ]

        if total > 0:

            rate = (
                paired_count
                / total
                * 100
            )

        else:

            rate = 0

        print()
        print(package_name)

        print(
            f"  JSON:             "
            f"{total:,}"
        )

        print(
            f"  Paired:           "
            f"{paired_count:,}"
        )

        print(
            f"  Missing:          "
            f"{missing_count:,}"
        )

        print(
            f"  Pairing rate:     "
            f"{rate:.2f}%"
        )

        print(
            f"  Dimension match:  "
            f"{stats['dimension_match']:,}"
        )

        print(
            f"  Dimension wrong:  "
            f"{stats['dimension_mismatch']:,}"
        )

    # ========================================================
    # SUCCESSFUL EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("SUCCESSFUL PAIRING EXAMPLES")
    print("=" * 70)

    for (
        json_path,
        image_path
    ) in successful_examples:

        print()
        print("JSON:")
        print(json_path)

        print("IMAGE:")
        print(image_path)

    # ========================================================
    # MISSING EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("MISSING PAIRING EXAMPLES")
    print("=" * 70)

    if missing_examples:

        for json_path in missing_examples:

            print()
            print(json_path)

    else:

        print(
            "No missing pairings."
        )

    # ========================================================
    # DIMENSION MISMATCH EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("DIMENSION MISMATCH EXAMPLES")
    print("=" * 70)

    if dimension_mismatch_examples:

        for (
            json_path,
            image_path,
            json_size,
            actual_size
        ) in dimension_mismatch_examples:

            print()
            print(
                "JSON:"
            )

            print(
                json_path
            )

            print(
                "IMAGE:"
            )

            print(
                image_path
            )

            print(
                "JSON dimensions:"
            )

            print(
                json_size
            )

            print(
                "Actual image dimensions:"
            )

            print(
                actual_size
            )

    else:

        print(
            "No dimension mismatches."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE 0.2.3 COMPLETE")
    print("=" * 70)

    print()
    print(
        "No dataset files were modified."
    )