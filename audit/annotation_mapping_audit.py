from pathlib import Path
import json
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"C:\Users\vamsh\Downloads\BeltCrack9kd")

PROJECT_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd_Damage_Intelligence"
)

OUTPUT_DIR = PROJECT_ROOT / "audit_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PACKAGES = [
    "BeltCrack9kd1",
    "BeltCrack9kd2",
    "BeltCrack9kd3",
    "BeltCrack9kd4",
]

SPLITS = ["train", "val", "test"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STAGE 0.2 - ANNOTATION / IMAGE MAPPING AUDIT")
    print("=" * 80)

    print(f"Dataset root : {DATASET_ROOT}")
    print()

    total_json = 0
    total_paired = 0
    total_missing = 0
    total_dimension_match = 0
    total_dimension_mismatch = 0
    total_json_errors = 0

    missing_examples = []
    mismatch_examples = []
    path_examples = []

    # ========================================================
    # PACKAGE LOOP
    # ========================================================

    for package in PACKAGES:

        package_root = DATASET_ROOT / package

        print("-" * 80)
        print(package)
        print("-" * 80)

        package_json = 0
        package_paired = 0
        package_missing = 0
        package_dim_match = 0
        package_dim_mismatch = 0
        package_errors = 0

        json_root = package_root / "json"
        image_root = package_root / "images"

        if not json_root.exists():

            print("JSON folder not found.")
            continue

        # ====================================================
        # SPLITS
        # ====================================================

        for split in SPLITS:

            json_split_dir = json_root / split
            image_split_dir = image_root / split

            if not json_split_dir.exists():
                continue

            # ------------------------------------------------
            # DATA FOLDERS
            # ------------------------------------------------

            for data_dir in sorted(
                json_split_dir.glob("data*")
            ):

                if not data_dir.is_dir():
                    continue

                data_name = data_dir.name

                image_data_dir = (
                    image_split_dir / data_name
                )

                # ------------------------------------------------
                # JSON FILES
                # ------------------------------------------------

                for json_file in sorted(
                    data_dir.glob("*.json")
                ):

                    package_json += 1
                    total_json += 1

                    stem = json_file.stem

                    # IMPORTANT:
                    #
                    # Do NOT use JSON imagePath.
                    #
                    # Correct identity is:
                    #
                    # json/<split>/<dataN>/<ID>.json
                    #
                    # ->
                    #
                    # images/<split>/<dataN>/<ID>.jpg

                    image_file = (
                        image_data_dir /
                        f"{stem}.jpg"
                    )

                    # ------------------------------------------------
                    # CHECK IMAGE EXISTENCE
                    # ------------------------------------------------

                    if not image_file.exists():

                        package_missing += 1
                        total_missing += 1

                        if len(missing_examples) < 20:

                            missing_examples.append(
                                str(json_file)
                            )

                        continue

                    package_paired += 1
                    total_paired += 1

                    # ------------------------------------------------
                    # SAMPLE PATH INFORMATION
                    # ------------------------------------------------

                    if len(path_examples) < 10:

                        try:

                            with open(
                                json_file,
                                "r",
                                encoding="utf-8"
                            ) as f:

                                json_data = json.load(f)

                            path_examples.append({
                                "json": str(json_file),
                                "json_imagePath":
                                    json_data.get(
                                        "imagePath"
                                    ),
                                "actual_image":
                                    str(image_file)
                            })

                        except Exception:
                            pass

                    # ------------------------------------------------
                    # READ JSON DIMENSIONS
                    # ------------------------------------------------

                    try:

                        with open(
                            json_file,
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

                        # ------------------------------------------------
                        # READ ACTUAL IMAGE DIMENSIONS
                        # ------------------------------------------------

                        try:

                            from PIL import Image

                            with Image.open(
                                image_file
                            ) as img:

                                actual_width, actual_height = (
                                    img.size
                                )

                        except Exception as e:

                            package_errors += 1
                            total_json_errors += 1
                            continue

                        # ------------------------------------------------
                        # DIMENSION COMPARISON
                        # ------------------------------------------------

                        if (
                            json_width == actual_width
                            and
                            json_height == actual_height
                        ):

                            package_dim_match += 1
                            total_dimension_match += 1

                        else:

                            package_dim_mismatch += 1
                            total_dimension_mismatch += 1

                            if len(
                                mismatch_examples
                            ) < 20:

                                mismatch_examples.append({
                                    "json":
                                        str(json_file),

                                    "image":
                                        str(image_file),

                                    "json_dimensions":
                                        (
                                            json_width,
                                            json_height
                                        ),

                                    "actual_dimensions":
                                        (
                                            actual_width,
                                            actual_height
                                        )
                                })

                    except Exception as e:

                        package_errors += 1
                        total_json_errors += 1

        # ====================================================
        # PACKAGE SUMMARY
        # ====================================================

        print(
            f"JSON files           : {package_json:,}"
        )

        print(
            f"Paired images        : {package_paired:,}"
        )

        print(
            f"Missing images       : {package_missing:,}"
        )

        print(
            f"Dimension matches    : {package_dim_match:,}"
        )

        print(
            f"Dimension mismatches : {package_dim_mismatch:,}"
        )

        print(
            f"Read errors          : {package_errors:,}"
        )

        if package_json > 0:

            print(
                f"Pairing rate         : "
                f"{package_paired / package_json * 100:.2f}%"
            )


    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    print(
        f"Total JSON files       : {total_json:,}"
    )

    print(
        f"Successfully paired    : {total_paired:,}"
    )

    print(
        f"Missing images         : {total_missing:,}"
    )

    print(
        f"Dimension matches      : {total_dimension_match:,}"
    )

    print(
        f"Dimension mismatches   : {total_dimension_mismatch:,}"
    )

    print(
        f"Read errors            : {total_json_errors:,}"
    )

    if total_json > 0:

        print(
            f"Overall pairing rate   : "
            f"{total_paired / total_json * 100:.2f}%"
        )

    # ========================================================
    # EXAMPLE MAPPING
    # ========================================================

    print()
    print("=" * 80)
    print("EXAMPLE JSON -> IMAGE MAPPING")
    print("=" * 80)

    for example in path_examples:

        print()
        print(
            f"JSON file:"
        )
        print(
            f"  {example['json']}"
        )

        print(
            f"JSON imagePath:"
        )
        print(
            f"  {example['json_imagePath']}"
        )

        print(
            f"Actual image used:"
        )
        print(
            f"  {example['actual_image']}"
        )

    # ========================================================
    # MISSING EXAMPLES
    # ========================================================

    if missing_examples:

        print()
        print("=" * 80)
        print("MISSING IMAGE EXAMPLES")
        print("=" * 80)

        for path in missing_examples:

            print(path)

    # ========================================================
    # DIMENSION MISMATCH EXAMPLES
    # ========================================================

    if mismatch_examples:

        print()
        print("=" * 80)
        print("DIMENSION MISMATCH EXAMPLES")
        print("=" * 80)

        for item in mismatch_examples:

            print()
            print(
                f"JSON   : {item['json']}"
            )

            print(
                f"IMAGE  : {item['image']}"
            )

            print(
                f"JSON dimensions    : "
                f"{item['json_dimensions']}"
            )

            print(
                f"Actual dimensions  : "
                f"{item['actual_dimensions']}"
            )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE 0.2 RESULT")
    print("=" * 80)

    if (
        total_json > 0
        and total_missing == 0
        and total_dimension_mismatch == 0
    ):

        print()
        print("PASS")

        print(
            "All annotation JSON files have a corresponding "
            "image using the dataset's structural mapping."
        )

        print(
            "All JSON-reported image dimensions match the "
            "actual image dimensions."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The JSON imagePath field should NOT be used "
            "as the primary image identity."
        )

        print(
            "Use:"
        )

        print(
            "json/<split>/<dataN>/<ID>.json"
        )

        print(
            "->"
        )

        print(
            "images/<split>/<dataN>/<ID>.jpg"
        )

    else:

        print()
        print("REVIEW REQUIRED")

        if total_missing > 0:

            print(
                f"Missing images: {total_missing:,}"
            )

        if total_dimension_mismatch > 0:

            print(
                f"Dimension mismatches: "
                f"{total_dimension_mismatch:,}"
            )

        if total_json_errors > 0:

            print(
                f"Read errors: "
                f"{total_json_errors:,}"
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_DIR /
        "stage_0_2_annotation_mapping_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STAGE 0.2 - ANNOTATION / IMAGE MAPPING AUDIT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Total JSON files: {total_json:,}\n"
        )

        f.write(
            f"Successfully paired: {total_paired:,}\n"
        )

        f.write(
            f"Missing images: {total_missing:,}\n"
        )

        f.write(
            f"Dimension matches: "
            f"{total_dimension_match:,}\n"
        )

        f.write(
            f"Dimension mismatches: "
            f"{total_dimension_mismatch:,}\n"
        )

        f.write(
            f"Read errors: "
            f"{total_json_errors:,}\n"
        )

        if total_json > 0:

            f.write(
                f"Pairing rate: "
                f"{total_paired / total_json * 100:.2f}%\n"
            )

        f.write("\n")

        f.write(
            "Mapping rule:\n"
        )

        f.write(
            "json/<split>/<dataN>/<ID>.json "
            "-> "
            "images/<split>/<dataN>/<ID>.jpg\n"
        )

    print()
    print(
        f"Summary saved to:\n{summary_path}"
    )

    print()
    print("=" * 80)
    print("STAGE 0.2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()