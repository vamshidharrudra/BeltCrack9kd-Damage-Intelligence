from pathlib import Path
import json


# ============================================================
# BELTCRACK9KD — STAGE 0.1
# RAW DATASET AUDIT
#
# Purpose:
#   Understand the actual downloaded dataset structure.
#
# This script DOES NOT:
#   - modify files
#   - copy files
#   - rename files
#   - create datasets
#   - train models
#   - preprocess images
# ============================================================


# ------------------------------------------------------------
# CHANGE ONLY THIS PATH
# ------------------------------------------------------------

DATASET_ROOT = Path(
    r"C:\Users\vamsh\Downloads\BeltCrack9kd"
)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


def count_images(folder):
    """Count image files recursively."""
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_json_files(folder):
    """Count JSON files recursively."""
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*.json")
        if file.is_file()
    )


def get_subdirectories(folder):
    """Return immediate subdirectories."""
    if not folder.exists():
        return []

    return sorted(
        [p for p in folder.iterdir() if p.is_dir()],
        key=lambda x: x.name.lower()
    )


def inspect_json(json_path):
    """
    Inspect one JSON file.

    We deliberately do NOT assume that the dataset uses
    COCO, VOC, YOLO, or any other annotation format.
    """

    print("\n" + "=" * 80)
    print("SAMPLE JSON INSPECTION")
    print("=" * 80)

    print("File:")
    print(json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print("\nCould not read JSON.")
        print("Error:", e)
        return

    print("\nPython object type:")
    print(type(data).__name__)

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        print("\nTop-level keys:")

        for key, value in data.items():

            if isinstance(value, list):
                print(
                    f"  {key} -> list "
                    f"({len(value)} items)"
                )

            elif isinstance(value, dict):
                print(
                    f"  {key} -> dictionary "
                    f"({len(value)} keys)"
                )

            else:
                print(
                    f"  {key} -> "
                    f"{type(value).__name__}: {value}"
                )

        # Show first item for list-valued fields
        print("\nFirst item from list-valued fields:")

        for key, value in data.items():

            if isinstance(value, list) and len(value) > 0:

                print("\n" + "-" * 60)
                print(f"FIELD: {key}")
                print("-" * 60)

                print(value[0])

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    elif isinstance(data, list):

        print("\nNumber of items:", len(data))

        if len(data) > 0:

            print("\nFirst item:")
            print(data[0])

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    else:

        print("\nRaw value:")
        print(data)


# ------------------------------------------------------------
# Main audit
# ------------------------------------------------------------

def main():

    print("\n")
    print("=" * 80)
    print("BELTCRACK9KD — RAW DATASET AUDIT")
    print("=" * 80)

    # --------------------------------------------------------
    # Check dataset path
    # --------------------------------------------------------

    print("\nDataset path:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():

        print("\nERROR: Dataset path does not exist.")

        print("\nPlease change DATASET_ROOT at the top of")
        print("dataset_audit.py to the actual location.")

        return

    print("\nDataset path exists: YES")

    # --------------------------------------------------------
    # Find BeltCrack9kd package folders
    # --------------------------------------------------------

    packages = sorted(
        [
            folder
            for folder in DATASET_ROOT.iterdir()
            if folder.is_dir()
            and folder.name.lower().startswith("beltcrack9kd")
        ],
        key=lambda x: x.name.lower()
    )

    print("\n" + "=" * 80)
    print("DATASET PACKAGES")
    print("=" * 80)

    print("\nNumber of package folders found:", len(packages))

    for package in packages:
        print("  -", package.name)

    # --------------------------------------------------------
    # Inspect every package
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PACKAGE SUMMARY")
    print("=" * 80)

    for package in packages:

        print("\n" + "-" * 80)
        print(package.name)
        print("-" * 80)

        # ----------------------------------------------------
        # Immediate folders
        # ----------------------------------------------------

        subdirs = get_subdirectories(package)

        print("\nImmediate folders:")

        if subdirs:
            for folder in subdirs:
                print("  -", folder.name)
        else:
            print("  None")

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        images_dir = package / "images"

        print("\nImages directory:")
        print("  Path:", images_dir)
        print("  Exists:", images_dir.exists())

        if images_dir.exists():

            image_count = count_images(images_dir)

            print("  Image files:", image_count)

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        json_dir = package / "json"

        print("\nJSON directory:")
        print("  Path:", json_dir)
        print("  Exists:", json_dir.exists())

        if json_dir.exists():

            json_count = count_json_files(json_dir)

            print("  JSON files:", json_count)

            # -----------------------------------------------
            # Inspect train / val / test
            # -----------------------------------------------

            for split in ["train", "val", "test"]:

                split_dir = json_dir / split

                print(f"\n  {split.upper()}:")

                print("    Exists:", split_dir.exists())

                if split_dir.exists():

                    split_json_count = count_json_files(
                        split_dir
                    )

                    print(
                        "    JSON files:",
                        split_json_count
                    )

                    subfolders = get_subdirectories(
                        split_dir
                    )

                    print("    Immediate folders:")

                    if subfolders:

                        for folder in subfolders:
                            print(
                                "      -",
                                folder.name
                            )

                    else:
                        print("      None")

    # --------------------------------------------------------
    # Inspect first JSON file found
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SEARCHING FOR A SAMPLE JSON FILE")
    print("=" * 80)

    json_files = list(
        DATASET_ROOT.rglob("*.json")
    )

    print("\nTotal JSON files discovered recursively:",
          len(json_files))

    if json_files:

        json_files = sorted(
            json_files,
            key=lambda x: str(x).lower()
        )

        inspect_json(json_files[0])

    else:

        print("\nNo JSON files were found.")

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STAGE 0.1 COMPLETE")
    print("=" * 80)

    print(
        "\nNo dataset files were modified by this script."
    )


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    main()