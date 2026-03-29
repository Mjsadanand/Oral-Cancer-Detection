"""
Rename cancerous test images to a standardized 'cancer' label format.

Example:
- OSCC_100x_2.jpg -> cancer_0001.jpg
"""

from pathlib import Path
import argparse
from typing import List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_last_number(stem: str) -> int:
    """Extract the trailing numeric token from names like OSCC_100x_2."""
    parts = stem.split("_")
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    return 0


def collect_target_files(folder: Path) -> List[Path]:
    """Collect image files, prioritizing OSCC-prefixed filenames."""
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    oscc_files = [p for p in files if p.stem.upper().startswith("OSCC")]
    return oscc_files if oscc_files else files


def plan_renames(folder: Path, label: str, start_index: int) -> List[Tuple[Path, Path]]:
    """Create a conflict-safe rename plan."""
    files = collect_target_files(folder)
    files_sorted = sorted(files, key=lambda p: extract_last_number(p.stem))

    plan: List[Tuple[Path, Path]] = []
    index = start_index

    for src in files_sorted:
        ext = src.suffix.lower()
        dst = folder / f"{label}_{index:04d}{ext}"

        while dst.exists() and dst != src:
            index += 1
            dst = folder / f"{label}_{index:04d}{ext}"

        plan.append((src, dst))
        index += 1

    return plan


def apply_renames(plan: List[Tuple[Path, Path]]) -> int:
    """Apply renames using a temporary two-phase strategy to avoid collisions."""
    temp_map: List[Tuple[Path, Path]] = []

    for i, (src, _) in enumerate(plan, start=1):
        temp = src.with_name(f".__tmp_rename_{i:04d}{src.suffix.lower()}")
        src.rename(temp)
        temp_map.append((temp, src))

    renamed = 0
    for (temp, _), (_, dst) in zip(temp_map, plan):
        temp.rename(dst)
        renamed += 1

    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename cancerous images in a test folder to label 'cancer'."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="dataset/test/cancerous",
        help="Path to test image folder (default: dataset/test/cancerous)",
    )
    parser.add_argument(
        "--label",
        default="cancer",
        help="Label prefix for renamed files (default: cancer)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index (default: 1)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform rename (without this flag, only preview is shown)",
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    if not folder.exists() or not folder.is_dir():
        print(f"Directory not found: {folder}")
        return

    plan = plan_renames(folder=folder, label=args.label, start_index=args.start)

    if not plan:
        print(f"No image files found in: {folder}")
        return

    print(f"Found {len(plan)} image(s) in: {folder}")
    print("Rename preview:")
    for src, dst in plan[:15]:
        print(f"  {src.name} -> {dst.name}")

    if len(plan) > 15:
        print(f"  ... and {len(plan) - 15} more")

    if not args.execute:
        print("\nDry run only. Use --execute to apply changes.")
        return

    count = apply_renames(plan)
    print(f"\nDone. Renamed {count} file(s) with label '{args.label}'.")


if __name__ == "__main__":
    main()
