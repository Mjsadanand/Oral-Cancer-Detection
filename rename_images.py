"""
Image Renaming Script for Histopathology Images
Renames cancerous images to a standardized format
"""

import os
import re
from pathlib import Path
import argparse
from typing import List, Tuple
import shutil

def extract_image_number(filename: str) -> int:
    """
    Extract numeric identifier from various filename patterns
    
    Handles patterns like:
    - aug_65_1927 -> 1927
    - OSCC_400x_54 -> 54
    - cancer_001 -> 1
    """
    # Try to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    
    if not numbers:
        return 0
    
    # If filename starts with 'aug', use the last number
    if filename.lower().startswith('aug'):
        return int(numbers[-1]) if numbers else 0
    
    # If filename contains 'OSCC' or similar, use the last number
    if 'oscc' in filename.lower() or '400x' in filename.lower():
        return int(numbers[-1]) if numbers else 0
    
    # Default: use the first number found
    return int(numbers[0])

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()

def rename_images_in_directory(
    directory: Path,
    prefix: str = "cancer",
    start_index: int = 1,
    dry_run: bool = True,
    backup: bool = True
) -> List[Tuple[str, str]]:
    """
    Rename all images in a directory to a standardized format
    
    Args:
        directory: Path to the directory containing images
        prefix: Prefix for renamed files (default: "cancer")
        start_index: Starting index for numbering (default: 1)
        dry_run: If True, only show what would be renamed without actually renaming
        backup: If True and not dry_run, create a backup of original files
    
    Returns:
        List of (old_name, new_name) tuples
    """
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {directory}")
        return []
    
    print(f"\n📁 Found {len(image_files)} images in {directory}")
    
    # Sort files by extracted number for consistent renaming
    try:
        sorted_files = sorted(image_files, key=lambda x: extract_image_number(x.stem))
    except Exception as e:
        print(f"⚠️ Warning: Could not sort by number, using alphabetical sort. Error: {e}")
        sorted_files = sorted(image_files)
    
    # Create backup directory if needed
    if backup and not dry_run:
        backup_dir = directory / "backup_originals"
        backup_dir.mkdir(exist_ok=True)
        print(f"📦 Backup directory created: {backup_dir}")
    
    # Generate new names
    renamed_pairs = []
    current_index = start_index
    
    for old_file in sorted_files:
        # Generate new filename
        extension = get_file_extension(old_file.name)
        new_name = f"{prefix}_{current_index:04d}{extension}"
        new_path = directory / new_name
        
        # Handle name conflicts
        while new_path.exists() and new_path != old_file:
            current_index += 1
            new_name = f"{prefix}_{current_index:04d}{extension}"
            new_path = directory / new_name
        
        renamed_pairs.append((old_file.name, new_name))
        current_index += 1
    
    # Display renaming plan
    print("\n🔄 Renaming Plan:")
    print("=" * 80)
    for old_name, new_name in renamed_pairs[:10]:  # Show first 10
        print(f"  {old_name:40} -> {new_name}")
    
    if len(renamed_pairs) > 10:
        print(f"  ... and {len(renamed_pairs) - 10} more files")
    print("=" * 80)
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files were actually renamed")
        print("   Run with --execute to perform the actual renaming")
        return renamed_pairs
    
    # Perform actual renaming
    print("\n🚀 Executing renaming...")
    success_count = 0
    
    for old_name, new_name in renamed_pairs:
        old_path = directory / old_name
        new_path = directory / new_name
        
        try:
            # Create backup if requested
            if backup:
                backup_path = backup_dir / old_name
                shutil.copy2(old_path, backup_path)
            
            # Rename file
            old_path.rename(new_path)
            success_count += 1
            print(f"  ✅ {old_name} -> {new_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to rename {old_name}: {e}")
    
    print(f"\n✅ Successfully renamed {success_count}/{len(renamed_pairs)} files")
    
    if backup:
        print(f"📦 Original files backed up to: {backup_dir}")
    
    return renamed_pairs

def rename_dataset_structure(
    dataset_path: Path,
    train_prefix: str = "cancer_train",
    val_prefix: str = "cancer_val",
    dry_run: bool = True,
    backup: bool = True
):
    """
    Rename images in the full dataset structure
    
    Expected structure:
    dataset/
    ├── train/
    │   └── cancerous/
    └── validation/
        └── cancerous/
    """
    
    print("\n" + "=" * 80)
    print("🏥 Histopathology Image Renaming Tool")
    print("=" * 80)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Rename training images
    train_cancer_path = dataset_path / "train" / "cancerous"
    if train_cancer_path.exists():
        print(f"\n📂 Processing: {train_cancer_path}")
        rename_images_in_directory(
            train_cancer_path,
            prefix=train_prefix,
            start_index=1,
            dry_run=dry_run,
            backup=backup
        )
    else:
        print(f"⚠️ Training cancerous folder not found: {train_cancer_path}")
    
    # Rename validation images
    val_cancer_path = dataset_path / "validation" / "cancerous"
    if val_cancer_path.exists():
        print(f"\n📂 Processing: {val_cancer_path}")
        rename_images_in_directory(
            val_cancer_path,
            prefix=val_prefix,
            start_index=1,
            dry_run=dry_run,
            backup=backup
        )
    else:
        print(f"⚠️ Validation cancerous folder not found: {val_cancer_path}")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("✅ Dry run complete! Review the renaming plan above.")
        print("   To execute the renaming, run with --execute flag")
    else:
        print("✅ Renaming complete!")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Rename histopathology images to standardized format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes):
  python rename_images.py
  
  # Execute renaming with default settings:
  python rename_images.py --execute
  
  # Execute without backup:
  python rename_images.py --execute --no-backup
  
  # Rename specific directory:
  python rename_images.py --dir dataset/train/cancerous --prefix oscc --execute
  
  # Custom dataset path:
  python rename_images.py --dataset-path /path/to/dataset --execute
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='dataset',
        help='Path to dataset folder (default: dataset)'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='Rename images in a specific directory (overrides dataset structure)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='cancer',
        help='Prefix for renamed files (default: cancer)'
    )
    
    parser.add_argument(
        '--start-index',
        type=int,
        default=1,
        help='Starting index for numbering (default: 1)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the renaming (default is dry run)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original files'
    )
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    backup = not args.no_backup
    
    if args.dir:
        # Rename specific directory
        directory = Path(args.dir)
        print("\n" + "=" * 80)
        print("🏥 Histopathology Image Renaming Tool")
        print("=" * 80)
        rename_images_in_directory(
            directory,
            prefix=args.prefix,
            start_index=args.start_index,
            dry_run=dry_run,
            backup=backup
        )
    else:
        # Rename full dataset structure
        dataset_path = Path(args.dataset_path)
        rename_dataset_structure(
            dataset_path,
            train_prefix="cancer_train",
            val_prefix="cancer_val",
            dry_run=dry_run,
            backup=backup
        )

if __name__ == "__main__":
    main()
