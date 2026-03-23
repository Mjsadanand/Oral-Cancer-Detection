"""
Image Renaming Script for Histopathology Normal Images
Renames normal (non-cancerous) images to a standardized format
"""

import os
import re
from pathlib import Path
import argparse
from typing import List, Tuple
import shutil
import sys

def extract_image_number(filename: str) -> int:
    """
    Extract numeric identifier from various filename patterns
    
    Handles patterns like:
    - aug_65_1927 -> 1927
    - OSCC_400x_54 -> 54
    - normal_001 -> 1
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
    prefix: str = "normal",
    start_index: int = 1,
    dry_run: bool = True,
    backup: bool = True
) -> List[Tuple[str, str]]:
    """
    Rename all images in a directory to a standardized format
    
    Args:
        directory: Path to the directory containing images
        prefix: Prefix for renamed files (default: "normal")
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
    train_prefix: str = "normal_train",
    val_prefix: str = "normal_val",
    dry_run: bool = True,
    backup: bool = True
):
    """
    Rename images in the full dataset structure
    
    Expected structure:
    dataset/
    ├── train/
    │   └── normal/
    └── validation/
        └── normal/
    """
    
    print("\n" + "=" * 80)
    print("🏥 Histopathology Normal Image Renaming Tool")
    print("=" * 80)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Rename training images
    train_normal_path = dataset_path / "train" / "normal"
    if train_normal_path.exists():
        print(f"\n📂 Processing: {train_normal_path}")
        rename_images_in_directory(
            train_normal_path,
            prefix=train_prefix,
            start_index=1,
            dry_run=dry_run,
            backup=backup
        )
    else:
        print(f"⚠️ Training normal folder not found: {train_normal_path}")
    
    # Rename validation images
    val_normal_path = dataset_path / "validation" / "normal"
    if val_normal_path.exists():
        print(f"\n📂 Processing: {val_normal_path}")
        rename_images_in_directory(
            val_normal_path,
            prefix=val_prefix,
            start_index=1,
            dry_run=dry_run,
            backup=backup
        )
    else:
        print(f"⚠️ Validation normal folder not found: {val_normal_path}")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("✅ Dry run complete! Review the renaming plan above.")
        print("   To execute the renaming, run with --execute flag")
    else:
        print("✅ Renaming complete!")
    print("=" * 80)

def interactive_directory_selector(dataset_path: Path):
    """
    Interactively select which directories to rename
    
    Args:
        dataset_path: Path to the dataset folder
    
    Returns:
        List of tuples: (directory_path, prefix)
    """
    print("\n" + "=" * 80)
    print("🏥 Histopathology Normal Image Renaming Tool - Directory Selector")
    print("=" * 80)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return []
    
    # Find all available directories
    available_dirs = []
    
    # Check train/normal
    train_normal = dataset_path / "train" / "normal"
    if train_normal.exists():
        available_dirs.append((train_normal, "normal_train", "Train - Normal"))
    
    # Check validation/normal
    val_normal = dataset_path / "validation" / "normal"
    if val_normal.exists():
        available_dirs.append((val_normal, "normal_val", "Validation - Normal"))
    
    if not available_dirs:
        print(f"❌ No standard directories found in {dataset_path}")
        return []
    
    # Display available directories
    print("\n📁 Available directories:\n")
    for idx, (dir_path, prefix, label) in enumerate(available_dirs, 1):
        image_count = len([f for f in dir_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}])
        print(f"  {idx}. {label}")
        print(f"     Path: {dir_path}")
        print(f"     Images: {image_count}")
        print()
    
    # Get user selection
    print("📋 Select directories to rename (comma-separated, e.g., 1,2 or 'a' for all):")
    user_input = input("  > ").strip().lower()
    
    selected_dirs = []
    
    if user_input == 'a':
        selected_dirs = available_dirs
    elif user_input == '':
        print("❌ No selection made")
        return []
    else:
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            for idx in indices:
                if 0 <= idx < len(available_dirs):
                    selected_dirs.append(available_dirs[idx])
                else:
                    print(f"⚠️ Invalid index: {idx + 1}")
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
            return []
    
    return selected_dirs

def main():
    parser = argparse.ArgumentParser(
        description='Rename normal histopathology images to standardized format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default - select directories):
  python rename_normal_images.py
  
  # Execute with interactive selection:
  python rename_normal_images.py --execute
  
  # Rename specific directory:
  python rename_normal_images.py --dir dataset/train/normal --prefix specimen --execute
  
  # Custom dataset path with interactive selection:
  python rename_normal_images.py --dataset-path /path/to/dataset --execute
  
  # Skip interactive and use batch mode:
  python rename_normal_images.py --no-interactive --execute
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
        help='Rename images in a specific directory (overrides interactive mode)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='normal',
        help='Prefix for renamed files (default: normal)'
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
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Skip interactive selection (use batch mode)'
    )
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    backup = not args.no_backup
    
    if args.dir:
        # Rename specific directory
        directory = Path(args.dir)
        print("\n" + "=" * 80)
        print("🏥 Histopathology Normal Image Renaming Tool")
        print("=" * 80)
        rename_images_in_directory(
            directory,
            prefix=args.prefix,
            start_index=args.start_index,
            dry_run=dry_run,
            backup=backup
        )
    elif args.no_interactive:
        # Batch mode - rename full dataset structure
        dataset_path = Path(args.dataset_path)
        print("\n" + "=" * 80)
        print("🏥 Histopathology Normal Image Renaming Tool - Batch Mode")
        print("=" * 80)
        rename_dataset_structure(
            dataset_path,
            train_prefix="normal_train",
            val_prefix="normal_val",
            dry_run=dry_run,
            backup=backup
        )
    else:
        # Interactive mode - let user select directories
        dataset_path = Path(args.dataset_path)
        selected_dirs = interactive_directory_selector(dataset_path)
        
        if selected_dirs:
            for dir_path, prefix, label in selected_dirs:
                print(f"\n{'='*80}")
                print(f"Processing: {label}")
                print(f"{'='*80}")
                rename_images_in_directory(
                    dir_path,
                    prefix=prefix,
                    start_index=1,
                    dry_run=dry_run,
                    backup=backup
                )
            
            print("\n" + "=" * 80)
            if dry_run:
                print("✅ All dry runs complete! Review the renaming plans above.")
                print("   Run with --execute flag to perform the actual renaming")
            else:
                print("✅ All renaming complete!")
            print("=" * 80)
        else:
            print("❌ No directories selected")

if __name__ == "__main__":
    main()
