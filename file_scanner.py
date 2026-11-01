"""
Common File Scanner Layer
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Responsibilities:
- Recursively scans directories for supported images and documents.
- Collects file metadata: path, name, extension, size, and category.
- Handles inaccessible or permission-restricted files gracefully.
- Ignores unsupported formats, temporary directories, caches, and generated CSV reports.
- Routes files into image, document, and cross-format categories.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}

# Supported document formats
SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = {
    ".pdf",
    ".docx",
    ".txt",
}

# Image formats that can also optionally be routed to OCR
OCR_ELIGIBLE_EXTENSIONS: Set[str] = {
    ".tiff",
    ".tif",
    ".png",
    ".jpg",
    ".jpeg",
}

# Directories to ignore during scanning
IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    ".idea",
    "venv",
    ".venv",
    "env",
    ".env",
    ".kilo",
    "node_modules",
}

# Generated reports and transient files to explicitly exclude
IGNORED_FILENAMES: Set[str] = {
    "duplicate_report.csv",
    "image_duplicate_report.csv",
    "combined_duplicate_report.csv",
    ".gitignore",
}

IGNORED_FILE_EXTENSIONS: Set[str] = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".bak",
}


@dataclass
class ScannedFile:
    """Represents a discovered file and its metadata."""
    path: Path
    file_path: str
    file_name: str
    extension: str
    file_size: int
    file_type: str  # 'image', 'document', or 'other'

    @property
    def is_image(self) -> bool:
        return self.file_type == "image"

    @property
    def is_document(self) -> bool:
        return self.file_type == "document"


@dataclass
class ScanInventory:
    """Structured collection of discovered files partitioned by type."""
    root_folder: Path
    images: List[ScannedFile] = field(default_factory=list)
    documents: List[ScannedFile] = field(default_factory=list)
    others: List[ScannedFile] = field(default_factory=list)
    skipped_files: List[Tuple[Path, str]] = field(default_factory=list)

    @property
    def all_files(self) -> List[ScannedFile]:
        return self.images + self.documents + self.others

    @property
    def total_count(self) -> int:
        return len(self.images) + len(self.documents) + len(self.others)


def is_image_file(file_path: Union[str, Path]) -> bool:
    """Checks if file is a supported image."""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_document_file(file_path: Union[str, Path]) -> bool:
    """Checks if file is a supported document."""
    return Path(file_path).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def classify_file_type(file_path: Union[str, Path]) -> str:
    """Classifies a file as 'image', 'document', or 'other'."""
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "document"
    return "other"


def should_skip_directory(dir_name: str) -> bool:
    """Checks whether directory should be excluded from recursive scanning."""
    return dir_name in IGNORED_DIRECTORIES or dir_name.startswith(".")


def should_skip_file(file_name: str) -> bool:
    """Checks whether file should be ignored (reports, hidden files, caches)."""
    name_lower = file_name.lower()
    if name_lower in IGNORED_FILENAMES:
        return True
    if name_lower.startswith("."):
        return True
    ext = Path(file_name).suffix.lower()
    if ext in IGNORED_FILE_EXTENSIONS:
        return True
    return False


def scan_directory(
    folder_path: Union[str, Path],
    recursive: bool = True,
    include_images: bool = True,
    include_documents: bool = True,
) -> ScanInventory:
    """
    Recursively scans the given folder and returns a structured ScanInventory.

    Args:
        folder_path: Root folder path.
        recursive: Whether to scan subdirectories recursively.
        include_images: Whether to inventory image files.
        include_documents: Whether to inventory document files.

    Returns:
        ScanInventory object.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Target directory does not exist: {root}")

    inventory = ScanInventory(root_folder=root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude ignored directories in-place during recursive traversal
        if recursive:
            dirnames[:] = [d for d in dirnames if not should_skip_directory(d)]
        else:
            dirnames.clear()

        for fname in filenames:
            if should_skip_file(fname):
                continue

            file_path = Path(dirpath) / fname

            try:
                # File access and size check
                file_stat = file_path.stat()
                file_size = file_stat.st_size
                ext = file_path.suffix.lower()
                category = classify_file_type(file_path)

                scanned = ScannedFile(
                    path=file_path,
                    file_path=str(file_path),
                    file_name=fname,
                    extension=ext,
                    file_size=file_size,
                    file_type=category,
                )

                if category == "image" and include_images:
                    inventory.images.append(scanned)
                elif category == "document" and include_documents:
                    inventory.documents.append(scanned)
                else:
                    inventory.others.append(scanned)

            except (PermissionError, OSError) as e:
                logger.warning("Could not access file '%s': %s", file_path, e)
                inventory.skipped_files.append((file_path, str(e)))

    logger.info(
        "Scanned '%s': %d images, %d documents, %d other files (%d skipped).",
        root,
        len(inventory.images),
        len(inventory.documents),
        len(inventory.others),
        len(inventory.skipped_files),
    )

    return inventory


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    inv = scan_directory(target)
    print(f"\nInventory for '{inv.root_folder}':")
    print(f" - Images ({len(inv.images)}): {[f.file_name for f in inv.images]}")
    print(f" - Documents ({len(inv.documents)}): {[f.file_name for f in inv.documents]}")
    print(f" - Others ({len(inv.others)}): {[f.file_name for f in inv.others]}")
    print(f" - Total files: {inv.total_count}")

