"""
Image Detection Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Features:
- Load images safely using Pillow.
- Generate perceptual hash (pHash) and difference hash (dHash) using ImageHash.
- Calculate Hamming distance and similarity percentage:
    Similarity (%) = (1 - Hamming Distance / 64) * 100
- Compare two images and identify near-duplicates with configurable thresholds.
- Scan directories recursively without loading full-resolution images into memory.
- Export candidate pairs to image_duplicate_report.csv.
- Robust error handling for corrupted and unsupported files.
"""

import csv
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imagehash
from PIL import Image, UnidentifiedImageError

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

# Default configuration constants
DEFAULT_HASH_SIZE = 8  # 8x8 yields a 64-bit hash
TOTAL_HASH_BITS = DEFAULT_HASH_SIZE * DEFAULT_HASH_SIZE  # 64 bits
DEFAULT_SIMILARITY_THRESHOLD = 85.0  # Percentage threshold for candidate duplicates
REPORT_FILE_NAME = "image_duplicate_report.csv"


@dataclass
class ImageHashRecord:
    """Stores precomputed hashes for an image file."""
    path: Path
    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash


@dataclass
class ComparisonResult:
    """Stores the comparison metrics between two images."""
    image_path_1: str
    image_path_2: str
    phash_distance: int
    phash_similarity: float
    dhash_distance: int
    dhash_similarity: float
    detection_algorithm: str
    duplicate_category: str
    is_duplicate: bool

    def to_csv_row(self) -> List[Union[str, int, float]]:
        """Converts result into a row matching the required CSV structure."""
        return [
            self.image_path_1,
            self.image_path_2,
            self.phash_distance,
            f"{self.phash_similarity:.2f}",
            self.dhash_distance,
            f"{self.dhash_similarity:.2f}",
            self.detection_algorithm,
            self.duplicate_category,
        ]


def is_supported_image(file_path: Union[str, Path]) -> bool:
    """Checks whether the file has a supported image extension."""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Safely opens and loads an image using Pillow.

    Args:
        image_path: Path to the image file.

    Returns:
        Pillow Image object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is unsupported.
        UnidentifiedImageError: If the image cannot be read or is corrupted.
        OSError: If an OS error occurs while reading the image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not is_supported_image(path):
        raise ValueError(
            f"Unsupported image format: {path.suffix}. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    # Open and verify image data
    try:
        with Image.open(path) as img:
            # Convert to RGB to ensure uniform color space and force loading image data
            loaded_img = img.convert("RGB")
            return loaded_img
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("Corrupted or unreadable image file '%s': %s", path, e)
        raise


def compute_phash(
    image_input: Union[str, Path, Image.Image], hash_size: int = DEFAULT_HASH_SIZE
) -> imagehash.ImageHash:
    """
    Generates perceptual hash (pHash) for an image.

    Args:
        image_input: Image path or PIL Image object.
        hash_size: Hash dimension (default 8 -> 64-bit hash).

    Returns:
        imagehash.ImageHash object.
    """
    if isinstance(image_input, (str, Path)):
        img = load_image(image_input)
    else:
        img = image_input
    return imagehash.phash(img, hash_size=hash_size)


def compute_dhash(
    image_input: Union[str, Path, Image.Image], hash_size: int = DEFAULT_HASH_SIZE
) -> imagehash.ImageHash:
    """
    Generates difference hash (dHash) for an image.

    Args:
        image_input: Image path or PIL Image object.
        hash_size: Hash dimension (default 8 -> 64-bit hash).

    Returns:
        imagehash.ImageHash object.
    """
    if isinstance(image_input, (str, Path)):
        img = load_image(image_input)
    else:
        img = image_input
    return imagehash.dhash(img, hash_size=hash_size)


def calculate_hamming_distance(
    hash1: imagehash.ImageHash, hash2: imagehash.ImageHash
) -> int:
    """
    Calculates the Hamming distance between two image hashes.
    """
    return hash1 - hash2


def calculate_similarity(
    hamming_distance: int, total_bits: int = TOTAL_HASH_BITS
) -> float:
    """
    Calculates similarity percentage using the formula:
    Similarity (%) = (1 - Hamming Distance / 64) * 100

    Args:
        hamming_distance: Number of bit differences.
        total_bits: Total bit length (default 64).

    Returns:
        Similarity score between 0.0 and 100.0.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be greater than 0")
    similarity = (1.0 - (hamming_distance / float(total_bits))) * 100.0
    return max(0.0, min(100.0, round(similarity, 2)))


def categorize_duplicate(
    phash_sim: float,
    dhash_sim: float,
    phash_dist: int,
    dhash_dist: int,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[str, bool]:
    """
    Categorizes the relationship between two images based on hash metrics.

    Categories:
    - Exact Duplicate: 100% similarity on both hashes (0 distance).
    - Near Duplicate: Both hashes meet or exceed the similarity threshold.
    - Potential Near-Duplicate: At least one hash meets or exceeds threshold.
    - Non-Duplicate: Neither hash meets the threshold.

    Returns:
        Tuple of (category_name, is_duplicate_boolean).
    """
    if phash_dist == 0 and dhash_dist == 0:
        return "Exact Duplicate", True

    if phash_sim >= threshold and dhash_sim >= threshold:
        return "Near Duplicate", True

    if phash_sim >= threshold or dhash_sim >= threshold:
        return "Potential Near-Duplicate", True

    return "Non-Duplicate", False


def compare_images(
    img1_path: Union[str, Path],
    img2_path: Union[str, Path],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> ComparisonResult:
    """
    Compares two image files using pHash and dHash.

    Args:
        img1_path: Path to first image.
        img2_path: Path to second image.
        threshold: Minimum similarity percentage to consider near-duplicate.
        hash_size: Hash size (default 8).

    Returns:
        ComparisonResult object.
    """
    total_bits = hash_size * hash_size

    # Load and hash first image
    p1 = compute_phash(img1_path, hash_size=hash_size)
    d1 = compute_dhash(img1_path, hash_size=hash_size)

    # Load and hash second image
    p2 = compute_phash(img2_path, hash_size=hash_size)
    d2 = compute_dhash(img2_path, hash_size=hash_size)

    # Distances
    p_dist = calculate_hamming_distance(p1, p2)
    d_dist = calculate_hamming_distance(d1, d2)

    # Similarities
    p_sim = calculate_similarity(p_dist, total_bits=total_bits)
    d_sim = calculate_similarity(d_dist, total_bits=total_bits)

    category, is_dup = categorize_duplicate(
        p_sim, d_sim, p_dist, d_dist, threshold=threshold
    )

    return ComparisonResult(
        image_path_1=str(img1_path),
        image_path_2=str(img2_path),
        phash_distance=p_dist,
        phash_similarity=p_sim,
        dhash_distance=d_dist,
        dhash_similarity=d_sim,
        detection_algorithm="pHash + dHash",
        duplicate_category=category,
        is_duplicate=is_dup,
    )


def scan_image_folder(
    folder_path: Union[str, Path],
    recursive: bool = True,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> Tuple[List[ImageHashRecord], List[Tuple[Path, str]]]:
    """
    Scans a folder for supported images and computes hashes on-the-fly.
    Images are closed immediately after computing hashes to prevent excessive memory usage.

    Args:
        folder_path: Directory path to scan.
        recursive: Whether to scan subdirectories recursively.
        hash_size: Hash size parameter.

    Returns:
        Tuple of (valid_records, failed_files).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Directory not found: {folder}")

    pattern = "**/*" if recursive else "*"
    valid_records: List[ImageHashRecord] = []
    failed_files: List[Tuple[Path, str]] = []

    for file_path in folder.glob(pattern):
        if not file_path.is_file():
            continue

        if not is_supported_image(file_path):
            continue

        try:
            # Load and compute hashes immediately, freeing memory right after
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                p = imagehash.phash(rgb_img, hash_size=hash_size)
                d = imagehash.dhash(rgb_img, hash_size=hash_size)
            valid_records.append(ImageHashRecord(path=file_path, phash=p, dhash=d))
        except (UnidentifiedImageError, OSError, Exception) as exc:
            logger.warning("Skipping unreadable image '%s': %s", file_path, exc)
            failed_files.append((file_path, str(exc)))

    return valid_records, failed_files


def find_duplicate_pairs(
    records: List[ImageHashRecord],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    include_non_duplicates: bool = False,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> List[ComparisonResult]:
    """
    Compares candidate image pairs without comparing an image with itself
    and without duplicate pair entries (e.g. comparing A and B once).

    Args:
        records: List of ImageHashRecord objects.
        threshold: Similarity percentage threshold.
        include_non_duplicates: If True, include pairs that did not meet threshold.
        hash_size: Hash size used.

    Returns:
        List of ComparisonResult objects.
    """
    total_bits = hash_size * hash_size
    results: List[ComparisonResult] = []

    # Use combinations to ensure no self-comparisons and no reversed duplicates (A,B) vs (B,A)
    for rec1, rec2 in combinations(records, 2):
        p_dist = calculate_hamming_distance(rec1.phash, rec2.phash)
        d_dist = calculate_hamming_distance(rec1.dhash, rec2.dhash)

        p_sim = calculate_similarity(p_dist, total_bits=total_bits)
        d_sim = calculate_similarity(d_dist, total_bits=total_bits)

        category, is_dup = categorize_duplicate(
            p_sim, d_sim, p_dist, d_dist, threshold=threshold
        )

        if is_dup or include_non_duplicates:
            results.append(
                ComparisonResult(
                    image_path_1=str(rec1.path),
                    image_path_2=str(rec2.path),
                    phash_distance=p_dist,
                    phash_similarity=p_sim,
                    dhash_distance=d_dist,
                    dhash_similarity=d_sim,
                    detection_algorithm="pHash + dHash",
                    duplicate_category=category,
                    is_duplicate=is_dup,
                )
            )

    return results


def export_image_report(
    results: List[ComparisonResult],
    report_path: Union[str, Path] = REPORT_FILE_NAME,
) -> Path:
    """
    Exports image comparison results into a CSV file.
    Does not overwrite duplicate_report.csv.

    Args:
        results: List of ComparisonResult objects.
        report_path: Target CSV file path (default: image_duplicate_report.csv).

    Returns:
        Path to generated CSV file.
    """
    target = Path(report_path)

    # Safety guard: Never overwrite teammate's duplicate_report.csv
    if target.name.lower() == "duplicate_report.csv":
        raise ValueError(
            "Target file cannot be 'duplicate_report.csv' as that is reserved for SHA-256 detection."
        )

    header = [
        "image_path_1",
        "image_path_2",
        "phash_distance",
        "phash_similarity",
        "dhash_distance",
        "dhash_similarity",
        "detection_algorithm",
        "duplicate_category",
    ]

    with open(target, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for res in results:
            writer.writerow(res.to_csv_row())

    logger.info("Image duplicate report generated: %s (%d rows)", target, len(results))
    return target


def run_image_detection(
    folder_path: Union[str, Path] = "dataset",
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    report_file: Union[str, Path] = REPORT_FILE_NAME,
) -> Tuple[List[ComparisonResult], Path]:
    """
    Orchestrates the entire image duplicate detection workflow:
    1. Scan folder.
    2. Compute pHash and dHash per image.
    3. Pairwise comparison of candidates.
    4. Generate CSV report.

    Args:
        folder_path: Path to dataset or image folder.
        threshold: Near-duplicate threshold percentage.
        report_file: Output CSV path.

    Returns:
        Tuple of (duplicate_results, report_path).
    """
    print(f"\nScanning directory '{folder_path}' for images...")
    records, failed = scan_image_folder(folder_path)
    print(f"Indexed {len(records)} valid image(s). (Failed/Skipped: {len(failed)})")

    if failed:
        for fpath, reason in failed:
            print(f" - Skipped: {fpath} ({reason})")

    if len(records) < 2:
        print("At least 2 valid images are required to perform duplicate detection.")
        # Generate an empty report with header
        out_path = export_image_report([], report_file)
        return [], out_path

    print(f"Comparing image pairs with similarity threshold >= {threshold}%...")
    duplicate_results = find_duplicate_pairs(records, threshold=threshold)

    print(f"Found {len(duplicate_results)} potential duplicate pair(s).")
    for res in duplicate_results:
        print(
            f" - [{res.duplicate_category}] {res.image_path_1} <-> {res.image_path_2} "
            f"(pHash Sim: {res.phash_similarity}%, dHash Sim: {res.dhash_similarity}%)"
        )

    out_path = export_image_report(duplicate_results, report_file)
    print(f"Report exported to: {out_path}\n")
    return duplicate_results, out_path


if __name__ == "__main__":
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    run_image_detection(target_dir)

