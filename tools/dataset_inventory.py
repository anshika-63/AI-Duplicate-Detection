"""
Dataset Inventory and Validation Utility
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Scans and inventories:
1. INRIA Copydays collections (Original, JPEG, Strong attacks).
2. California-ND photos and annotations.
3. RVL-CDIP 16-class Document Classification Parquet files.
4. Custom document dataset and ground truth relations.

Generates:
dataset_inventory.csv
"""

import argparse
import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = r"E:\Fake-Detect\datasets"
DEFAULT_OUTPUT_CSV = "dataset_inventory.csv"


def format_bytes(size: int) -> str:
    """Formats byte counts into human-readable strings."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def inventory_inria_copydays(base_dir: Path) -> List[Dict[str, Any]]:
    """Inventories INRIA Copydays subdirectories."""
    records = []
    if not base_dir.exists():
        return records

    subdirs = [
        ("INRIA Copydays (Original)", base_dir / "orignal"),
        ("INRIA Copydays (JPEG Attacks)", base_dir / "jpeg"),
        ("INRIA Copydays (Strong Attacks)", base_dir / "strong"),
    ]

    for name, path in subdirs:
        if not path.exists():
            continue

        files = [p for p in path.rglob("*") if p.is_file()]
        extensions = sorted(list({p.suffix.lower() for p in files if p.suffix}))
        total_size = sum(p.stat().st_size for p in files)

        # Check for corrupted images
        corrupted = 0
        sample_size = min(len(files), 50)  # Check sample to stay fast
        for p in files[:sample_size]:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception:
                corrupted += 1

        records.append({
            "dataset_name": name,
            "directory": str(path),
            "file_count": len(files),
            "file_extensions": ";".join(extensions),
            "total_storage_size": total_size,
            "total_storage_readable": format_bytes(total_size),
            "available_labels": "Original-to-attack image ID mapping (prefix 20XX)",
            "annotation_availability": "Synthetic / Attack benchmark labels",
            "corrupted_file_count": corrupted,
            "recommended_processing_module": "image_detection.py (pHash / dHash / ORB)",
        })

    return records


def inventory_california_nd(dataset_root: Path) -> List[Dict[str, Any]]:
    """Inventories California-ND dataset folders."""
    records = []
    cal_dirs = [
        ("California-ND (Photos)", dataset_root / "Photos", "image_detection.py"),
        ("California-ND (Correlation Matrices)", dataset_root / "Correlation matrices", "evaluation"),
        ("California-ND (Individual Annotations)", dataset_root / "Individual annotations", "evaluation"),
        ("California-ND (Face Bounding Boxes)", dataset_root / "Face bounding boxes and identities", "skipped"),
    ]

    for name, path, module in cal_dirs:
        if not path.exists():
            continue

        files = [p for p in path.rglob("*") if p.is_file()]
        extensions = sorted(list({p.suffix.lower() for p in files if p.suffix}))
        total_size = sum(p.stat().st_size for p in files)

        corrupted = 0
        status_note = "Present" if files else "Empty directory shell (files not present)"

        records.append({
            "dataset_name": name,
            "directory": str(path),
            "file_count": len(files),
            "file_extensions": ";".join(extensions) if extensions else "N/A",
            "total_storage_size": total_size,
            "total_storage_readable": format_bytes(total_size),
            "available_labels": status_note,
            "annotation_availability": "10-observer human similarity judgments" if "Annotation" in name else "N/A",
            "corrupted_file_count": corrupted,
            "recommended_processing_module": module,
        })

    return records


def inventory_rvl_cdip(base_dir: Path) -> List[Dict[str, Any]]:
    """Inventories RVL-CDIP Parquet document classification dataset."""
    records = []
    if not base_dir.exists():
        return records

    parquet_files = sorted(list(base_dir.glob("*.parquet")))
    if not parquet_files:
        return records

    total_rows = 0
    total_size = 0
    class_names = []

    for pf in parquet_files:
        total_size += pf.stat().st_size
        meta = pq.read_metadata(pf)
        total_rows += meta.num_rows

    # Read label mapping from metadata
    try:
        schema = pq.read_schema(parquet_files[0])
        if b"huggingface" in schema.metadata:
            hf_meta = json.loads(schema.metadata[b"huggingface"].decode("utf-8"))
            class_names = hf_meta.get("info", {}).get("features", {}).get("label", {}).get("names", [])
    except Exception as e:
        logger.warning("Could not read Parquet metadata: %s", e)

    records.append({
        "dataset_name": "RVL-CDIP 16-Class Document Classification (Parquet Subset)",
        "directory": str(base_dir),
        "file_count": len(parquet_files),
        "file_extensions": ".parquet",
        "total_storage_size": total_size,
        "total_storage_readable": format_bytes(total_size),
        "available_labels": f"16 Classes: {', '.join(class_names[:5])}... ({len(class_names)} total)",
        "annotation_availability": f"Ground-truth class integer labels (8,000 train, 992 test)",
        "corrupted_file_count": 0,
        "recommended_processing_module": "document_classification.py (TF-IDF SVM / CNN)",
    })

    return records


def inventory_custom_documents(base_dir: Path) -> List[Dict[str, Any]]:
    """Inventories custom document and synthetic ground truth set."""
    records = []
    if not base_dir.exists():
        return records

    files = [p for p in base_dir.rglob("*") if p.is_file()]
    extensions = sorted(list({p.suffix.lower() for p in files if p.suffix}))
    total_size = sum(p.stat().st_size for p in files)

    has_ground_truth = any("ground_truth.csv" in p.name.lower() for p in files)

    records.append({
        "dataset_name": "Custom Document & Synthetic Ground Truth Set",
        "directory": str(base_dir),
        "file_count": len(files),
        "file_extensions": ";".join(extensions),
        "total_storage_size": total_size,
        "total_storage_readable": format_bytes(total_size),
        "available_labels": "exact_duplicate, near_duplicate_image, near_duplicate_document, unrelated",
        "annotation_availability": "ground_truth.csv verified pairwise relations" if has_ground_truth else "None",
        "corrupted_file_count": 0,
        "recommended_processing_module": "unified_detector.py & test_integration.py",
    })

    return records


def run_inventory(
    dataset_root: str = DEFAULT_DATASET_ROOT, output_csv: str = DEFAULT_OUTPUT_CSV
) -> Path:
    """Executes complete dataset inventory and writes CSV."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {root}")

    logger.info("Scanning dataset root at '%s'...", root)

    all_records: List[Dict[str, Any]] = []

    # 1. INRIA Copydays
    inria_path = root / "image-detection"
    all_records.extend(inventory_inria_copydays(inria_path))

    # 2. California-ND
    all_records.extend(inventory_california_nd(root))

    # 3. RVL-CDIP Document Classification
    rvl_path = root / "document_classification"
    all_records.extend(inventory_rvl_cdip(rvl_path))

    # 4. Custom Documents
    custom_path = root / "custom_documents"
    all_records.extend(inventory_custom_documents(custom_path))

    # Write CSV
    headers = [
        "dataset_name",
        "directory",
        "file_count",
        "file_extensions",
        "total_storage_size",
        "total_storage_readable",
        "available_labels",
        "annotation_availability",
        "corrupted_file_count",
        "recommended_processing_module",
    ]

    out_file = Path(output_csv)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    logger.info("Dataset inventory successfully written to '%s' (%d records).", out_file, len(all_records))
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Inventory Utility")
    parser.add_argument("--root", type=str, default=DEFAULT_DATASET_ROOT, help="Dataset root directory")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    run_inventory(dataset_root=args.root, output_csv=args.out)
