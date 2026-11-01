"""
Unified Report Generator Layer
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Responsibilities:
- Generates combined_duplicate_report.csv containing all modalities.
- Ensures existing reports (duplicate_report.csv, image_duplicate_report.csv) are protected.
- Provides flexible filtering by file type, duplicate category, similarity, and algorithm.
- Computes execution summary metrics from actual scan results.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from result_manager import DuplicateRecord

logger = logging.getLogger(__name__)

COMBINED_REPORT_DEFAULT = "combined_duplicate_report.csv"
PROTECTED_REPORT_FILENAMES: Set[str] = {
    "duplicate_report.csv",
    "image_duplicate_report.csv",
}

REPORT_HEADER: List[str] = [
    "file_path_1",
    "file_path_2",
    "file_name_1",
    "file_name_2",
    "file_type",
    "duplicate_type",
    "similarity_score",
    "detection_algorithm",
    "detection_details",
    "detection_timestamp",
]


@dataclass
class ExecutionSummary:
    """Summary metrics calculated strictly from actual execution."""
    total_files_scanned: int
    exact_duplicate_pairs: int
    near_duplicate_image_pairs: int
    near_duplicate_document_pairs: int
    cross_format_pairs: int
    possible_duplicate_pairs: int
    total_detected_pairs: int
    scan_duration_seconds: float
    output_report_file: str

    def print_summary(self):
        """Prints formatted summary to console."""
        print("\n" + "=" * 50)
        print("           T243 DUPLICATE DETECTION")
        print("=" * 50)
        print(f"Total files scanned:               {self.total_files_scanned}")
        print(f"Exact duplicate pairs (SHA-256):   {self.exact_duplicate_pairs}")
        print(f"Near-duplicate image pairs:        {self.near_duplicate_image_pairs}")
        print(f"Near-duplicate document pairs:     {self.near_duplicate_document_pairs}")
        if self.cross_format_pairs > 0:
            print(f"Cross-format document pairs:       {self.cross_format_pairs}")
        print(f"Total detected duplicate pairs:    {self.total_detected_pairs}")
        print(f"Scan duration:                     {self.scan_duration_seconds:.2f}s")
        print(f"Combined report generated:         {self.output_report_file}")
        print("=" * 50 + "\n")


def export_combined_report(
    records: List[DuplicateRecord],
    output_path: Union[str, Path] = COMBINED_REPORT_DEFAULT,
) -> Path:
    """
    Exports unified duplicate records into combined_duplicate_report.csv.
    Enforces protection of individual existing module reports.

    Args:
        records: List of DuplicateRecord objects.
        output_path: Output CSV destination.

    Returns:
        Path to generated CSV file.
    """
    out_file = Path(output_path)

    # Strictly protect teammate's and individual reports from being overwritten
    if out_file.name.lower() in PROTECTED_REPORT_FILENAMES:
        raise ValueError(
            f"Cannot overwrite protected report file: '{out_file.name}'. "
            f"Use '{COMBINED_REPORT_DEFAULT}' or another destination."
        )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REPORT_HEADER)
        for rec in records:
            writer.writerow(rec.to_csv_row())

    logger.info("Exported %d records to '%s'.", len(records), out_file)
    return out_file


def calculate_execution_summary(
    records: List[DuplicateRecord],
    total_scanned: int,
    duration: float,
    report_path: Union[str, Path] = COMBINED_REPORT_DEFAULT,
) -> ExecutionSummary:
    """
    Aggregates metrics directly from detected duplicate records.
    """
    exact_count = 0
    image_count = 0
    doc_count = 0
    cross_count = 0
    possible_count = 0

    for r in records:
        cat = r.duplicate_type
        if cat == "exact_duplicate":
            exact_count += 1
        elif cat == "near_duplicate_image":
            image_count += 1
        elif cat == "near_duplicate_document":
            doc_count += 1
        elif cat == "cross_format_document":
            cross_count += 1
        elif cat == "possible_duplicate":
            possible_count += 1

    return ExecutionSummary(
        total_files_scanned=total_scanned,
        exact_duplicate_pairs=exact_count,
        near_duplicate_image_pairs=image_count,
        near_duplicate_document_pairs=doc_count,
        cross_format_pairs=cross_count,
        possible_duplicate_pairs=possible_count,
        total_detected_pairs=len(records),
        scan_duration_seconds=duration,
        output_report_file=str(report_path),
    )

