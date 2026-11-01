"""
Common Result Manager Layer
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Responsibilities:
- Standardized data structure for duplicate candidates across all modalities.
- Canonical file pair ordering to prevent reverse duplicates (A <-> B vs B <-> A).
- Prevention of self-comparisons (A <-> A).
- Result deduplication and multi-algorithm result merging.
- Normalization of similarity scores to a 0-100 scale.
- Accurate category assignment:
    1. exact_duplicate
    2. near_duplicate_image
    3. near_duplicate_document
    4. cross_format_document
    5. possible_duplicate
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Allowed standardized duplicate categories
VALID_DUPLICATE_CATEGORIES = {
    "exact_duplicate",
    "near_duplicate_image",
    "near_duplicate_document",
    "cross_format_document",
    "possible_duplicate",
}


@dataclass
class DuplicateRecord:
    """Unified representation of a detected duplicate file pair."""
    file_path_1: str
    file_path_2: str
    file_name_1: str
    file_name_2: str
    file_type: str  # 'image', 'document', 'cross_format', 'other'
    duplicate_type: str  # One of VALID_DUPLICATE_CATEGORIES
    similarity_score: float  # Normalized 0.0 to 100.0
    detection_algorithm: str  # e.g., 'SHA-256', 'pHash + dHash', 'TF-IDF + Cosine Similarity'
    detection_details: str  # Descriptive metrics, hash distances, or digest fragments
    detection_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def to_csv_row(self) -> List[Union[str, float]]:
        """Exports the record matching the required combined CSV schema."""
        return [
            self.file_path_1,
            self.file_path_2,
            self.file_name_1,
            self.file_name_2,
            self.file_type,
            self.duplicate_type,
            f"{self.similarity_score:.2f}",
            self.detection_algorithm,
            self.detection_details,
            self.detection_timestamp,
        ]


def canonicalize_pair(
    path1: Union[str, Path], path2: Union[str, Path]
) -> Optional[Tuple[str, str, str, str]]:
    """
    Returns (p1, p2, name1, name2) sorted canonically.
    Returns None if self-comparison (path1 == path2).
    """
    p1_str = str(Path(path1).resolve())
    p2_str = str(Path(path2).resolve())

    if p1_str == p2_str:
        return None

    if p1_str < p2_str:
        return p1_str, p2_str, Path(p1_str).name, Path(p2_str).name
    else:
        return p2_str, p1_str, Path(p2_str).name, Path(p1_str).name


class ResultManager:
    """
    Manages and deduplicates detection results from multiple algorithms.
    Merges multi-algorithm hits for identical file pairs without data loss.
    """

    def __init__(self):
        # Key: (file_path_1, file_path_2) -> DuplicateRecord
        self._records: Dict[Tuple[str, str], DuplicateRecord] = {}

    def add_record(
        self,
        file_path_1: Union[str, Path],
        file_path_2: Union[str, Path],
        file_type: str,
        duplicate_type: str,
        similarity_score: float,
        detection_algorithm: str,
        detection_details: str,
        timestamp: Optional[str] = None,
    ) -> Optional[DuplicateRecord]:
        """
        Adds or merges a duplicate pair. Automatically enforces canonical ordering.
        """
        pair = canonicalize_pair(file_path_1, file_path_2)
        if pair is None:
            # Self-comparison ignored
            return None

        p1, p2, name1, name2 = pair
        key = (p1, p2)

        # Normalize score
        normalized_score = max(0.0, min(100.0, round(float(similarity_score), 2)))

        if key in self._records:
            # Merge algorithm details if this pair was already detected by another algorithm
            existing = self._records[key]

            # If the new finding is higher priority (e.g. exact_duplicate > near_duplicate)
            if duplicate_type == "exact_duplicate" or existing.duplicate_type != "exact_duplicate":
                existing.duplicate_type = duplicate_type
                existing.similarity_score = max(existing.similarity_score, normalized_score)

            if detection_algorithm not in existing.detection_algorithm:
                existing.detection_algorithm = f"{existing.detection_algorithm} | {detection_algorithm}"

            existing.detection_details = (
                f"{existing.detection_details}; [{detection_algorithm}]: {detection_details}"
            )
            return existing

        record = DuplicateRecord(
            file_path_1=p1,
            file_path_2=p2,
            file_name_1=name1,
            file_name_2=name2,
            file_type=file_type,
            duplicate_type=duplicate_type,
            similarity_score=normalized_score,
            detection_algorithm=detection_algorithm,
            detection_details=detection_details,
            detection_timestamp=timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._records[key] = record
        return record

    def get_records(self) -> List[DuplicateRecord]:
        """Returns all deduplicated records ordered canonically."""
        return sorted(self._records.values(), key=lambda r: (r.file_path_1, r.file_path_2))

    def filter_records(
        self,
        file_type: Optional[str] = None,
        duplicate_type: Optional[str] = None,
        min_similarity: Optional[float] = None,
        algorithm: Optional[str] = None,
    ) -> List[DuplicateRecord]:
        """Filters stored records according to criteria."""
        filtered = []
        for r in self.get_records():
            if file_type and r.file_type.lower() != file_type.lower():
                continue
            if duplicate_type and r.duplicate_type.lower() != duplicate_type.lower():
                continue
            if min_similarity is not None and r.similarity_score < min_similarity:
                continue
            if algorithm and algorithm.lower() not in r.detection_algorithm.lower():
                continue
            filtered.append(r)
        return filtered

    def clear(self):
        """Clears all stored records."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

