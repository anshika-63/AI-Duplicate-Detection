"""
Document Similarity Detection Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Dedicated document duplicate and near-duplicate detection layer.
Decoupled from document classification while sharing text extraction and NLP utilities.

Features:
- Multi-format text extraction: PDF, DOCX, TXT, scanned images.
- Text preprocessing and normalization.
- TF-IDF Vectorization and Cosine Similarity calculation (0-100% scale).
- Pairwise document deduplication without self or reverse comparisons.
- Identifies exact duplicates, near-duplicate text, and cross-format document matches.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from document_detection import (
    DEFAULT_DOC_SIMILARITY_THRESHOLD,
    DocumentComparisonResult,
    calculate_tfidf_similarity,
    categorize_document_duplicate,
    compare_two_documents,
    extract_document_text,
    extract_text_with_ocr,
    preprocess_text,
    scan_and_compare_documents,
)


def compute_document_similarity(
    doc1_path: Union[str, Path],
    doc2_path: Union[str, Path],
    threshold: float = DEFAULT_DOC_SIMILARITY_THRESHOLD,
    enable_ocr: bool = False,
) -> DocumentComparisonResult:
    """Computes TF-IDF Cosine Similarity between two documents."""
    return compare_two_documents(
        doc1_path=doc1_path,
        doc2_path=doc2_path,
        threshold=threshold,
        enable_ocr=enable_ocr,
    )


def find_similar_documents_in_paths(
    file_paths: List[Union[str, Path]],
    threshold: float = DEFAULT_DOC_SIMILARITY_THRESHOLD,
    enable_ocr: bool = False,
) -> List[DocumentComparisonResult]:
    """Finds near-duplicate document pairs from a list of paths."""
    paths = [Path(p) for p in file_paths]
    return scan_and_compare_documents(
        doc_paths=paths,
        threshold=threshold,
        enable_ocr=enable_ocr,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        res = compute_document_similarity(sys.argv[1], sys.argv[2])
        print(f"\nDocument Similarity Analysis:")
        print(f" - Document 1: {res.file_name_1}")
        print(f" - Document 2: {res.file_name_2}")
        print(f" - TF-IDF Cosine Similarity: {res.similarity_score:.2f}%")
        print(f" - Category: {res.duplicate_type}")
        print(f" - Is Duplicate: {res.is_duplicate}\n")
    else:
        print("Usage: python document_similarity.py <file1> <file2>")
