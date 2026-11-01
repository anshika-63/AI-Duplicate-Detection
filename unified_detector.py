"""
Unified Duplicate Detection Engine
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Orchestrates:
- Module 1: Exact Duplicate Detection using SHA-256 (reusing main.py).
- Module 2: Near-Duplicate Image Detection using pHash and dHash (reusing image_detection.py).
- Module 3: Document Similarity Detection using TF-IDF & Cosine Similarity (reusing document_detection.py).

Features:
- Automatic file routing based on format and MIME characteristics.
- Canonical pair deduplication through ResultManager.
- Exports combined_duplicate_report.csv while safeguarding existing module reports.
- Comprehensive CLI interface with selective execution flags.
"""

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Module 1: Exact SHA-256 hashing (preserved from teammate's main.py)
from main import calculate_hash

# Module 2: Near-duplicate image detection
from image_detection import (
    DEFAULT_HASH_SIZE,
    calculate_hamming_distance,
    calculate_similarity,
    categorize_duplicate,
    compute_dhash,
    compute_phash,
)

# Module 3: Document similarity detection
from document_detection import (
    calculate_tfidf_similarity,
    categorize_document_duplicate,
    extract_document_text,
)

# Core Integration Layers
from file_scanner import (
    OCR_ELIGIBLE_EXTENSIONS,
    ScanInventory,
    ScannedFile,
    classify_file_type,
    scan_directory,
)
from report_generator import (
    COMBINED_REPORT_DEFAULT,
    ExecutionSummary,
    calculate_execution_summary,
    export_combined_report,
)
from result_manager import DuplicateRecord, ResultManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 80.0


class UnifiedDetector:
    """
    Central orchestration engine coordinating Exact (SHA-256),
    Image (pHash/dHash), and Document (TF-IDF/Cosine) detection pipelines.
    """

    def __init__(
        self,
        target_folder: Union[str, Path],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        enable_ocr: bool = False,
        report_output: Union[str, Path] = COMBINED_REPORT_DEFAULT,
    ):
        self.target_folder = Path(target_folder).resolve()
        self.threshold = float(threshold)
        self.enable_ocr = enable_ocr
        self.report_output = Path(report_output)
        self.result_manager = ResultManager()

    def run_exact_detection(self, inventory: ScanInventory) -> int:
        """
        Executes Module 1: Exact duplicate detection using SHA-256.
        Groups files by SHA-256 hash. Complexity: O(N) hashing.
        """
        logger.info("Executing Exact Duplicate Detection (SHA-256)...")
        hash_groups = {}
        target_files = inventory.all_files

        for file_rec in target_files:
            try:
                f_hash = calculate_hash(file_rec.path)
                if f_hash not in hash_groups:
                    hash_groups[f_hash] = []
                hash_groups[f_hash].append(file_rec)
            except Exception as e:
                logger.warning("Could not calculate SHA-256 for '%s': %s", file_rec.path, e)

        added_pairs = 0
        for f_hash, group in hash_groups.items():
            if len(group) > 1:
                # Every pairwise combination within the identical hash group is an exact duplicate
                for f1, f2 in combinations(group, 2):
                    file_type = "mixed" if f1.file_type != f2.file_type else f1.file_type
                    self.result_manager.add_record(
                        file_path_1=f1.path,
                        file_path_2=f2.path,
                        file_type=file_type,
                        duplicate_type="exact_duplicate",
                        similarity_score=100.0,
                        detection_algorithm="SHA-256",
                        detection_details=f"Identical SHA-256 Digest: {f_hash}",
                    )
                    added_pairs += 1

        logger.info("Exact detection identified %d matching pair(s).", added_pairs)
        return added_pairs

    def run_image_detection(self, inventory: ScanInventory) -> int:
        """
        Executes Module 2: Near-duplicate image detection using pHash & dHash.
        Memory-efficient: Computes hashes and closes images immediately.
        """
        images = inventory.images
        if len(images) < 2:
            logger.info("Fewer than 2 images found; skipping image pair comparison.")
            return 0

        logger.info("Executing Near-Duplicate Image Detection (pHash + dHash) on %d images...", len(images))

        # 1. Precompute 64-bit perceptual and difference hashes
        image_hashes = []
        for img_rec in images:
            try:
                p = compute_phash(img_rec.path)
                d = compute_dhash(img_rec.path)
                image_hashes.append((img_rec, p, d))
            except Exception as e:
                logger.warning("Failed to generate image hashes for '%s': %s", img_rec.file_name, e)

        # 2. Pairwise comparison using itertools.combinations (eliminates self and reverse pairs)
        added_pairs = 0
        for (img1, p1, d1), (img2, p2, d2) in combinations(image_hashes, 2):
            p_dist = calculate_hamming_distance(p1, p2)
            d_dist = calculate_hamming_distance(d1, d2)

            p_sim = calculate_similarity(p_dist, total_bits=64)
            d_sim = calculate_similarity(d_dist, total_bits=64)

            # Categorize duplicate
            category, is_dup = categorize_duplicate(
                p_sim, d_sim, p_dist, d_dist, threshold=self.threshold
            )

            if is_dup:
                dup_type = "exact_duplicate" if (p_dist == 0 and d_dist == 0) else "near_duplicate_image"
                overall_sim = max(p_sim, d_sim)
                details = f"pHash dist={p_dist} ({p_sim:.2f}%), dHash dist={d_dist} ({d_sim:.2f}%)"

                self.result_manager.add_record(
                    file_path_1=img1.path,
                    file_path_2=img2.path,
                    file_type="image",
                    duplicate_type=dup_type,
                    similarity_score=overall_sim,
                    detection_algorithm="pHash + dHash",
                    detection_details=details,
                )
                added_pairs += 1

        logger.info("Image detection identified %d duplicate pair(s).", added_pairs)
        return added_pairs

    def run_document_detection(self, inventory: ScanInventory) -> int:
        """
        Executes Module 3: Document similarity detection using TF-IDF & Cosine Similarity.
        Supports TXT, PDF, DOCX, and optional OCR for scanned images/TIFFs.
        """
        docs = inventory.documents
        if len(docs) < 2:
            logger.info("Fewer than 2 documents found; skipping document comparison.")
            return 0

        logger.info(
            "Executing Document Similarity Detection (TF-IDF + Cosine) on %d documents...",
            len(docs),
        )

        # 1. Pre-extract and cache text
        extracted_docs = []
        for doc_rec in docs:
            try:
                text = extract_document_text(
                    doc_rec.path, enable_ocr=self.enable_ocr, use_cache=True
                )
                if text and text.strip():
                    extracted_docs.append((doc_rec, text))
                else:
                    logger.debug("Document '%s' contains no extractable text.", doc_rec.file_name)
            except Exception as e:
                logger.warning("Failed to extract text from '%s': %s", doc_rec.file_name, e)

        # 2. Pairwise comparison
        added_pairs = 0
        for (doc1, text1), (doc2, text2) in combinations(extracted_docs, 2):
            sim = calculate_tfidf_similarity(text1, text2)
            category, is_dup = categorize_document_duplicate(
                sim, doc1.extension, doc2.extension, threshold=self.threshold
            )

            if is_dup:
                is_cross = doc1.extension.lower() != doc2.extension.lower()
                f_type = "cross_format" if is_cross else "document"

                details = f"TF-IDF Cosine Similarity: {sim:.2f}% (Formats: {doc1.extension} vs {doc2.extension})"
                self.result_manager.add_record(
                    file_path_1=doc1.path,
                    file_path_2=doc2.path,
                    file_type=f_type,
                    duplicate_type=category,
                    similarity_score=sim,
                    detection_algorithm="TF-IDF + Cosine Similarity",
                    detection_details=details,
                )
                added_pairs += 1

        logger.info("Document detection identified %d matching pair(s).", added_pairs)
        return added_pairs

    def run(
        self,
        run_exact: bool = True,
        run_images: bool = True,
        run_documents: bool = True,
    ) -> Tuple[List[DuplicateRecord], ExecutionSummary]:
        """
        Orchestrates full detection pipeline and generates combined report.
        """
        start_time = time.time()
        self.result_manager.clear()

        # Step 1: Scan directory and inventory files
        inventory = scan_directory(
            self.target_folder,
            recursive=True,
            include_images=run_images or run_exact,
            include_documents=run_documents or run_exact,
        )

        # Step 2: Route and execute selected modules
        if run_exact:
            self.run_exact_detection(inventory)

        if run_images:
            self.run_image_detection(inventory)

        if run_documents:
            self.run_document_detection(inventory)

        # Step 3: Collect deduplicated canonical records
        records = self.result_manager.get_records()

        # Step 4: Export combined CSV report
        out_csv = export_combined_report(records, output_path=self.report_output)

        duration = time.time() - start_time
        summary = calculate_execution_summary(
            records=records,
            total_scanned=inventory.total_count,
            duration=duration,
            report_path=out_csv,
        )

        return records, summary


def main():
    """Command-line interface entrypoint for Unified Duplicate Detection."""
    parser = argparse.ArgumentParser(
        description="T243 – Unified AI-Based Duplicate Document & Near-Duplicate Image Detection"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="dataset",
        help="Target folder to scan (default: dataset)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all detection modules (Exact SHA-256, Images, and Documents)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Run exact duplicate detection using SHA-256",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Run near-duplicate image detection using pHash & dHash",
    )
    parser.add_argument(
        "--documents",
        action="store_true",
        help="Run document similarity detection using TF-IDF & Cosine Similarity",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity percentage threshold for near-duplicates (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR text extraction for scanned documents and TIFF images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=COMBINED_REPORT_DEFAULT,
        help=f"Path to export combined report (default: {COMBINED_REPORT_DEFAULT})",
    )
    parser.add_argument(
        "--db",
        type=str,
        nargs="?",
        const="duplicate_detection.db",
        default=None,
        help="Optional SQLite database path to store scan history and detection results",
    )

    args = parser.parse_args()

    # Determine execution scope
    run_all = args.all or (not args.exact and not args.images and not args.documents)
    run_exact = run_all or args.exact
    run_images = run_all or args.images
    run_docs = run_all or args.documents

    detector = UnifiedDetector(
        target_folder=args.folder,
        threshold=args.threshold,
        enable_ocr=args.ocr,
        report_output=args.output,
    )

    records, summary = detector.run(
        run_exact=run_exact,
        run_images=run_images,
        run_documents=run_docs,
    )

    if args.db:
        from database_manager import DatabaseManager
        db = DatabaseManager(args.db)
        db.save_results(records)
        db.log_scan(
            scanned_folder=str(detector.target_folder),
            total_files=summary.total_files_scanned,
            total_matches=summary.total_detected_pairs,
        )
        print(f"Results stored to database: {args.db}")

    summary.print_summary()


if __name__ == "__main__":
    main()
