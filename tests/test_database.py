"""
Unit and Integration Test Suite for SQLite Database Layer
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from database_manager import DatabaseManager
from result_manager import DuplicateRecord


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="t243_db_test_")
        self.db_path = Path(self.test_dir) / "test_duplicates.db"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_database_initialization(self):
        """Verify that all 4 tables are created during initialization."""
        stats = self.db.get_database_stats()
        self.assertIn("files", stats)
        self.assertIn("detection_results", stats)
        self.assertIn("classification_results", stats)
        self.assertIn("scan_history", stats)
        self.assertEqual(stats["files"], 0)

    def test_02_upsert_file(self):
        """Verify file upserting and duplicate path idempotency."""
        f_id_1 = self.db.upsert_file(
            file_path="C:/docs/report1.pdf",
            filename="report1.pdf",
            extension=".pdf",
            file_size=2048,
            file_type="document",
            sha256_hash="abc123hash",
        )
        self.assertGreater(f_id_1, 0)

        # Upserting identical path should update existing record and return same ID
        f_id_2 = self.db.upsert_file(
            file_path="C:/docs/report1.pdf",
            filename="report1.pdf",
            extension=".pdf",
            file_size=4096,
            file_type="document",
            sha256_hash="abc123hash",
        )
        self.assertEqual(f_id_1, f_id_2)

        stats = self.db.get_database_stats()
        self.assertEqual(stats["files"], 1)

    def test_03_save_and_query_detection_results(self):
        """Verify saving and filtering duplicate records."""
        rec1 = DuplicateRecord(
            file_path_1="C:/images/photo_a.jpg",
            file_path_2="C:/images/photo_b.jpg",
            file_name_1="photo_a.jpg",
            file_name_2="photo_b.jpg",
            file_type="image",
            duplicate_type="near_duplicate_image",
            similarity_score=88.5,
            detection_algorithm="pHash + dHash",
            detection_details="pHash 89%, dHash 88%",
        )
        rec2 = DuplicateRecord(
            file_path_1="C:/docs/contract_v1.txt",
            file_path_2="C:/docs/contract_v2.docx",
            file_name_1="contract_v1.txt",
            file_name_2="contract_v2.docx",
            file_type="cross_format",
            duplicate_type="cross_format_document",
            similarity_score=100.0,
            detection_algorithm="TF-IDF + Cosine Similarity",
            detection_details="100% text match",
        )

        saved = self.db.save_results([rec1, rec2])
        self.assertEqual(saved, 2)

        stats = self.db.get_database_stats()
        self.assertEqual(stats["files"], 4)
        self.assertEqual(stats["detection_results"], 2)

        # Query all >= 75%
        results_all = self.db.get_detection_results(min_similarity=75.0)
        self.assertEqual(len(results_all), 2)

        # Query filtered by duplicate_type
        results_cross = self.db.get_detection_results(duplicate_type="cross_format_document")
        self.assertEqual(len(results_cross), 1)
        self.assertEqual(results_cross[0]["similarity_score"], 100.0)

        # Query filtered by higher similarity >= 90%
        results_high = self.db.get_detection_results(min_similarity=90.0)
        self.assertEqual(len(results_high), 1)
        self.assertEqual(results_high[0]["duplicate_type"], "cross_format_document")

    def test_04_scan_history_audit(self):
        """Verify logging scan events and querying audit history."""
        scan_id = self.db.log_scan(
            scanned_folder="dataset",
            total_files=50,
            total_matches=5,
            status="COMPLETED",
        )
        self.assertGreater(scan_id, 0)

        history = self.db.get_scan_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["scanned_folder"], "dataset")
        self.assertEqual(history[0]["total_files"], 50)
        self.assertEqual(history[0]["total_matches"], 5)


if __name__ == "__main__":
    unittest.main()

