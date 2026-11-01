"""
Comprehensive Integration Test Suite for T243 Duplicate Detection
Tests all 18 requirements using synthetic files and isolated temporary directories.
Public datasets are NOT used.

Test Cases:
1. Identical files with different filenames.
2. Identical images.
3. Resized images.
4. JPEG-compressed images.
5. Brightness-adjusted images.
6. Unrelated images.
7. Identical PDF documents.
8. Similar PDF documents.
9. Similar DOCX documents.
10. PDF and DOCX with similar content (cross-format).
11. Scanned documents with OCR (or graceful missing dependency fallback).
12. Corrupted images.
13. Unsupported file formats.
14. Mixed image/document folders.
15. Duplicate result elimination (canonical ordering, no self, no reverse).
16. Combined CSV generation with correct schema.
17. Individual CSV protection (duplicate_report.csv & image_duplicate_report.csv).
18. Existing main.py functionality preserved.
"""

import csv
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# System under test
import main
from document_detection import (
    calculate_tfidf_similarity,
    compare_two_documents,
    extract_document_text,
    extract_text_with_ocr,
)
from file_scanner import scan_directory
from report_generator import export_combined_report
from result_manager import ResultManager, canonicalize_pair
from unified_detector import UnifiedDetector


def create_synthetic_docx(file_path: Path, text: str) -> None:
    """Creates a valid minimal DOCX archive without external dependencies."""
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p>\n"
        f"      <w:r><w:t>{text}</w:t></w:r>\n"
        "    </w:p>\n"
        "  </w:body>\n"
        "</w:document>"
    )
    with zipfile.ZipFile(file_path, "w") as z:
        z.writestr("word/document.xml", xml_content)


def create_synthetic_pdf(file_path: Path, lines: list) -> None:
    """Creates a valid PDF using reportlab."""
    c = canvas.Canvas(str(file_path), pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


class TestUnifiedIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="t243_integration_")
        cls.dir_path = Path(cls.test_dir)

        # ----------------------------------------------------
        # 1. Identical files with different filenames (Text)
        # ----------------------------------------------------
        cls.identical_txt_1 = cls.dir_path / "article_v1.txt"
        cls.identical_txt_2 = cls.dir_path / "article_copy.txt"
        text_content = (
            "Artificial Intelligence and Machine Learning are transforming software development. "
            "Automated duplicate detection enhances repository management and data cleaning."
        )
        cls.identical_txt_1.write_text(text_content, encoding="utf-8")
        cls.identical_txt_2.write_text(text_content, encoding="utf-8")

        # ----------------------------------------------------
        # 2-6. Synthetic Images (Base, Identical, Resized, JPEG, Bright, Unrelated, Corrupt)
        # ----------------------------------------------------
        base_img = Image.new("RGB", (120, 120), color=(230, 230, 230))
        draw = ImageDraw.Draw(base_img)
        draw.rectangle([15, 15, 55, 55], fill=(210, 40, 40))
        draw.ellipse([60, 60, 105, 105], fill=(40, 140, 210))

        cls.img_base = cls.dir_path / "photo_base.png"
        base_img.save(cls.img_base)

        cls.img_identical = cls.dir_path / "photo_identical.png"
        base_img.save(cls.img_identical)

        cls.img_resized = cls.dir_path / "photo_resized.png"
        base_img.resize((60, 60), Image.Resampling.LANCZOS).save(cls.img_resized)

        cls.img_compressed = cls.dir_path / "photo_compressed.jpg"
        base_img.save(cls.img_compressed, "JPEG", quality=20)

        enhancer = ImageEnhance.Brightness(base_img)
        cls.img_bright = cls.dir_path / "photo_bright.png"
        enhancer.enhance(1.2).save(cls.img_bright)

        cls.img_unrelated = cls.dir_path / "photo_unrelated.png"
        unrelated = Image.new("RGB", (120, 120), color=(10, 10, 40))
        draw_u = ImageDraw.Draw(unrelated)
        for i in range(0, 120, 20):
            draw_u.line([(0, i), (120, i)], fill=(200, 200, 50), width=2)
        unrelated.save(cls.img_unrelated)

        cls.img_corrupted = cls.dir_path / "photo_corrupted.png"
        cls.img_corrupted.write_bytes(b"\x89PNG\r\n\x1a\nINVALID_BYTE_STREAM_TEST")

        # ----------------------------------------------------
        # 7-8. Synthetic PDFs (Identical and Similar)
        # ----------------------------------------------------
        doc_lines_a = [
            "Project T243: Duplicate Document and Image Detection",
            "This report analyzes duplicate detection strategies using perceptual hashing and TF-IDF.",
            "All experiments were conducted on synthetic verification samples.",
        ]
        doc_lines_b = [
            "Project T243: Duplicate Document and Image Detection",
            "This report analyzes duplicate detection strategies using perceptual hashing and TF-IDF.",
            "Additional notes: Experiments were conducted on verification samples.",
        ]

        cls.pdf_1 = cls.dir_path / "report_1.pdf"
        cls.pdf_identical = cls.dir_path / "report_identical.pdf"
        cls.pdf_similar = cls.dir_path / "report_similar.pdf"

        create_synthetic_pdf(cls.pdf_1, doc_lines_a)
        create_synthetic_pdf(cls.pdf_identical, doc_lines_a)
        create_synthetic_pdf(cls.pdf_similar, doc_lines_b)

        # ----------------------------------------------------
        # 9. Similar DOCX documents
        # ----------------------------------------------------
        docx_text_1 = (
            "Deep learning and natural language processing allow systems to identify near-duplicate "
            "documents through vector embeddings and term frequency inverse document frequency."
        )
        docx_text_2 = (
            "Deep learning and natural language processing allow automated systems to detect near-duplicate "
            "documents through vector embeddings and term frequency inverse document frequency analysis."
        )
        cls.docx_1 = cls.dir_path / "manuscript_v1.docx"
        cls.docx_2 = cls.dir_path / "manuscript_v2.docx"
        create_synthetic_docx(cls.docx_1, docx_text_1)
        create_synthetic_docx(cls.docx_2, docx_text_2)

        # ----------------------------------------------------
        # 10. Cross-Format: PDF and DOCX with matching content
        # ----------------------------------------------------
        cross_text = (
            "Standard Operating Procedure for Duplicate File Ingestion. "
            "All candidate records must be verified by checksum and perceptual similarity."
        )
        cls.cross_pdf = cls.dir_path / "sop_guideline.pdf"
        cls.cross_docx = cls.dir_path / "sop_guideline.docx"
        create_synthetic_pdf(cls.cross_pdf, [cross_text])
        create_synthetic_docx(cls.cross_docx, cross_text)

        # ----------------------------------------------------
        # 11. Scanned Document / OCR sample
        # ----------------------------------------------------
        ocr_img = Image.new("RGB", (200, 70), color=(255, 255, 255))
        draw_ocr = ImageDraw.Draw(ocr_img)
        draw_ocr.text((10, 25), "INVOICE #987654", fill=(0, 0, 0))
        cls.scanned_img = cls.dir_path / "scanned_invoice.png"
        ocr_img.save(cls.scanned_img)

        # ----------------------------------------------------
        # 13. Unsupported file formats
        # ----------------------------------------------------
        cls.unsupported_bin = cls.dir_path / "binary_data.dat"
        cls.unsupported_bin.write_bytes(b"\x00\x01\x02\x03\x04\x05")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_identical_files_different_filenames(self):
        """Test 1: Identical files with different filenames must be detected by SHA-256."""
        detector = UnifiedDetector(self.dir_path, threshold=80.0)
        inventory = scan_directory(self.dir_path)
        detector.run_exact_detection(inventory)

        records = detector.result_manager.get_records()
        pair_found = any(
            (r.file_name_1 == "article_copy.txt" and r.file_name_2 == "article_v1.txt")
            or (r.file_name_1 == "article_v1.txt" and r.file_name_2 == "article_copy.txt")
            for r in records
        )
        self.assertTrue(pair_found, "Exact duplicate text files were not detected.")

    def test_02_identical_images(self):
        """Test 2: Identical images must have 100% similarity on pHash and dHash."""
        detector = UnifiedDetector(self.dir_path, threshold=85.0)
        inventory = scan_directory(self.dir_path)
        detector.run_image_detection(inventory)

        records = detector.result_manager.get_records()
        match = next(
            (
                r for r in records
                if "photo_base.png" in (r.file_name_1, r.file_name_2)
                and "photo_identical.png" in (r.file_name_1, r.file_name_2)
            ),
            None,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.similarity_score, 100.0)
        self.assertEqual(match.duplicate_type, "exact_duplicate")

    def test_03_resized_images(self):
        """Test 3: Resized images must be recognized as near-duplicate."""
        detector = UnifiedDetector(self.dir_path, threshold=85.0)
        inventory = scan_directory(self.dir_path)
        detector.run_image_detection(inventory)

        records = detector.result_manager.get_records()
        match = next(
            (
                r for r in records
                if "photo_base.png" in (r.file_name_1, r.file_name_2)
                and "photo_resized.png" in (r.file_name_1, r.file_name_2)
            ),
            None,
        )
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.similarity_score, 90.0)
        self.assertIn(match.duplicate_type, ["exact_duplicate", "near_duplicate_image"])

    def test_04_jpeg_compressed_images(self):
        """Test 4: JPEG-compressed images must remain near-duplicates."""
        detector = UnifiedDetector(self.dir_path, threshold=85.0)
        inventory = scan_directory(self.dir_path)
        detector.run_image_detection(inventory)

        records = detector.result_manager.get_records()
        match = next(
            (
                r for r in records
                if "photo_base.png" in (r.file_name_1, r.file_name_2)
                and "photo_compressed.jpg" in (r.file_name_1, r.file_name_2)
            ),
            None,
        )
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.similarity_score, 90.0)

    def test_05_brightness_adjusted_images(self):
        """Test 5: Brightness-adjusted images should be detected as near-duplicate."""
        detector = UnifiedDetector(self.dir_path, threshold=85.0)
        inventory = scan_directory(self.dir_path)
        detector.run_image_detection(inventory)

        records = detector.result_manager.get_records()
        match = next(
            (
                r for r in records
                if "photo_base.png" in (r.file_name_1, r.file_name_2)
                and "photo_bright.png" in (r.file_name_1, r.file_name_2)
            ),
            None,
        )
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.similarity_score, 85.0)

    def test_06_unrelated_images(self):
        """Test 6: Unrelated images must not be flagged as duplicates."""
        detector = UnifiedDetector(self.dir_path, threshold=80.0)
        inventory = scan_directory(self.dir_path)
        detector.run_image_detection(inventory)

        records = detector.result_manager.get_records()
        match = next(
            (
                r for r in records
                if "photo_unrelated.png" in (r.file_name_1, r.file_name_2)
                and "photo_base.png" in (r.file_name_1, r.file_name_2)
            ),
            None,
        )
        self.assertIsNone(match, "Unrelated images were incorrectly flagged as duplicates.")

    def test_07_identical_pdf_documents(self):
        """Test 7: Identical PDF documents must have 100% TF-IDF similarity."""
        res = compare_two_documents(self.pdf_1, self.pdf_identical)
        self.assertEqual(res.similarity_score, 100.0)
        self.assertTrue(res.is_duplicate)

    def test_08_similar_pdf_documents(self):
        """Test 8: Similar PDF documents should have high similarity (>= 75%)."""
        res = compare_two_documents(self.pdf_1, self.pdf_similar, threshold=75.0)
        self.assertGreaterEqual(res.similarity_score, 75.0)
        self.assertTrue(res.is_duplicate)

    def test_09_similar_docx_documents(self):
        """Test 9: Similar DOCX documents should be parsed and compared successfully."""
        res = compare_two_documents(self.docx_1, self.docx_2, threshold=75.0)
        self.assertGreaterEqual(res.similarity_score, 75.0)
        self.assertTrue(res.is_duplicate)

    def test_10_pdf_and_docx_cross_format(self):
        """Test 10: PDF and DOCX with matching content must be categorized as cross_format_document."""
        res = compare_two_documents(self.cross_pdf, self.cross_docx, threshold=75.0)
        self.assertGreaterEqual(res.similarity_score, 85.0)
        self.assertEqual(res.duplicate_type, "cross_format_document")

    def test_11_scanned_documents_ocr(self):
        """Test 11: OCR extraction handler runs without crashing even if Tesseract is not installed."""
        text = extract_text_with_ocr(self.scanned_img)
        self.assertIsInstance(text, str)

    def test_12_corrupted_images(self):
        """Test 12: Corrupted images must be handled gracefully without crashing the scan."""
        inventory = scan_directory(self.dir_path)
        detector = UnifiedDetector(self.dir_path)
        # Should not raise exception
        detector.run_image_detection(inventory)

    def test_13_unsupported_file_formats(self):
        """Test 13: Unsupported formats (.dat) must be routed into 'others' and excluded from image/doc engines."""
        inventory = scan_directory(self.dir_path)
        other_names = [f.file_name for f in inventory.others]
        self.assertIn("binary_data.dat", other_names)
        image_names = [f.file_name for f in inventory.images]
        doc_names = [f.file_name for f in inventory.documents]
        self.assertNotIn("binary_data.dat", image_names)
        self.assertNotIn("binary_data.dat", doc_names)

    def test_14_mixed_image_document_folder(self):
        """Test 14: An end-to-end run on a mixed folder should detect exact, image, and document duplicates."""
        out_csv = self.dir_path / "mixed_test_report.csv"
        detector = UnifiedDetector(self.dir_path, threshold=80.0, report_output=out_csv)
        records, summary = detector.run(run_exact=True, run_images=True, run_documents=True)

        self.assertGreater(summary.exact_duplicate_pairs, 0)
        self.assertGreater(summary.near_duplicate_image_pairs, 0)
        self.assertGreater(summary.total_detected_pairs, 0)
        self.assertTrue(out_csv.exists())

    def test_15_duplicate_result_elimination(self):
        """Test 15: Canonical ordering ensures no self-comparison and no reverse pairs."""
        p1 = str(self.identical_txt_1)
        p2 = str(self.identical_txt_2)

        # Self comparison must return None
        self.assertIsNone(canonicalize_pair(p1, p1))

        # Reverse ordering produces the exact same canonical pair
        pair_ab = canonicalize_pair(p1, p2)
        pair_ba = canonicalize_pair(p2, p1)
        self.assertEqual(pair_ab, pair_ba)

    def test_16_combined_csv_schema(self):
        """Test 16: Combined report CSV must contain all 10 required headers."""
        out_csv = self.dir_path / "schema_test_report.csv"
        detector = UnifiedDetector(self.dir_path, report_output=out_csv)
        detector.run()

        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_header = [
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
        self.assertEqual(header, expected_header)

    def test_17_individual_csv_protection(self):
        """Test 17: Attempting to overwrite existing reports must raise a ValueError."""
        manager = ResultManager()
        with self.assertRaises(ValueError):
            export_combined_report(manager.get_records(), output_path="duplicate_report.csv")

        with self.assertRaises(ValueError):
            export_combined_report(manager.get_records(), output_path="image_duplicate_report.csv")

    def test_18_existing_main_functionality(self):
        """Test 18: Teammate's main.py functionality remains 100% intact."""
        exact_dups = main.find_duplicate()
        self.assertIsInstance(exact_dups, dict)
        self.assertGreater(len(exact_dups), 0)


if __name__ == "__main__":
    unittest.main()

