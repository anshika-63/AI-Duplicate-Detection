"""
Unit and Integration Tests for Image Duplicate Detection Module.
Tests all requirements of Phase 4:
1. Identical images.
2. Resized images.
3. JPEG-compressed images.
4. Brightness-adjusted images.
5. Unrelated images.
6. Corrupted images.
Plus folder scanning, pair generation, and CSV report export.
"""

import csv
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from image_detection import (
    calculate_hamming_distance,
    calculate_similarity,
    categorize_duplicate,
    compare_images,
    compute_dhash,
    compute_phash,
    export_image_report,
    find_duplicate_pairs,
    is_supported_image,
    load_image,
    scan_image_folder,
)


class TestImageDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with synthetic test images."""
        cls.test_dir = tempfile.mkdtemp(prefix="img_test_")
        cls.dir_path = Path(cls.test_dir)

        # 1. Base image: 128x128 RGB image with distinct geometric patterns
        base_img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(base_img)
        draw.rectangle([20, 20, 60, 60], fill=(200, 50, 50), outline=(0, 0, 0))
        draw.ellipse([70, 70, 110, 110], fill=(50, 150, 200), outline=(0, 0, 0))
        draw.line([10, 118, 118, 10], fill=(50, 200, 50), width=3)

        cls.base_path = cls.dir_path / "base.png"
        base_img.save(cls.base_path)

        # 1. Identical copy
        cls.identical_path = cls.dir_path / "identical.png"
        base_img.save(cls.identical_path)

        # 2. Resized copy (scaled down to 64x64, then saved as PNG)
        resized_img = base_img.resize((64, 64), Image.Resampling.LANCZOS)
        cls.resized_path = cls.dir_path / "resized.png"
        resized_img.save(cls.resized_path)

        # 3. JPEG compressed copy (quality = 25)
        cls.jpeg_compressed_path = cls.dir_path / "compressed.jpg"
        base_img.save(cls.jpeg_compressed_path, "JPEG", quality=25)

        # 4. Brightness adjusted copy (enhanced by factor 1.25)
        enhancer = ImageEnhance.Brightness(base_img)
        bright_img = enhancer.enhance(1.25)
        cls.bright_path = cls.dir_path / "bright.png"
        bright_img.save(cls.bright_path)

        # 5. Completely unrelated image (checkerboard / different colors)
        unrelated_img = Image.new("RGB", (128, 128), color=(10, 10, 40))
        draw_u = ImageDraw.Draw(unrelated_img)
        for i in range(0, 128, 16):
            for j in range(0, 128, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    draw_u.rectangle([i, j, i + 16, j + 16], fill=(220, 220, 60))
        cls.unrelated_path = cls.dir_path / "unrelated.png"
        unrelated_img.save(cls.unrelated_path)

        # 6. Corrupted image file (broken bytes)
        cls.corrupted_path = cls.dir_path / "corrupted.png"
        with open(cls.corrupted_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nCORRUPTED_GARBAGE_PAYLOAD_1234567890")

        # 7. Unsupported text file
        cls.unsupported_path = cls.dir_path / "notes.txt"
        with open(cls.unsupported_path, "w", encoding="utf-8") as f:
            f.write("This is a plain text file, not an image.")

    @classmethod
    def tearDownClass(cls):
        """Remove temporary directory."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_identical_images(self):
        """Test 1: Identical images should have 0 distance and 100% similarity."""
        result = compare_images(self.base_path, self.identical_path)
        self.assertEqual(result.phash_distance, 0)
        self.assertEqual(result.dhash_distance, 0)
        self.assertEqual(result.phash_similarity, 100.0)
        self.assertEqual(result.dhash_similarity, 100.0)
        self.assertEqual(result.duplicate_category, "Exact Duplicate")
        self.assertTrue(result.is_duplicate)

    def test_02_resized_images(self):
        """Test 2: Resized images should retain high similarity (near duplicate)."""
        result = compare_images(self.base_path, self.resized_path)
        self.assertLessEqual(result.phash_distance, 4)
        self.assertLessEqual(result.dhash_distance, 4)
        self.assertGreaterEqual(result.phash_similarity, 90.0)
        self.assertGreaterEqual(result.dhash_similarity, 90.0)
        self.assertIn(result.duplicate_category, ["Exact Duplicate", "Near Duplicate"])
        self.assertTrue(result.is_duplicate)

    def test_03_jpeg_compressed_images(self):
        """Test 3: JPEG compressed images should remain near-duplicates."""
        result = compare_images(self.base_path, self.jpeg_compressed_path)
        self.assertLessEqual(result.phash_distance, 6)
        self.assertLessEqual(result.dhash_distance, 6)
        self.assertGreaterEqual(result.phash_similarity, 90.0)
        self.assertGreaterEqual(result.dhash_similarity, 90.0)
        self.assertIn(result.duplicate_category, ["Exact Duplicate", "Near Duplicate"])
        self.assertTrue(result.is_duplicate)

    def test_04_brightness_adjusted_images(self):
        """Test 4: Moderate brightness changes should be detected as near-duplicate."""
        result = compare_images(self.base_path, self.bright_path)
        self.assertLessEqual(result.phash_distance, 6)
        self.assertLessEqual(result.dhash_distance, 10)
        self.assertGreaterEqual(result.phash_similarity, 90.0)
        self.assertGreaterEqual(result.dhash_similarity, 85.0)
        self.assertIn(result.duplicate_category, ["Exact Duplicate", "Near Duplicate"])
        self.assertTrue(result.is_duplicate)

    def test_05_unrelated_images(self):
        """Test 5: Completely unrelated images should have low similarity."""
        result = compare_images(self.base_path, self.unrelated_path)
        self.assertGreater(result.phash_distance, 15)
        self.assertGreater(result.dhash_distance, 15)
        self.assertLess(result.phash_similarity, 75.0)
        self.assertLess(result.dhash_similarity, 75.0)
        self.assertEqual(result.duplicate_category, "Non-Duplicate")
        self.assertFalse(result.is_duplicate)

    def test_06_corrupted_images(self):
        """Test 6: Corrupted image should raise an error during load and be skipped by folder scanner."""
        with self.assertRaises(Exception):
            load_image(self.corrupted_path)

        records, failed = scan_image_folder(self.dir_path, recursive=False)
        failed_paths = [str(f[0]) for f in failed]
        self.assertIn(str(self.corrupted_path), failed_paths)

    def test_07_unsupported_format(self):
        """Unsupported extensions should not be treated as supported images."""
        self.assertFalse(is_supported_image(self.unsupported_path))
        with self.assertRaises(ValueError):
            load_image(self.unsupported_path)

    def test_08_similarity_formula(self):
        """Verify the formula: Similarity (%) = (1 - Hamming Distance / 64) * 100."""
        # 0 distance -> 100.0%
        self.assertEqual(calculate_similarity(0, total_bits=64), 100.0)
        # 32 distance -> 50.0%
        self.assertEqual(calculate_similarity(32, total_bits=64), 50.0)
        # 64 distance -> 0.0%
        self.assertEqual(calculate_similarity(64, total_bits=64), 0.0)
        # 6 distance -> (1 - 6/64) * 100 = 90.625 -> rounded 90.62%
        self.assertEqual(calculate_similarity(6, total_bits=64), 90.62)

    def test_09_folder_scanner_and_duplicate_pairs(self):
        """Verify recursive folder scanning avoids self-comparison and duplicate pairs."""
        records, failed = scan_image_folder(self.dir_path, recursive=False)
        # We created 5 valid images (base, identical, resized, compressed, bright, unrelated = 6 total valid)
        valid_filenames = {r.path.name for r in records}
        self.assertIn("base.png", valid_filenames)
        self.assertIn("identical.png", valid_filenames)
        self.assertIn("resized.png", valid_filenames)
        self.assertIn("compressed.jpg", valid_filenames)
        self.assertIn("bright.png", valid_filenames)
        self.assertIn("unrelated.png", valid_filenames)
        self.assertNotIn("notes.txt", valid_filenames)
        self.assertNotIn("corrupted.png", valid_filenames)

        # Pairwise comparison
        pairs = find_duplicate_pairs(records, threshold=85.0)
        seen_pairs = set()
        for p in pairs:
            # Check no self-comparison
            self.assertNotEqual(p.image_path_1, p.image_path_2)
            pair_tuple = tuple(sorted([p.image_path_1, p.image_path_2]))
            # Check no duplicate reverse pairs
            self.assertNotIn(pair_tuple, seen_pairs)
            seen_pairs.add(pair_tuple)

    def test_10_csv_report_export(self):
        """Verify CSV report format and ensure duplicate_report.csv cannot be overwritten."""
        records, _ = scan_image_folder(self.dir_path, recursive=False)
        pairs = find_duplicate_pairs(records, threshold=85.0)

        report_file = self.dir_path / "test_image_duplicate_report.csv"
        out_path = export_image_report(pairs, report_path=report_file)
        self.assertTrue(out_path.exists())

        # Validate headers and content
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = [
                "image_path_1",
                "image_path_2",
                "phash_distance",
                "phash_similarity",
                "dhash_distance",
                "dhash_similarity",
                "detection_algorithm",
                "duplicate_category",
            ]
            self.assertEqual(header, expected_header)

            rows = list(reader)
            self.assertEqual(len(rows), len(pairs))

        # Test safety guard against overwriting teammate's duplicate_report.csv
        with self.assertRaises(ValueError):
            export_image_report(pairs, report_path="duplicate_report.csv")


if __name__ == "__main__":
    unittest.main()
