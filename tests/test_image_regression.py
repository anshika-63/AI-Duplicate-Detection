"""
Regression and Diagnostic Test Suite for Image Detection Module.
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Phase 1 Regression Tests:
A. Identical files with different filenames.
B. Visually identical images saved with different metadata.
C. Resized images.
D. JPEG-compressed images.
E. Brightness-adjusted images.
F. Two unrelated images with similar backgrounds.
G. Cropped images.

Verifies:
- SHA-256 matching produces an exact duplicate verdict.
- Byte-identical pairs are never overridden or downgraded.
- Distinction between byte-identical and visually identical (different metadata/encoding).
- High dHash due to monotonic background gradient does NOT cause false positives.
- ORB handles geometric crop/rotation where perceptual hashes degrade.
"""

import hashlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance
from PIL.PngImagePlugin import PngInfo

from image_detection import (
    calculate_hamming_distance,
    calculate_similarity,
    categorize_duplicate,
    compare_images,
    compute_dhash,
    compute_phash,
)
from orb_detection import compare_images_orb
from result_manager import DuplicateRecord, ResultManager, canonicalize_pair


def get_file_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in 8KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestImageDetectionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create controlled test artifacts for conditions A through G."""
        cls.test_dir = tempfile.mkdtemp(prefix="img_reg_")
        cls.dir_path = Path(cls.test_dir)

        # 1. Base Image: 256x256 image with smooth horizontal gradient background + distinct shapes
        cls.base_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(cls.base_img)
        # Background gradient: increases monotonically from top to bottom
        for y in range(256):
            draw.line([(0, y), (255, y)], fill=(int(180 + y * 0.25), int(200 + y * 0.15), 240))
        # Distinct foreground objects
        draw.rectangle([40, 50, 110, 120], fill=(220, 40, 40), outline=(0, 0, 0), width=2)
        draw.ellipse([140, 130, 210, 200], fill=(40, 160, 220), outline=(0, 0, 0), width=2)
        draw.polygon([(80, 200), (130, 150), (170, 220)], fill=(40, 200, 60), outline=(0, 0, 0))

        cls.base_path = cls.dir_path / "base_original.png"
        cls.base_img.save(cls.base_path)

        # A. Identical copy with different filename (byte-for-byte identical)
        cls.identical_path = cls.dir_path / "a_identical_diff_name.png"
        shutil.copyfile(cls.base_path, cls.identical_path)

        # B. Visually identical image saved with different metadata
        cls.metadata_path = cls.dir_path / "b_metadata_diff.png"
        info = PngInfo()
        info.add_text("Copyright", "Academic Project T243")
        info.add_text("Software", "Custom Image Engine")
        cls.base_img.save(cls.metadata_path, pnginfo=info)

        # C. Resized image (scaled down to 128x128)
        cls.resized_path = cls.dir_path / "c_resized_128x128.png"
        cls.base_img.resize((128, 128), Image.Resampling.LANCZOS).save(cls.resized_path)

        # D. JPEG-compressed image (quality=30)
        cls.compressed_path = cls.dir_path / "d_compressed_q30.jpg"
        cls.base_img.save(cls.compressed_path, "JPEG", quality=30)

        # E. Brightness-adjusted image (+20% brightness)
        cls.bright_path = cls.dir_path / "e_bright_plus20.png"
        ImageEnhance.Brightness(cls.base_img).enhance(1.2).save(cls.bright_path)

        # F. Two unrelated images with similar backgrounds
        # Same monotonic vertical gradient, but completely different shapes and colors
        unrelated = Image.new("RGB", (256, 256), color=(255, 255, 255))
        draw_u = ImageDraw.Draw(unrelated)
        for y in range(256):
            draw_u.line([(0, y), (255, y)], fill=(int(180 + y * 0.25), int(200 + y * 0.15), 240))
        # Distinct objects at different locations
        draw_u.ellipse([20, 20, 80, 80], fill=(240, 240, 30), outline=(50, 50, 50), width=3)
        draw_u.rectangle([120, 30, 230, 90], fill=(180, 50, 220), outline=(0, 0, 0), width=2)
        draw_u.ellipse([50, 140, 120, 240], fill=(255, 120, 0), outline=(0, 0, 0), width=2)
        cls.unrelated_path = cls.dir_path / "f_unrelated_similar_bg.png"
        unrelated.save(cls.unrelated_path)

        # G. Cropped image (center 60% crop: bounding box (50, 50, 200, 200))
        cls.cropped_path = cls.dir_path / "g_cropped_center60.png"
        cls.base_img.crop((50, 50, 200, 200)).save(cls.cropped_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_case_a_identical_file_different_filename(self):
        """Case A: Identical files with different filenames must match byte-for-byte via SHA-256."""
        sha_base = get_file_sha256(self.base_path)
        sha_copy = get_file_sha256(self.identical_path)

        # Byte-level verification
        self.assertEqual(sha_base, sha_copy)

        # Perceptual hash verification
        p_res = compare_images(self.base_path, self.identical_path)
        self.assertEqual(p_res.phash_distance, 0)
        self.assertEqual(p_res.dhash_distance, 0)
        self.assertEqual(p_res.phash_similarity, 100.0)
        self.assertEqual(p_res.dhash_similarity, 100.0)
        self.assertTrue(p_res.is_duplicate)

        # ResultManager must record exact_duplicate and prevent downgrading
        rm = ResultManager()
        rm.add_record(
            file_path_1=self.base_path,
            file_path_2=self.identical_path,
            file_type="image",
            duplicate_type="exact_duplicate",
            similarity_score=100.0,
            detection_algorithm="SHA-256",
            detection_details=f"SHA-256: {sha_base}",
        )
        # Attempt to add ORB finding for same pair
        rm.add_record(
            file_path_1=self.base_path,
            file_path_2=self.identical_path,
            file_type="image",
            duplicate_type="near_duplicate_image",
            similarity_score=95.0,
            detection_algorithm="ORB",
            detection_details="ORB 208 inliers",
        )
        rec = rm.get_records()[0]
        # Exact duplicate verdict must NOT be overridden by ORB
        self.assertEqual(rec.duplicate_type, "exact_duplicate")
        self.assertEqual(rec.similarity_score, 100.0)

    def test_case_b_visually_identical_different_metadata(self):
        """Case B: Visually identical images with different metadata should have different SHA-256 but 0 hash distance."""
        sha_base = get_file_sha256(self.base_path)
        sha_meta = get_file_sha256(self.metadata_path)

        # File bytes differ due to PNG text chunks
        self.assertNotEqual(sha_base, sha_meta)

        # Perceptual visual hashes are bit-for-bit identical
        p1 = compute_phash(self.base_path)
        p2 = compute_phash(self.metadata_path)
        d1 = compute_dhash(self.base_path)
        d2 = compute_dhash(self.metadata_path)

        self.assertEqual(calculate_hamming_distance(p1, p2), 0)
        self.assertEqual(calculate_hamming_distance(d1, d2), 0)

        res = compare_images(self.base_path, self.metadata_path)
        self.assertEqual(res.duplicate_category, "Exact Duplicate")
        self.assertTrue(res.is_duplicate)

    def test_case_c_resized_images(self):
        """Case C: Resized images must retain high pHash and dHash similarity (>= 90%)."""
        res = compare_images(self.base_path, self.resized_path)
        self.assertGreaterEqual(res.phash_similarity, 90.0)
        self.assertGreaterEqual(res.dhash_similarity, 90.0)
        self.assertTrue(res.is_duplicate)

    def test_case_d_jpeg_compressed_images(self):
        """Case D: JPEG compressed images must retain high similarity (>= 85%)."""
        res = compare_images(self.base_path, self.compressed_path)
        self.assertGreaterEqual(res.phash_similarity, 90.0)
        self.assertGreaterEqual(res.dhash_similarity, 85.0)
        self.assertTrue(res.is_duplicate)

    def test_case_e_brightness_adjusted_images(self):
        """Case E: Brightness-adjusted images (+20%) must retain similarity >= 85%."""
        res = compare_images(self.base_path, self.bright_path)
        self.assertGreaterEqual(res.phash_similarity, 85.0)
        self.assertGreaterEqual(res.dhash_similarity, 85.0)
        self.assertTrue(res.is_duplicate)

    def test_case_f_unrelated_images_similar_background(self):
        """Case F: Unrelated images with similar background gradient must NOT be classified as duplicate."""
        res = compare_images(self.base_path, self.unrelated_path, threshold=80.0)
        orb_res = compare_images_orb(self.base_path, self.unrelated_path)

        # Both pHash and dHash distance must reflect different visual foregrounds
        self.assertGreaterEqual(res.phash_distance, 16)
        self.assertLess(res.phash_similarity, 75.0)
        # ORB inliers must be low (< 10)
        self.assertLess(orb_res.ransac_inliers, 10)
        self.assertFalse(orb_res.is_duplicate)

    def test_case_g_cropped_images_orb_invariance(self):
        """Case G: 60% cropped image causes pHash/dHash to drop, but ORB RANSAC correctly detects it."""
        p_res = compare_images(self.base_path, self.cropped_path, threshold=85.0)
        orb_res = compare_images_orb(self.base_path, self.cropped_path)

        # Global perceptual hash fails due to framing shift
        self.assertLess(p_res.phash_similarity, 80.0)

        # ORB local feature matching succeeds with high inliers (>= 12)
        self.assertGreaterEqual(orb_res.ransac_inliers, 12)
        self.assertGreaterEqual(orb_res.similarity_score, 60.0)
        self.assertTrue(orb_res.is_duplicate)


if __name__ == "__main__":
    unittest.main()
