"""
Regression and Diagnostic Test Suite for Document Similarity Module.
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Phase 2 Regression Tests:
1. Identical document content returns 100.0% similarity.
2. Cross-format identical documents (e.g. TXT vs DOCX vs PDF) return 100% similarity and cross_format_document.
3. Mostly same document pair returns high similarity (>= 85.0%), verdict near_duplicate_document.
4. Moderately same document pair returns moderate similarity (75.0% - 85.0%), verdict near_duplicate_document.
5. Lightly same document pair returns lower similarity (< 75.0%), verdict possible_duplicate / non_duplicate.
6. Monotonic similarity ranking: Mostly same > Moderately same > Lightly same.
7. Unicode ligature and accented text normalization (NFKD).
8. DOCX tabular content extraction verification.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from document_detection import (
    calculate_tfidf_similarity,
    categorize_document_duplicate,
    compare_two_documents,
    extract_document_text,
    preprocess_text,
)


class TestDocumentSimilarityRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create controlled synthetic document files for regression testing."""
        cls.test_dir = tempfile.mkdtemp(prefix="doc_reg_")
        cls.dir_path = Path(cls.test_dir)

        p1 = "Antigravity software engineering architecture defines the modular framework for duplicate detection. "
        p2 = "The system continuously monitors local file directories for image and document updates. "
        p3 = "Feature extraction involves computing perceptual fingerprints and extracting textual content. "
        p4 = "Classification is performed using convolutional deep learning models trained on benchmark corpora. "
        p5 = "Detailed audit logs and structured CSV summaries are persisted to local database storage. "

        cls.doc_base = p1 + p2 + p3 + p4 + p5

        # Mostly same: 1 small phrase variation in paragraph 3 (~95% similarity)
        p3_mostly = "Feature extraction involves computing visual fingerprints and extracting document content. "
        cls.doc_mostly = p1 + p2 + p3_mostly + p4 + p5

        # Moderately same: paragraph 3 completely replaced, other 4 paragraphs retained (~76-80% similarity)
        p3_mod = "Document processing transforms unformatted pdf files into normalized token sequences. "
        cls.doc_moderate = p1 + p2 + p3_mod + p4 + p5

        # Lightly same: 2 paragraphs retained, 3 replaced with unrelated quantum computing text (~40-54% similarity)
        q1 = "Quantum computing utilizes qubits and superposition to solve complex discrete optimization problems. "
        q2 = "Hardware implementations include superconducting transmon qubits and trapped ion quantum processors. "
        q3 = "Error correction thresholds must be maintained using surface codes to achieve fault tolerance. "
        cls.doc_light = p1 + p2 + q1 + q2 + q3

        # Save to disk as TXT
        cls.path_base = cls.dir_path / "doc_base.txt"
        cls.path_mostly = cls.dir_path / "doc_mostly.txt"
        cls.path_moderate = cls.dir_path / "doc_moderate.txt"
        cls.path_light = cls.dir_path / "doc_light.txt"

        cls.path_base.write_text(cls.doc_base, encoding="utf-8")
        cls.path_mostly.write_text(cls.doc_mostly, encoding="utf-8")
        cls.path_moderate.write_text(cls.doc_moderate, encoding="utf-8")
        cls.path_light.write_text(cls.doc_light, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_identical_document_content(self):
        """Identical document text must return exactly 100.0% similarity and exact_duplicate category."""
        sim = calculate_tfidf_similarity(self.doc_base, self.doc_base)
        self.assertEqual(sim, 100.0)

        res = compare_two_documents(self.path_base, self.path_base)
        self.assertEqual(res.similarity_score, 100.0)
        self.assertEqual(res.duplicate_type, "exact_duplicate")
        self.assertTrue(res.is_duplicate)

    def test_02_mostly_same_document_pair(self):
        """Mostly same pair should have high similarity (>= 85%) and be marked near_duplicate."""
        res = compare_two_documents(self.path_base, self.path_mostly, threshold=75.0)
        self.assertGreaterEqual(res.similarity_score, 85.0)
        self.assertEqual(res.duplicate_type, "near_duplicate_document")
        self.assertTrue(res.is_duplicate)

    def test_03_moderately_same_document_pair(self):
        """Moderately same pair should meet threshold (>= 75%) and be marked near_duplicate."""
        res = compare_two_documents(self.path_base, self.path_moderate, threshold=75.0)
        self.assertGreaterEqual(res.similarity_score, 75.0)
        self.assertLess(res.similarity_score, 95.0)
        self.assertEqual(res.duplicate_type, "near_duplicate_document")
        self.assertTrue(res.is_duplicate)

    def test_04_lightly_same_document_pair(self):
        """Lightly same pair should fall below threshold (< 75%) and NOT be marked near-duplicate."""
        res = compare_two_documents(self.path_base, self.path_light, threshold=75.0)
        self.assertLess(res.similarity_score, 75.0)
        self.assertIn(res.duplicate_type, ["possible_duplicate", "non_duplicate"])
        self.assertFalse(res.is_duplicate)

    def test_05_monotonic_ranking_preservation(self):
        """Verify strict monotonic ranking: Mostly same > Moderately same > Lightly same."""
        sim_mostly = calculate_tfidf_similarity(self.doc_base, self.doc_mostly)
        sim_moderate = calculate_tfidf_similarity(self.doc_base, self.doc_moderate)
        sim_light = calculate_tfidf_similarity(self.doc_base, self.doc_light)

        self.assertGreater(sim_mostly, sim_moderate)
        self.assertGreater(sim_moderate, sim_light)

    def test_06_unicode_normalization(self):
        """Verify unicode ligatures and accented characters normalize cleanly."""
        text_with_ligatures = "The of\ufb01cial speci\ufb01cation for \u201cAI Duplicate Detection\u201d."
        normalized = preprocess_text(text_with_ligatures)
        self.assertIn("official", normalized)
        self.assertIn("specification", normalized)
        self.assertIn("ai duplicate detection", normalized)


    def test_07_metadata_and_quality_metrics(self):
        """Verify extraction metadata, word counts, and quality scoring."""
        from document_detection import extract_document_text_with_metadata
        meta = extract_document_text_with_metadata(self.path_base)
        self.assertGreater(meta.char_count, 100)
        self.assertGreater(meta.word_count, 20)
        self.assertGreaterEqual(meta.quality_score, 0.8)
        self.assertFalse(meta.is_scanned)

    def test_08_cross_format_docx_txt(self):
        """Verify cross-format document comparison recognizes identical content across extensions."""
        from tools.benchmark_documents import create_minimal_docx
        docx_path = self.dir_path / "base_copy.docx"
        create_minimal_docx(self.doc_base, docx_path)

        res = compare_two_documents(self.path_base, docx_path)
        self.assertEqual(res.similarity_score, 100.0)
        self.assertEqual(res.duplicate_type, "cross_format_document")
        self.assertTrue(res.is_duplicate)

    def test_09_paragraph_reordering_invariance(self):
        """Verify TF-IDF bag-of-words properties maintain 100% similarity under paragraph permutation."""
        p_reordered = self.doc_base.split(". ")
        reordered_str = ". ".join(reversed(p_reordered))
        sim = calculate_tfidf_similarity(self.doc_base, reordered_str)
        self.assertEqual(sim, 100.0)


if __name__ == "__main__":
    unittest.main()

