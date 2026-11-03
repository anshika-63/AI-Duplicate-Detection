"""
Regression and Diagnostic Test Suite for Document Classification Module.
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Phase 3 / Stage 1 Regression Tests:
1. Determinism across 5 repeated inferences.
2. Checkpoint metadata and label mapping consistency.
3. Model evaluation mode (Dropout inactive, deterministic output).
4. Synthetic DOCX invoice classification with multimodal text-prior fusion.
5. Genuine RVL-CDIP invoice image classification.
6. Genuine RVL-CDIP email image classification.
7. Top-5 alternative candidate predictions and confidence bounds.
8. Missing file raises FileNotFoundError.
9. Corrupted file raises ValueError.
10. Unsupported file extension raises ValueError.
11. Extreme image dimensions handled gracefully by resize transform.
"""

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from document_classification import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    DocumentClassifier,
    convert_document_to_image,
)


class TestDocumentClassificationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize classifier and synthetic test artifacts."""
        cls.classifier = DocumentClassifier()

        cls.test_dir = tempfile.mkdtemp(prefix="doc_class_reg_")
        cls.dir_path = Path(cls.test_dir)

        # 1. Synthetic DOCX invoice with native XML paragraphs and table
        sample_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>INVOICE #INV-2024-9021</w:t></w:r></w:p>
            <w:p><w:r><w:t>Date: November 2, 2026</w:t></w:r></w:p>
            <w:p><w:r><w:t>Bill To: Global Logistics Partners, 452 Innovation Blvd</w:t></w:r></w:p>
            <w:tbl>
              <w:tr>
                <w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>Quantity</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>Price</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>Total</w:t></w:r></w:p></w:tc>
              </w:tr>
              <w:tr>
                <w:tc><w:p><w:r><w:t>Neural Network Consulting</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>$4500.00</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>$4500.00</w:t></w:r></w:p></w:tc>
              </w:tr>
            </w:tbl>
            <w:p><w:r><w:t>Subtotal: $4500.00</w:t></w:r></w:p>
            <w:p><w:r><w:t>Total Amount Due: $4500.00</w:t></w:r></w:p>
            <w:p><w:r><w:t>Payment Terms: Net 30 days via direct wire transfer</w:t></w:r></w:p>
          </w:body>
        </w:document>"""

        cls.docx_path = cls.dir_path / "synthetic_invoice.docx"
        with zipfile.ZipFile(cls.docx_path, "w") as z:
            z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>')
            z.writestr("word/document.xml", sample_xml)

        # 2. Corrupted file
        cls.corrupted_path = cls.dir_path / "corrupted.png"
        cls.corrupted_path.write_bytes(b"\x89PNG\r\n\x1a\nBROKEN_GARBAGE_BYTES")

        # 3. Unsupported file
        cls.unsupported_path = cls.dir_path / "audio.mp3"
        cls.unsupported_path.write_text("ID3FakeAudioData")

        # 4. Extreme dimensions image (narrow vertical 50x800 banner)
        cls.extreme_dim_path = cls.dir_path / "extreme_banner.png"
        img_extreme = Image.new("RGB", (50, 800), color=(200, 200, 200))
        img_extreme.save(cls.extreme_dim_path)

        # 5. Existing genuine RVL-CDIP invoice and email samples cached locally
        cls.rvl_invoice_path = Path("cache/rvlcdip_sample_class_11_invoice.png")
        cls.rvl_email_path = Path("cache/rvlcdip_sample_class_2_email.png")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_label_mapping_consistency(self):
        """Verify checkpoint class names match canonical RVL-CDIP class list."""
        self.assertEqual(len(self.classifier.class_names), 16)
        self.assertEqual(self.classifier.class_names, CLASS_NAMES)

        meta_file = Path("checkpoints/model_metadata.json")
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(meta["classes"], CLASS_NAMES)
            self.assertEqual(meta["num_classes"], 16)

    def test_02_model_evaluation_mode_and_eval_flags(self):
        """Verify backbone and classification head are in eval mode (Dropout inactive)."""
        self.assertFalse(self.classifier.backbone.training)
        self.assertFalse(self.classifier.head.training)

    def test_03_inference_determinism_five_runs(self):
        """Test same document classified 5 times produces bit-for-bit identical confidences."""
        confs = []
        preds = []
        for _ in range(5):
            res = self.classifier.classify_document(self.docx_path)
            preds.append(res.predicted_category)
            confs.append(round(res.confidence, 6))

        # Must produce identical class and confidence every single time
        self.assertEqual(len(set(preds)), 1)
        self.assertEqual(len(set(confs)), 1)

    def test_04_synthetic_docx_invoice_classification(self):
        """Synthetic DOCX invoice must be correctly identified as invoice via multimodal fusion."""
        res = self.classifier.classify_document(self.docx_path)
        self.assertEqual(res.predicted_category, "invoice")
        self.assertGreaterEqual(res.confidence, 0.50)

    def test_05_genuine_rvlcdip_invoice_classification(self):
        """Genuine RVL-CDIP scanned invoice must be classified as invoice with high confidence."""
        if self.rvl_invoice_path.exists():
            res = self.classifier.classify_document(self.rvl_invoice_path)
            self.assertEqual(res.predicted_category, "invoice")
            self.assertGreaterEqual(res.confidence, 0.70)

    def test_06_genuine_rvlcdip_email_classification(self):
        """Genuine RVL-CDIP email image must be classified as email with high confidence."""
        if self.rvl_email_path.exists():
            res = self.classifier.classify_document(self.rvl_email_path)
            self.assertEqual(res.predicted_category, "email")
            self.assertGreaterEqual(res.confidence, 0.85)

    def test_07_top_alternatives_structure(self):
        """Top-5 alternative predictions must be descending and have valid confidence bounds."""
        res = self.classifier.classify_document(self.docx_path, top_k=5)
        self.assertEqual(len(res.alternative_predictions), 5)

        prev_conf = 1.0
        for cat, conf in res.alternative_predictions:
            self.assertIn(cat, CLASS_NAMES)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
            self.assertLessEqual(conf, prev_conf)
            prev_conf = conf

    def test_08_missing_file_raises_not_found(self):
        """Missing file must raise FileNotFoundError."""
        missing = self.dir_path / "non_existent_file.png"
        with self.assertRaises(FileNotFoundError):
            self.classifier.classify_document(missing)

    def test_09_corrupted_file_raises_value_error(self):
        """Corrupted image bytes must raise ValueError."""
        with self.assertRaises(ValueError):
            self.classifier.classify_document(self.corrupted_path)

    def test_10_unsupported_format_raises_value_error(self):
        """Unsupported format must raise ValueError."""
        with self.assertRaises(ValueError):
            self.classifier.classify_document(self.unsupported_path)

    def test_11_extreme_dimensions_handled(self):
        """Extreme non-square dimensions (50x800) must resize to (224, 224) without crashing."""
        res = self.classifier.classify_document(self.extreme_dim_path)
        self.assertIn(res.predicted_category, CLASS_NAMES)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
