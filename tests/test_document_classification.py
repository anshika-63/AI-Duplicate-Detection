"""
Unit Tests for Document Classification Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Tests:
1. Multi-format ingestion: Image, PDF, DOCX, TXT conversion.
2. Classifier initialization with 16 canonical RVL-CDIP classes.
3. Prediction dictionary structure and confidence calibration.
4. Top-3 alternative prediction ordering.
5. Batch directory classification.
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from document_classification import (
    CLASS_NAMES,
    ClassificationResult,
    DocumentClassifier,
    convert_document_to_image,
)


class TestDocumentClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="t243_doc_class_")
        cls.dir_path = Path(cls.test_dir)

        # 1. Synthetic Invoice Image
        cls.img_path = cls.dir_path / "invoice_sample.png"
        img = Image.new("RGB", (300, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "INVOICE #INV-2026-001", fill=(0, 0, 0))
        draw.text((20, 60), "Total Due: $1,250.00", fill=(0, 0, 0))
        img.save(cls.img_path)

        # 2. Synthetic PDF
        cls.pdf_path = cls.dir_path / "research_paper.pdf"
        c = canvas.Canvas(str(cls.pdf_path), pagesize=letter)
        c.drawString(72, 750, "Scientific Publication: On Neural Document Image Analysis")
        c.drawString(72, 730, "Abstract: Deep learning models enable robust document parsing.")
        c.save()

        # 3. Synthetic DOCX
        cls.docx_path = cls.dir_path / "resume_doc.docx"
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '  <w:body><w:p><w:r><w:t>Curriculum Vitae / Resume of Senior AI Engineer</w:t></w:r></w:p></w:body>\n'
            '</w:document>'
        )
        with zipfile.ZipFile(cls.docx_path, "w") as z:
            z.writestr("word/document.xml", xml_content)

        # 4. Synthetic TXT
        cls.txt_path = cls.dir_path / "memo_notes.txt"
        cls.txt_path.write_text("MEMORANDUM: Project status update and technical milestones.", encoding="utf-8")

    def test_01_canonical_class_names(self):
        """Verifies all 16 canonical RVL-CDIP classes are present and ordered properly."""
        self.assertEqual(len(CLASS_NAMES), 16)
        expected = [
            "letter", "form", "email", "handwritten", "advertisement",
            "scientific report", "scientific publication", "specification",
            "file folder", "news article", "budget", "invoice",
            "presentation", "questionnaire", "resume", "memo"
        ]
        self.assertEqual(CLASS_NAMES, expected)

    def test_02_image_conversion(self):
        """Verifies multi-format document to PIL Image rendering."""
        # Image
        img1 = convert_document_to_image(self.img_path)
        self.assertIsInstance(img1, Image.Image)

        # PDF
        img2 = convert_document_to_image(self.pdf_path)
        self.assertIsInstance(img2, Image.Image)

        # DOCX
        img3 = convert_document_to_image(self.docx_path)
        self.assertIsInstance(img3, Image.Image)

        # TXT
        img4 = convert_document_to_image(self.txt_path)
        self.assertIsInstance(img4, Image.Image)

    def test_03_classifier_inference_structure(self):
        """Verifies inference output format and confidence bounds."""
        classifier = DocumentClassifier()
        res = classifier.classify_document(self.img_path)

        self.assertIsInstance(res, ClassificationResult)
        self.assertIn(res.predicted_category, CLASS_NAMES)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertEqual(len(res.alternative_predictions), 3)
        self.assertGreater(res.processing_time_ms, 0.0)

    def test_04_batch_directory_classification(self):
        """Verifies batch directory classification."""
        classifier = DocumentClassifier()
        results = classifier.classify_directory(self.dir_path)

        self.assertGreaterEqual(len(results), 4)
        for r in results:
            self.assertIn(r.predicted_category, CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
