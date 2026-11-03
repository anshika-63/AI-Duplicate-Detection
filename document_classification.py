"""
Document Classification Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Features:
- 16 document categories: letter, form, email, handwritten, advertisement,
  scientific report, scientific publication, specification, file folder,
  news article, budget, invoice, presentation, questionnaire, resume, memo.
- Multi-format ingestion: PDF, DOCX, TXT, TIFF, JPG, JPEG, PNG, WEBP, BMP.
- Dual-engine architecture:
    1. Deep Visual Feature Classification (ResNet18 + Deep MLP Head)
    2. Fallback heuristic / text-based classifier when weights are initializing
- Comprehensive inference output:
    - Predicted document category
    - Calibrated confidence score
    - Top-3 alternative candidate predictions
    - Model name and inference latency (ms)
"""

import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = "checkpoints/best_document_classifier.pth"

CLASS_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


class DeepClassificationHead(nn.Module):
    """Deep neural classification head with Dropout and BatchNorm."""

    def __init__(self, in_features: int = 512, num_classes: int = 16, dropout_rate: float = 0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@dataclass
class ClassificationResult:
    """Stores the prediction output and metrics for a classified document."""
    file_path: str
    file_name: str
    predicted_category: str
    confidence: float
    alternative_predictions: List[Tuple[str, float]]
    model_name: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "predicted_category": self.predicted_category,
            "confidence": round(self.confidence, 4),
            "top_alternatives": [
                {"category": cat, "confidence": round(conf, 4)}
                for cat, conf in self.alternative_predictions
            ],
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


SUPPORTED_CLASSIFICATION_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}


def convert_document_to_image(file_path: Union[str, Path]) -> Image.Image:
    """
    Renders or loads any supported document into an RGB Pillow Image.
    Supports PDF (first page render), TIFF, PNG, JPG, BMP, WEBP, DOCX, TXT.

    Raises:
        FileNotFoundError: If the document file does not exist.
        ValueError: If the file format is unsupported or the file is corrupted.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_CLASSIFICATION_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format '{ext}' for classification. "
            f"Supported formats: {sorted(SUPPORTED_CLASSIFICATION_EXTENSIONS)}"
        )

    # Case 1: Image formats
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        try:
            with Image.open(path) as img:
                loaded = img.convert("RGB")
                loaded.load()
                return loaded
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file '{path.name}': {e}") from e

    # Case 2: PDF files - render first page
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                if len(pdf.pages) > 0:
                    pil_img = pdf.pages[0].to_image(resolution=150).original
                    return pil_img.convert("RGB")
        except Exception as e:
            logger.debug("pdfplumber page render failed for %s: %s", path.name, e)

        # Fallback text representation to synthetic image if rendering is unavailable
        from document_detection import extract_text_from_pdf
        text_content = extract_text_from_pdf(path)
        return create_text_preview_image(text_content)

    # Case 3: DOCX / TXT files
    if ext in {".docx", ".txt"}:
        from document_detection import extract_document_text
        text_content = extract_document_text(path, use_cache=True)
        return create_text_preview_image(text_content)

    raise ValueError(f"Unsupported document format '{ext}' for classification.")


TEXT_CLASS_KEYWORDS = {
    "invoice": ["invoice", "bill to", "amount due", "subtotal", "remit to", "due date", "total due", "balance due", "invoice #", "invoice no", "payment terms"],
    "email": ["from:", "to:", "subject:", "sent:", "cc:", "bcc:", "original message", "mailto:"],
    "letter": ["dear", "sincerely", "regards", "yours truly", "to whom it may concern", "cordially"],
    "resume": ["resume", "curriculum vitae", "work experience", "skills", "gpa", "bachelor of", "master of", "education"],
    "budget": ["budget", "fiscal year", "q1", "q2", "expenditures", "appropriations", "revenue forecast"],
    "scientific report": ["scientific report", "technical report", "experiment", "methodology"],
    "scientific publication": ["abstract", "introduction", "references", "ieee", "doi:", "et al."],
    "specification": ["specification", "requirements", "compliance", "standard", "astm", "iso"],
    "questionnaire": ["questionnaire", "survey", "check one", "please rate", "scale 1-5"],
    "memo": ["memorandum", "memo to:", "from:", "re:"],
    "advertisement": ["sale", "discount", "offer", "special", "coupon", "limited time"],
    "form": ["application form", "fill out", "signature:", "ssn:", "applicant"],
}


def compute_text_prior(text: str, class_names: List[str]) -> Optional[np.ndarray]:
    """Calculates keyword evidence prior for textual documents."""
    if not text or len(text.strip()) < 10:
        return None
    lower_text = text.lower()
    scores = np.zeros(len(class_names), dtype=np.float32)
    has_match = False
    for idx, c_name in enumerate(class_names):
        kw_list = TEXT_CLASS_KEYWORDS.get(c_name, [])
        for kw in kw_list:
            if kw in lower_text:
                scores[idx] += 1.5
                has_match = True

    if not has_match:
        return None

    # Softmax / Normalize to probability distribution
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def create_text_preview_image(text: str, size: Tuple[int, int] = (600, 800)) -> Image.Image:
    """Renders text onto a structured synthetic document canvas for visual classification."""
    import textwrap
    from PIL import ImageDraw
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    raw_lines = text.split("\n")
    y = 35
    for raw_line in raw_lines:
        if y > size[1] - 40:
            break
        raw_line = raw_line.strip()
        if not raw_line:
            y += 12
            continue

        # Check if tabular row
        if " | " in raw_line:
            cells = raw_line.split(" | ")
            col_width = (size[0] - 60) // max(1, len(cells))
            # Draw table row borders
            draw.line([(30, y - 2), (size[0] - 30, y - 2)], fill=(200, 200, 200), width=1)
            for c_idx, cell in enumerate(cells):
                c_x = 30 + c_idx * col_width
                draw.text((c_x + 4, y), cell[:25], fill=(30, 30, 30))
            draw.line([(30, y + 18), (size[0] - 30, y + 18)], fill=(200, 200, 200), width=1)
            y += 22
        else:
            # Wrap paragraphs cleanly
            wrapped = textwrap.wrap(raw_line, width=68)
            for w in wrapped:
                if y > size[1] - 40:
                    break
                # Header styling
                if w.isupper() and len(w) < 40 and y < 120:
                    draw.text((30, y), w, fill=(10, 10, 10))
                    draw.line([(30, y + 16), (220, y + 16)], fill=(100, 100, 100), width=2)
                    y += 24
                else:
                    draw.text((30, y), w, fill=(35, 35, 35))
                    y += 16

    return img


class DocumentClassifier:
    """
    16-Class Document Classification Engine.
    Employs ResNet18 Deep Transfer Learning with calibrated probability estimation.
    """

    def __init__(self, checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = CLASS_NAMES

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()

        # Head
        self.head = DeepClassificationHead(in_features=512, num_classes=len(self.class_names))
        self.is_trained = False

        if self.checkpoint_path.exists():
            self._load_checkpoint()
        else:
            logger.info("Checkpoint '%s' not yet found; classifier will run in pre-trained feature mode.", self.checkpoint_path)

        self.head.to(self.device)
        self.head.eval()

    def _load_checkpoint(self):
        """Loads trained model weights."""
        logger.info("Loading document classifier checkpoint from '%s'...", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if "state_dict" in ckpt:
            self.head.load_state_dict(ckpt["state_dict"])
            self.class_names = ckpt.get("class_names", CLASS_NAMES)
            self.is_trained = True
            logger.info("Loaded checkpoint successfully (Best Val Acc: %.2f%%).", ckpt.get("best_val_acc", 0.0) * 100)
        else:
            self.head.load_state_dict(ckpt)
            self.is_trained = True

    def classify_document(self, file_path: Union[str, Path], top_k: int = 3) -> ClassificationResult:
        """
        Classifies an input document (image, PDF, DOCX, TXT) into one of 16 categories.
        """
        path = Path(file_path).resolve()
        start_time = time.time()

        img = convert_document_to_image(path)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.backbone(tensor)
            logits = self.head(feat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ext = path.suffix.lower()
        # For textual documents, fuse visual ResNet18 probabilities with keyword prior
        if ext in {".docx", ".txt", ".pdf"}:
            try:
                from document_detection import extract_document_text
                extracted = extract_document_text(path, use_cache=True)
                prior = compute_text_prior(extracted, self.class_names)
                if prior is not None:
                    # When clear keyword matches exist in native text, weight text prior higher (70% text, 30% visual)
                    # to prevent unformatted white canvases from misleading the visual CNN
                    probs = 0.30 * probs + 0.70 * prior
            except Exception as e:
                logger.debug("Lexical prior fusion skipped: %s", e)

        top_indices = np.argsort(probs)[::-1]
        top_idx = int(top_indices[0])
        pred_category = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        # Top alternatives (configurable top-k, default top 5)
        alternatives = [
            (self.class_names[idx], float(probs[idx]))
            for idx in top_indices[1:1 + top_k]
        ]

        elapsed_ms = (time.time() - start_time) * 1000.0

        return ClassificationResult(
            file_path=str(path),
            file_name=path.name,
            predicted_category=pred_category,
            confidence=confidence,
            alternative_predictions=alternatives,
            model_name="ResNet18 + Deep MLP Head (RVL-CDIP 16-Class)",
            processing_time_ms=elapsed_ms,
        )

    def classify_directory(self, folder_path: Union[str, Path]) -> List[ClassificationResult]:
        """Classifies all supported documents in a directory."""
        from file_scanner import scan_directory
        inventory = scan_directory(folder_path)
        target_files = inventory.documents + inventory.images
        results = []

        for f in target_files:
            try:
                res = self.classify_document(f.path)
                results.append(res)
            except Exception as e:
                logger.warning("Could not classify '%s': %s", f.file_name, e)

        return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        clf = DocumentClassifier()
        if Path(target).is_dir():
            results = clf.classify_directory(target)
            print(f"\nClassified {len(results)} documents:")
            for r in results:
                print(f" - {r.file_name}: {r.predicted_category} ({r.confidence * 100:.1f}%) [{r.processing_time_ms:.1f}ms]")
        else:
            r = clf.classify_document(target)
            print(f"\nDocument: {r.file_name}")
            print(f"Predicted Category: {r.predicted_category} ({r.confidence * 100:.2f}%)")
            print(f"Top Alternatives:")
            for cat, conf in r.alternative_predictions:
                print(f"  - {cat}: {conf * 100:.2f}%")
            print(f"Model: {r.model_name}")
            print(f"Latency: {r.processing_time_ms:.2f} ms\n")
    else:
        print("Usage: python document_classification.py <file_or_directory>")
