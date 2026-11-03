"""
Document Similarity Detection Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Features:
- Local text extraction for TXT, PDF, and DOCX files.
- OCR text extraction adapter for scanned files (with graceful fallback).
- In-memory document text caching to prevent redundant parsing and OCR.
- Text preprocessing: case normalization, whitespace stripping.
- TF-IDF Vectorization and Cosine Similarity calculation.
- Pairwise document comparison with configurable similarity thresholds.
- Safe handling of corrupted or unreadable documents.
- Preserves compatibility with teammate's remote RVL-CDIP dataset exploration.
"""

import logging
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Supported document formats
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

# Supported image formats for OCR routing
OCR_IMAGE_EXTENSIONS = {
    ".tiff",
    ".tif",
    ".png",
    ".jpg",
    ".jpeg",
}

DEFAULT_DOC_SIMILARITY_THRESHOLD = 75.0  # Percentage threshold for document near-duplicates

# In-memory document text cache to avoid repeated file parsing / OCR
_DOCUMENT_TEXT_CACHE: Dict[str, str] = {}


# ==========================================
# 1. CORE ALGORITHMS (PRESERVED FROM TEAMMATE)
# ==========================================

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes text for NLP similarity.
    Converts unicode ligatures/accents, converts to lowercase, and normalizes whitespace.
    """
    if not text:
        return ""
    # Unicode NFKD normalization (e.g. transforms ligatures like 'fi' -> 'f'+'i' and standardizes quotes)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    """
    Calculates TF-IDF Cosine Similarity between two text strings.
    Formula returns percentage on a 0-100 scale.
    """
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Empty text check
    if not text1.strip() or not text2.strip():
        return 0.0

    # If texts are character-for-character identical after preprocessing
    if text1 == text2:
        return 100.0

    documents = [text1, text2]

    try:
        vectorizer = TfidfVectorizer()
        tfidf_vector = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_vector[0:1], tfidf_vector[1:2])[0][0]
        return float(round(similarity * 100, 2))
    except ValueError as e:
        # Occurs if vocabulary is empty after stopword stripping
        logger.debug("TF-IDF calculation issue: %s", e)
        return 0.0


# ==========================================
# 2. LOCAL TEXT EXTRACTION ENGINES
# ==========================================

def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """Reads a plain text file using UTF-8 with fallback encodings."""
    path = Path(file_path)
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                return f.read()
        except Exception:
            continue
    raise IOError(f"Could not read text file: {path}")


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extracts text from PDF files using pdfplumber or PyPDF2.
    """
    path = Path(file_path)
    extracted_text = []

    # Attempt 1: pdfplumber (best formatting and accuracy)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)
        if extracted_text:
            return "\n".join(extracted_text)
    except Exception as e:
        logger.debug("pdfplumber extraction failed for %s, falling back to PyPDF2: %s", path, e)

    # Attempt 2: PyPDF2 fallback
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(page_text)
        return "\n".join(extracted_text)
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s", path, e)
        return ""


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """
    Extracts text from DOCX files.
    Uses native zipfile + XML parsing to avoid external dependencies.
    """
    path = Path(file_path)
    try:
        # Check if python-docx is installed
        try:
            import docx
            doc = docx.Document(path)
            parts = [p.text for p in doc.paragraphs if p.text]
            # Also extract table text so tabular documents (invoices, forms) are fully represented
            for table in doc.tables:
                for row in table.rows:
                    row_parts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_parts:
                        parts.append(" | ".join(row_parts))
            return "\n".join(parts)
        except ImportError:
            pass

        # Native fallback: DOCX is a zip file containing word/document.xml
        with zipfile.ZipFile(path, "r") as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)

            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            body = tree.find("w:body", ns)

            lines = []
            if body is not None:
                for elem in body:
                    tag = elem.tag.split("}")[-1]
                    if tag == "p":
                        p_text = "".join(node.text for node in elem.iter() if node.tag.endswith("}t") and node.text)
                        if p_text.strip():
                            lines.append(p_text.strip())
                    elif tag == "tbl":
                        for row in elem.findall(".//w:tr", ns):
                            cells = []
                            for cell in row.findall(".//w:tc", ns):
                                c_text = "".join(node.text for node in cell.iter() if node.tag.endswith("}t") and node.text)
                                if c_text.strip():
                                    cells.append(c_text.strip())
                            if cells:
                                lines.append(" | ".join(cells))
            if lines:
                return "\n".join(lines)

            # Fallback if specific body structure is absent
            texts = [
                node.text
                for node in tree.iter()
                if node.tag.endswith("}t") and node.text
            ]
            return "\n".join(texts)
    except Exception as e:
        logger.warning("DOCX extraction failed for %s: %s", path, e)
        return ""


def extract_text_with_ocr(target: Union[str, Path, object]) -> str:
    """
    Extracts text using OCR (Tesseract) for scanned documents, images, or PIL Images.
    Gracefully logs and returns empty string if Tesseract is not installed.
    """
    try:
        import pytesseract
        from PIL import Image

        if isinstance(target, (str, Path)):
            path = Path(target)
            with Image.open(path) as img:
                return pytesseract.image_to_string(img)
        elif hasattr(target, "convert"):
            # Target is already a PIL Image
            return pytesseract.image_to_string(target)
        else:
            return ""
    except ImportError:
        target_name = getattr(target, "name", str(target))
        logger.info(
            "OCR skipped for '%s': 'pytesseract' is not installed.", target_name
        )
        return ""
    except Exception as e:
        target_name = getattr(target, "name", str(target))
        logger.warning("OCR processing failed for '%s': %s", target_name, e)
        return ""


@dataclass
class TextExtractionResult:
    """Stores text extraction outputs alongside data quality metrics."""
    text: str
    char_count: int
    word_count: int
    line_count: int
    whitespace_ratio: float
    alphanumeric_ratio: float
    is_scanned: bool = False
    ocr_applied: bool = False
    quality_score: float = 1.0


def compute_extraction_quality(
    text: str,
    is_scanned: bool = False,
    ocr_applied: bool = False,
) -> TextExtractionResult:
    """Calculates text health and quality indicators."""
    total_chars = len(text)
    if total_chars == 0:
        return TextExtractionResult(
            text="",
            char_count=0,
            word_count=0,
            line_count=0,
            whitespace_ratio=0.0,
            alphanumeric_ratio=0.0,
            is_scanned=is_scanned,
            ocr_applied=ocr_applied,
            quality_score=0.0,
        )

    words = text.split()
    word_count = len(words)
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = len(lines)

    whitespace_chars = sum(1 for c in text if c.isspace())
    alphanumeric_chars = sum(1 for c in text if c.isalnum())

    ws_ratio = round(whitespace_chars / total_chars, 4)
    alnum_ratio = round(alphanumeric_chars / total_chars, 4)

    # Heuristic quality score: penalize extreme whitespace or non-alphanumeric noise
    score = 1.0
    if total_chars < 30:
        score *= 0.5
    if alnum_ratio < 0.4:
        score *= (alnum_ratio / 0.4)
    if ws_ratio > 0.6:
        score *= (1.0 - (ws_ratio - 0.6))
    score = max(0.0, min(1.0, round(score, 3)))

    return TextExtractionResult(
        text=text,
        char_count=total_chars,
        word_count=word_count,
        line_count=line_count,
        whitespace_ratio=ws_ratio,
        alphanumeric_ratio=alnum_ratio,
        is_scanned=is_scanned,
        ocr_applied=ocr_applied,
        quality_score=score,
    )


def extract_document_text_with_metadata(
    file_path: Union[str, Path],
    enable_ocr: bool = False,
    use_cache: bool = True,
) -> TextExtractionResult:
    """
    Extracts text from a document and returns detailed quality metrics.
    Automatically detects scanned PDFs (< 20 chars embedded text) and triggers OCR if enabled.
    """
    path = Path(file_path).resolve()
    cache_key = str(path)

    if use_cache and cache_key in _DOCUMENT_TEXT_CACHE:
        cached_text = _DOCUMENT_TEXT_CACHE[cache_key]
        return compute_extraction_quality(cached_text)

    ext = path.suffix.lower()
    text = ""
    is_scanned = False
    ocr_applied = False

    if ext == ".txt":
        text = extract_text_from_txt(path)
    elif ext == ".pdf":
        text = extract_text_from_pdf(path)
        # Scanned PDF check: embedded text is empty or fewer than 20 characters
        if len(text.strip()) < 20:
            is_scanned = True
            if enable_ocr:
                # Render PDF pages to images and execute OCR
                try:
                    import pdfplumber
                    with pdfplumber.open(path) as pdf:
                        ocr_pages = []
                        for idx, page in enumerate(pdf.pages):
                            try:
                                page_img = page.to_image(resolution=200).original
                                p_text = extract_text_with_ocr(page_img)
                                if p_text.strip():
                                    ocr_pages.append(p_text.strip())
                            except Exception as render_err:
                                logger.debug("Page %d rendering for OCR failed: %s", idx, render_err)
                        if ocr_pages:
                            text = "\n".join(ocr_pages)
                            ocr_applied = True
                except Exception as pdf_ocr_err:
                    logger.debug("PDF page OCR pipeline failed: %s", pdf_ocr_err)
    elif ext == ".docx":
        text = extract_text_from_docx(path)
    elif ext in OCR_IMAGE_EXTENSIONS and enable_ocr:
        is_scanned = True
        text = extract_text_with_ocr(path)
        ocr_applied = bool(text.strip())
    else:
        logger.debug("Unsupported document format for text extraction: %s", ext)

    if use_cache:
        _DOCUMENT_TEXT_CACHE[cache_key] = text

    return compute_extraction_quality(
        text=text,
        is_scanned=is_scanned,
        ocr_applied=ocr_applied,
    )


def extract_document_text(
    file_path: Union[str, Path],
    enable_ocr: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Extracts plain text from a document based on its file extension.
    Uses memory cache to prevent redundant reads.

    Args:
        file_path: Path to the document.
        enable_ocr: Whether to attempt OCR for images/scanned documents.
        use_cache: Whether to use cached text if available.

    Returns:
        Extracted plain text.
    """
    result = extract_document_text_with_metadata(
        file_path=file_path,
        enable_ocr=enable_ocr,
        use_cache=use_cache,
    )
    return result.text


def clear_document_cache() -> None:
    """Clears the in-memory document text cache."""
    _DOCUMENT_TEXT_CACHE.clear()


# ==========================================
# 3. DOCUMENT COMPARISON AND SCANNERS
# ==========================================


@dataclass
class DocumentComparisonResult:
    """Stores the comparison metrics between two documents."""
    file_path_1: str
    file_path_2: str
    file_name_1: str
    file_name_2: str
    similarity_score: float
    duplicate_type: str
    is_duplicate: bool
    detection_algorithm: str = "TF-IDF + Cosine Similarity"


def categorize_document_duplicate(
    similarity_score: float,
    ext1: str,
    ext2: str,
    threshold: float = DEFAULT_DOC_SIMILARITY_THRESHOLD,
) -> Tuple[str, bool]:
    """
    Categorizes the relationship between two documents based on similarity score.
    """
    is_cross_format = ext1.lower() != ext2.lower()

    if similarity_score >= 99.5:
        category = "cross_format_document" if is_cross_format else "exact_duplicate"
        return category, True

    if similarity_score >= threshold:
        category = "cross_format_document" if is_cross_format else "near_duplicate_document"
        return category, True

    if similarity_score >= max(50.0, threshold - 15.0):
        return "possible_duplicate", False

    return "non_duplicate", False


def compare_two_documents(
    doc1_path: Union[str, Path],
    doc2_path: Union[str, Path],
    threshold: float = DEFAULT_DOC_SIMILARITY_THRESHOLD,
    enable_ocr: bool = False,
) -> DocumentComparisonResult:
    """
    Compares two documents and returns the similarity metrics.
    """
    p1 = Path(doc1_path)
    p2 = Path(doc2_path)

    text1 = extract_document_text(p1, enable_ocr=enable_ocr)
    text2 = extract_document_text(p2, enable_ocr=enable_ocr)

    sim = calculate_tfidf_similarity(text1, text2)
    category, is_dup = categorize_document_duplicate(
        sim, p1.suffix, p2.suffix, threshold=threshold
    )

    return DocumentComparisonResult(
        file_path_1=str(p1),
        file_path_2=str(p2),
        file_name_1=p1.name,
        file_name_2=p2.name,
        similarity_score=sim,
        duplicate_type=category,
        is_duplicate=is_dup,
    )


def scan_and_compare_documents(
    doc_paths: List[Path],
    threshold: float = DEFAULT_DOC_SIMILARITY_THRESHOLD,
    enable_ocr: bool = False,
) -> List[DocumentComparisonResult]:
    """
    Performs pairwise document comparison without self-comparison or duplicate pairs.
    """
    results: List[DocumentComparisonResult] = []

    # Pre-extract text for all documents
    for path in doc_paths:
        extract_document_text(path, enable_ocr=enable_ocr, use_cache=True)

    for p1, p2 in combinations(doc_paths, 2):
        text1 = extract_document_text(p1, use_cache=True)
        text2 = extract_document_text(p2, use_cache=True)

        sim = calculate_tfidf_similarity(text1, text2)
        category, is_dup = categorize_document_duplicate(
            sim, p1.suffix, p2.suffix, threshold=threshold
        )

        if is_dup:
            results.append(
                DocumentComparisonResult(
                    file_path_1=str(p1),
                    file_path_2=str(p2),
                    file_name_1=p1.name,
                    file_name_2=p2.name,
                    similarity_score=sim,
                    duplicate_type=category,
                    is_duplicate=is_dup,
                )
            )

    return results


# ==========================================
# 4. REMOTE DATASET STUBS (PRESERVED FOR COMPATIBILITY)
# ==========================================

def get_duplicate_documents():
    """Preserved teammate function. Requires 'datasets' package when active."""
    try:
        from datasets import load_dataset
        duplicate_dataset = load_dataset(
            "stefan-hf/rvlcdip-duplicates", "documents", split="test"
        )
        for row in duplicate_dataset:
            if row.get("has_train_duplicate"):
                return row
        return None
    except ImportError:
        logger.warning("'datasets' package not installed. Remote RVL-CDIP access disabled.")
        return None


def get_ocr_text(target_id, split):
    """Preserved teammate function. Requires 'datasets' package when active."""
    try:
        from datasets import load_dataset
        ocr_dataset = load_dataset(
            "stefan-hf/rvlcdip-redact", "textract_redact", split=split, streaming=True
        )
        for row in ocr_dataset:
            if row.get("id") == target_id:
                return row.get("text")
        return None
    except ImportError:
        logger.warning("'datasets' package not installed. Remote RVL-CDIP access disabled.")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        f1, f2 = sys.argv[1], sys.argv[2]
        res = compare_two_documents(f1, f2)
        print(f"\nDocument 1: {res.file_name_1}")
        print(f"Document 2: {res.file_name_2}")
        print(f"Similarity Score: {res.similarity_score}%")
        print(f"Category: {res.duplicate_type}")
        print(f"Is Duplicate: {res.is_duplicate}\n")
    else:
        print("Usage: python document_detection.py <file1> <file2>")
