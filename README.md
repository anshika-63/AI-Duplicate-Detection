# AI-Based Duplicate Document & Near-Duplicate Image Detection

**T243 – Two-Member Academic Project**  
Unified architecture coordinating exact duplicate hashing, perceptual image hashing, and natural language document similarity detection.

---

## Architecture Overview

The system integrates three specialized detection modules coordinated by an automated routing and deduplication engine:

```text
AI-Duplicate-Detection/
│
├── main.py                          # Module 1: Exact Duplicate Detection (SHA-256)
├── image_detection.py               # Module 2: Near-Duplicate Image Detection (pHash, dHash)
├── document_detection.py            # Module 3: Document Similarity Detection (TF-IDF, Cosine)
│
├── unified_detector.py              # Central Orchestrator & Unified CLI
├── file_scanner.py                  # Common File Ingestion & Format Classifier
├── result_manager.py                # Canonical Pair Deduplication & Normalization
├── report_generator.py              # Unified Multi-Modal CSV & Summary Generator
├── database_manager.py              # Optional Centralized SQLite Storage
│
├── test_image_detection.py          # Module 2 Unit Tests (10 tests)
├── tests/
│   └── test_integration.py          # End-to-End Synthetic Integration Tests (18 tests)
│
├── duplicate_report.csv             # Module 1 Report (SHA-256 exact matches)
├── image_duplicate_report.csv       # Module 2 Report (Perceptual image pairs)
├── combined_duplicate_report.csv    # Unified Multi-Modal Report
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Ignored build and cache artifacts
```

---

## Modules

### Module 1: Exact Duplicate Detection Using SHA-256 (`main.py`)
- Calculates chunked 4096-byte SHA-256 cryptographic digests.
- Groups identical files across arbitrary directory structures.
- Generates `duplicate_report.csv`.

### Module 2: Near-Duplicate Image Detection (`image_detection.py`)
- Employs 64-bit Perceptual Hash (`pHash`) and Difference Hash (`dHash`).
- Computes Hamming distance and similarity: $\text{Similarity (\%)} = (1 - \text{Hamming} / 64) \times 100$.
- Memory-efficient streaming (closes full images immediately after hashing).
- Generates `image_duplicate_report.csv`.

### Module 3: Document Similarity Detection (`document_detection.py`)
- Native text extraction for `.txt`, `.pdf` (`PyPDF2`, `pdfplumber`), and `.docx` (native XML parser).
- Extracted text caching prevents redundant document reads or OCR runs.
- TF-IDF Vectorization with Cosine Similarity ($0\text{--}100\%$).
- Handles cross-format similarity (e.g. comparing PDF vs DOCX guidelines).

---

## Unified Execution & CLI

The unified pipeline routes each file to appropriate algorithms based on extension:
* **Images (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`)** $\rightarrow$ SHA-256 + pHash + dHash
* **Documents (`.pdf`, `.docx`, `.txt`)** $\rightarrow$ SHA-256 + Document Similarity (TF-IDF + Cosine)
* **TIFF / Scanned Documents** $\rightarrow$ SHA-256 + Image Detection + Optional OCR

### Run All Modules
```bash
python unified_detector.py --folder dataset --all
```

### Run Selectively
```bash
# Exact duplicates only (SHA-256)
python unified_detector.py --folder dataset --exact

# Near-duplicate images only (pHash + dHash)
python unified_detector.py --folder dataset --images --threshold 85.0

# Document similarity only (TF-IDF + Cosine)
python unified_detector.py --folder dataset --documents --threshold 75.0

# Store scan results to SQLite database
python unified_detector.py --folder dataset --all --db duplicate_detection.db
```

---

## Testing

All test suites run independently on synthetic data without requiring external datasets:

```bash
# Run Module 2 unit tests
python -m unittest test_image_detection.py

# Run complete 18-condition integration test suite
python -m unittest tests/test_integration.py
```

---

## Installation

```bash
git clone https://github.com/anshika-63/AI-Duplicate-Detection.git
cd AI-Duplicate-Detection
git checkout feature/image-detection
pip install -r requirements.txt
```

---

## Repository
GitHub: https://github.com/anshika-63/AI-Duplicate-Detection
