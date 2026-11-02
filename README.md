# AI-Based Duplicate Document & Near-Duplicate Image Detection

**T243 – Two-Member Academic Project**  
Unified architecture coordinating exact duplicate hashing, perceptual and feature-based image comparison, natural language document similarity detection, and deep-learning 16-class document classification.

---

## Architecture Overview

The system integrates four specialized detection and classification modules coordinated by an automated routing, deduplication, and persistence engine:

```text
AI-Duplicate-Detection/
│
├── main.py                          # Module 1: Exact Duplicate Detection (SHA-256)
├── image_detection.py               # Module 2: Near-Duplicate Image Detection (pHash, dHash)
├── orb_detection.py                 # Module 2: ORB Feature Matching & RANSAC Homography
├── document_similarity.py           # Module 3: Document Similarity Detection (TF-IDF, Cosine)
├── document_detection.py            # Module 3: Legacy Document Ingestion & PyMuPDF Support
├── document_classification.py       # Module 4: 16-Class Document Classifier Inference Engine
├── train_document_classifier.py     # Module 4: ResNet18 Feature Extraction & MLP Head Trainer
│
├── unified_detector.py              # Central Orchestrator & Multi-Modal Unified CLI
├── file_scanner.py                  # Common Multi-Format Ingestion & File Classifier
├── result_manager.py                # Canonical Pair Deduplication & Normalization
├── report_generator.py              # Unified Multi-Modal CSV & Summary Generator
├── database_manager.py              # Centralized SQLite Persistence (4 tables)
├── app.py                           # Interactive Streamlit Web Dashboard (6 tabs)
│
├── test_image_detection.py          # Module 2 Unit Tests (10 tests)
├── tests/
│   ├── test_integration.py          # End-to-End Synthetic Integration Tests (18 tests)
│   └── test_document_classification.py # Module 4 Classifier Tests (4 tests)
│
├── tools/
│   ├── dataset_inventory.py         # Automated scan & audit of local datasets
│   └── benchmark_images.py          # Evaluation on INRIA Copydays & Synthetic Crops
│
├── reports/                         # Measured evaluation CSVs and visualizations
│   ├── document_classification_report.csv
│   ├── model_comparison.csv
│   ├── class_distribution.csv
│   ├── training_history.csv
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── image_benchmark_results.csv
│   ├── threshold_analysis.csv
│   └── synthetic_crop_benchmark.csv
│
├── duplicate_report.csv             # Module 1 Report (SHA-256 exact matches)
├── image_duplicate_report.csv       # Module 2 Report (Perceptual image pairs)
├── combined_duplicate_report.csv    # Unified Multi-Modal Report (10 columns)
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Comprehensive system documentation
└── .gitignore                       # Checkpoint, cache, and database exclusions
```

---

## Detection & Classification Modules

### Module 1: Exact Duplicate Detection Using SHA-256 (`main.py`)
- Calculates chunked 4096-byte SHA-256 cryptographic digests.
- Groups identical files across arbitrary directory structures.
- Generates `duplicate_report.csv` without overwriting teammate work.

### Module 2: Near-Duplicate Image Detection (`image_detection.py`, `orb_detection.py`)
- **Perceptual Hashing:** 64-bit Perceptual Hash (`pHash`) and Difference Hash (`dHash`).
  $$\text{Similarity (\%)} = \left(1 - \frac{\text{Hamming Distance}}{64}\right) \times 100$$
- **Geometric Invariance (ORB + RANSAC):** Fast oriented FAST and rotated BRIEF descriptor matching with Lowe's ratio test ($0.75$) and RANSAC homography estimation for heavy cropping ($50\%\text{--}70\%$) and arbitrary rotations.
- Memory-efficient streaming (closes full images immediately after hashing).
- Generates `image_duplicate_report.csv`.

### Module 3: Document Similarity Detection (`document_similarity.py`, `document_detection.py`)
- Native text extraction for `.txt`, `.pdf` (`PyMuPDF`, `pdfplumber`), and `.docx` (XML / python-docx parser).
- Extracted text caching prevents redundant document reads or OCR runs.
- TF-IDF Vectorization with Cosine Similarity ($0\text{--}100\%$).
- Handles cross-format similarity (e.g. comparing PDF vs DOCX files).

### Module 4: 16-Class Document Classification (`document_classification.py`, `train_document_classifier.py`)
- **Dataset:** RVL-CDIP (8,992 document images across 16 categories: Letter, Form, Email, Handwritten, Advertisement, Scientific Report, Scientific Publication, Specification, File Folder, News Article, Budget, Invoice, Presentation, Questionnaire, Resume, Memo).
- **Architecture:** Pretrained ResNet-18 feature extraction (512-dim cached embeddings) with Deep MLP Classification Head (`Linear(512, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 16)`).
- **Performance:** **87.40% Test Accuracy** and **0.8737 Weighted F1** on official RVL-CDIP test split.
- **Multimodal Inference:** Classifies images directly; extracts text from PDFs/DOCX and renders representative previews for classification with top-3 prediction confidence ranking.

---

## Experimental Evaluation & Benchmark Results

### 1. Near-Duplicate Image Benchmark (INRIA Copydays)
Evaluated across 157 originals, 1,413 JPEG quality attacks, and synthetic cropping/rotation transformations:

| Algorithm | Attack Type | Optimal Threshold | Precision | Recall | F1 Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pHash** | JPEG Compression | $\ge 80.0\%$ | 100.0% | 100.0% | **1.0000** | 100.0% |
| **dHash** | JPEG Compression | $\ge 80.0\%$ | 100.0% | 100.0% | **1.0000** | 100.0% |
| **pHash** | 50% Crop / 90° Rotation | $\ge 80.0\%$ | 100.0% | 0.0% – 4.0% | 0.0769 | 50.0% |
| **dHash** | 50% Crop / 90° Rotation | $\ge 80.0\%$ | 100.0% | 0.0% – 16.0% | 0.2759 | 56.0% |
| **ORB + RANSAC** | 50% Crop / 90° Rotation | Inliers $\ge 12$ | 95.7% | 88.0% – 100.0% | **0.9362** | **94.0%** |

*Takeaway:* Perceptual hashing (pHash/dHash) provides ultra-fast $O(1)$ duplicate lookup for compression and resizing, while ORB feature matching provides robustness against aggressive cropping and rotation.

### 2. RVL-CDIP 16-Class Document Classification
Trained on 8,000 samples (85% train / 15% validation) and evaluated on official 992-sample test partition:

| Model Architecture | Train Acc | Val Acc | Test Acc | Weighted F1 | Macro F1 | Epochs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Linear Classifier) | 88.16% | 82.08% | 79.84% | 0.7963 | 0.7964 | 20 |
| **Deep MLP Head (BatchNorm + Dropout)** | **97.88%** | **89.50%** | **87.40%** | **0.8737** | **0.8739** | 25 |

Confusion matrix and loss curves are stored in `reports/confusion_matrix.png` and `reports/training_curves.png`.

---

## Interactive Streamlit Web Dashboard

Launch the browser application:

```bash
streamlit run app.py
```

### Dashboard Tabs:
1. **Near-Duplicate Image Detection:** Upload two images to compare with pHash, dHash, ORB feature inliers, and visual side-by-side difference overlay.
2. **Document Classification:** Upload any document (`.png`, `.jpg`, `.tiff`, `.pdf`, `.docx`, `.txt`) to view predicted RVL-CDIP category, confidence percentage, and top-3 ranking breakdown.
3. **Document Similarity:** Upload two documents to compute TF-IDF cosine similarity, character/word metrics, and inline color-coded diff comparison.
4. **Unified Scanner:** Recursively scan any directory with all four modules active, filter pairs by threshold, and export custom CSV reports.
5. **Benchmark & Evaluation:** Interactive visualizations of confusion matrices, class distribution, training loss/accuracy curves, and algorithm comparison charts.
6. **Database & History:** View past scan runs, query scanned files and deduplication records, and filter results by duplicate category.

---

## Unified Command-Line Interface (CLI)

The unified pipeline automatically routes files by extension:
* **Images (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`)** $\rightarrow$ SHA-256 + pHash + dHash + ORB
* **Documents (`.pdf`, `.docx`, `.txt`)** $\rightarrow$ SHA-256 + TF-IDF Cosine Similarity
* **Document Images / Scanned Files** $\rightarrow$ 16-Class Document Classifier

### CLI Usage Commands:

```bash
# 1. Run all modules (Exact SHA-256, Images, Documents, and Document Classification)
python unified_detector.py --folder dataset --all

# 2. Run exact duplicate detection only (SHA-256)
python unified_detector.py --folder dataset --exact

# 3. Run near-duplicate image detection only
python unified_detector.py --folder dataset --images --threshold 85.0

# 4. Run document similarity detection only
python unified_detector.py --folder dataset --documents --threshold 75.0

# 5. Run 16-class document classification only
python unified_detector.py --folder dataset --classification

# 6. Run all modules and persist results to SQLite database
python unified_detector.py --folder dataset --all --db duplicate_detection.db
```

---

## Running Test Suites

Execute the comprehensive unit and integration test suites:

```bash
# Run Module 2 unit tests (10 tests)
python -m unittest test_image_detection.py

# Run Module 4 document classification unit tests (4 tests)
python -m unittest tests/test_document_classification.py

# Run synthetic end-to-end integration tests (18 tests)
python -m unittest tests/test_integration.py

# Run all 32 tests in one command
python -m unittest discover -s tests -p "test_*.py"
```

All 32 tests pass completely on synthetic datasets without external downloads.

---

## Installation & Setup

```bash
git clone https://github.com/anshika-63/AI-Duplicate-Detection.git
cd AI-Duplicate-Detection
git checkout feature/image-detection
pip install -r requirements.txt
```

---

## Team Contributions & Roles
* **Teammate (Module 1):** Exact file duplicate detection using chunked cryptographic SHA-256 (`main.py`, `duplicate_report.csv`).
* **Author (Modules 2, 3, 4 & Unified System):** Near-duplicate perceptual hashing & ORB matching (`image_detection.py`, `orb_detection.py`), natural language document similarity (`document_similarity.py`, `document_detection.py`), 16-class deep document classification (`document_classification.py`, `train_document_classifier.py`), unified detection engine (`unified_detector.py`), SQLite manager (`database_manager.py`), and Streamlit web UI (`app.py`).
