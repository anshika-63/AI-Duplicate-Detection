# AI-Based Duplicate Document & Near-Duplicate Image Detection

**Project Code:** T243 – Two-Member Academic Project  
**Repository:** [https://github.com/anshika-63/AI-Duplicate-Detection](https://github.com/anshika-63/AI-Duplicate-Detection)  
**Branch:** `feature/image-detection`  
**Execution Environment:** Windows 11, Python 3.13.5, PyTorch 2.9.1+cpu  

Unified multi-modal architecture coordinating exact cryptographic file deduplication, perceptual and feature-based image comparison, natural language document similarity detection, and deep-learning 16-class document classification.

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
├── document_detection.py            # Module 3: Text Extraction (PDF, DOCX, TXT) & OCR Adapter
├── document_classification.py       # Module 4: 16-Class Document Classifier Inference Engine
├── train_document_classifier.py     # Module 4: ResNet18 Feature Extraction & MLP Head Trainer
│
├── unified_detector.py              # Central Orchestrator & Multi-Modal Unified CLI
├── file_scanner.py                  # Common Multi-Format Ingestion & File Classifier
├── result_manager.py                # Canonical Pair Deduplication & Normalization
├── report_generator.py              # Unified Multi-Modal CSV & Execution Summary Generator
├── database_manager.py              # Centralized SQLite Persistence (4 tables)
├── app.py                           # Interactive Streamlit Web Dashboard (7 tabs)
│
├── test_image_detection.py          # Module 2 Unit Tests (10 tests)
├── tests/
│   ├── test_database.py             # SQLite Persistence Tests (4 tests)
│   ├── test_document_classification.py          # Classifier Baseline Tests (4 tests)
│   ├── test_document_classification_regression.py # Multimodal Regression Tests (11 tests)
│   ├── test_document_similarity_regression.py   # Document Similarity Tests (9 tests)
│   ├── test_image_regression.py     # Image Transformation Regression Tests (7 tests)
│   └── test_integration.py          # End-to-End Synthetic Integration Tests (18 tests)
│
├── tools/
│   ├── dataset_inventory.py         # Automated scan & audit of local datasets
│   ├── benchmark_images.py          # Evaluation on INRIA Copydays & Synthetic Attacks
│   └── benchmark_documents.py       # Evaluation on Document Similarity Benchmark Matrix
│
├── reports/                         # Measured evaluation CSVs and visualizations
│   ├── document_classification_report.csv
│   ├── document_classification_predictions.csv
│   ├── model_comparison.csv
│   ├── class_distribution.csv
│   ├── training_history.csv
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── image_benchmark_results.csv
│   ├── synthetic_crop_benchmark.csv
│   ├── synthetic_transformation_benchmark.csv
│   └── document_similarity_benchmark.csv
│
├── duplicate_report.csv             # Module 1 Report (SHA-256 exact matches)
├── image_duplicate_report.csv       # Module 2 Report (Perceptual image pairs)
├── combined_duplicate_report.csv    # Unified Multi-Modal Report (10 columns)
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Comprehensive system documentation
├── PROJECT_REPORT.md                # Comprehensive academic technical report
├── BENCHMARK_REPORT.md              # Empirical benchmark tables and evaluations
├── TESTING_REPORT.md                # Automated test suite and regression log
├── KNOWN_LIMITATIONS.md             # Dataset and hardware disclosures
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
- In-memory text caching prevents redundant document reads or OCR runs.
- Extraction health metrics: character count, word count, line count, whitespace ratio, alphanumeric ratio, quality score.
- Scanned PDF detection ($< 20$ embedded characters) with automatic page rendering and OCR fallback.
- TF-IDF Vectorization with Cosine Similarity ($0\text{--}100\%$).
- Handles cross-format similarity (e.g. comparing PDF vs DOCX files).

### Module 4: 16-Class Document Classification (`document_classification.py`, `train_document_classifier.py`)
- **Dataset:** RVL-CDIP (8,992 document images across 16 categories: Letter, Form, Email, Handwritten, Advertisement, Scientific Report, Scientific Publication, Specification, File Folder, News Article, Budget, Invoice, Presentation, Questionnaire, Resume, Memo).
- **Architecture:** Pretrained ResNet-18 feature extraction (512-dim cached embeddings) with Deep MLP Classification Head (`Linear(512, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 16)`).
- **Performance:** **87.40% Test Accuracy** and **0.8737 Weighted F1** on official RVL-CDIP test split.
- **Multimodal Inference:** Classifies images directly; extracts text from PDFs/DOCX with keyword prior fusion (70% text / 30% visual for native text; 100% visual for scanned images) and configurable top-k ranking.

---

## Experimental Evaluation & Benchmark Results

### 1. RVL-CDIP 16-Class Document Classification
Evaluated on the official untouched 992-sample test partition (exactly 62 samples per category across 16 classes):

| Model Architecture | Train Acc | Val Acc | Test Acc | Weighted F1 | Macro F1 | Epochs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Linear Classifier) | 88.16% | 82.08% | 79.84% | 0.7963 | 0.7964 | 20 |
| **Deep MLP Head (BatchNorm + Dropout)** | **97.88%** | **89.50%** | **87.40%** | **0.8737** | **0.8739** | 25 |

Confusion matrix and loss curves are stored in `reports/confusion_matrix.png` and `reports/training_curves.png`.

### 2. Near-Duplicate Image Benchmark (INRIA Copydays & Synthetic Attacks)
Evaluated across 157 originals, 1,413 JPEG quality attacks, and 8 controlled transformation categories:

| Transformation Category | Applied Parameters | pHash Sim | dHash Sim | Predicted Status | Ground Truth | Verdict |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| Exact File Copy | Byte-identical duplicate | 100.00% | 100.00% | Exact Duplicate | Duplicate | **PASS** |
| Re-encoded PNG | Decoded and re-saved | 100.00% | 100.00% | Exact Duplicate | Duplicate | **PASS** |
| Resized 50% | Downscaled $60 \times 60$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| JPEG Compressed | Quality factor $Q=25$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| Brightness Shift | Enhanced brightness $+25\%$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| Rotated 15° | $15^\circ$ affine rotation | 84.38% | 85.94% | Near Duplicate | Duplicate | **PASS** |
| Cropped 50% | Center bounding crop | 82.81% | 84.38% | Near Duplicate | Duplicate | **PASS** |
| Negative Control | Striped pattern image | 65.62% | 46.88% | Non-Duplicate | Non-Duplicate | **PASS** |

**Summary Evaluation:** 8/8 evaluated, **100.00% Accuracy**, **1.0000 Precision**, **1.0000 Recall**, **1.0000 F1-Score**.

### 3. Document Similarity Benchmark Matrix (9 Scenarios)

| Benchmark Case | Evaluated Documents | Cosine Sim | Predicted Category | Ground Truth | Verdict |
| :--- | :--- | :---: | :--- | :---: | :---: |
| 1. Exact Copy (TXT) | Identical plain text files | 100.00% | `exact_duplicate` | Duplicate | **PASS** |
| 2. Cross-Format Equivalence | TXT vs native XML DOCX | 100.00% | `cross_format_document` | Duplicate | **PASS** |
| 3. Minor Token Change | 1 token modified | 97.47% | `near_duplicate_document` | Duplicate | **PASS** |
| 4. Paragraph Permutation | Shuffled paragraph order | 100.00% | `exact_duplicate` | Duplicate | **PASS** |
| 5. Moderate Modification | 1 sentence paraphrased | 92.59% | `near_duplicate_document` | Duplicate | **PASS** |
| 6. Borderline Review | 2 sentences rewritten | 69.53% | `possible_duplicate` | Non-Duplicate | **PASS** |
| 7. Heavy Modification | Majority replaced | 29.21% | `non_duplicate` | Non-Duplicate | **PASS** |
| 8. Unrelated Domain | NLP vs Quantum Physics | 0.88% | `non_duplicate` | Non-Duplicate | **PASS** |
| 9. Empty Document | Empty vs Base | 0.00% | `non_duplicate` | Non-Duplicate | **PASS** |

**Summary Evaluation:** 9/9 evaluated, **100.00% Accuracy**, **1.0000 Precision**, **1.0000 Recall**, **1.0000 F1-Score**.

---

## Interactive Streamlit Web Dashboard

Launch the browser application:

```bash
streamlit run app.py
```

### Dashboard Tabs (7 Tabs):
1. **📊 Overview:** Live system metrics, database statistics, and unified directory scanner.
2. **🔒 Exact Duplicates:** Cryptographic SHA-256 duplicate clusters (`duplicate_report.csv`).
3. **🖼️ Image Detection:** Upload two images to compare with pHash, dHash, ORB feature inliers, and visual verdicts.
4. **📑 Document Classification:** Upload any document (`.png`, `.jpg`, `.tiff`, `.pdf`, `.docx`, `.txt`) to view predicted RVL-CDIP category, confidence percentage, and top-5 ranking breakdown.
5. **📝 Document Similarity:** Upload two documents to compute TF-IDF cosine similarity, character/word metrics, and text previews.
6. **📈 Reports & Evaluation:** Interactive visualizations of confusion matrices, class distribution, training loss/accuracy curves, and benchmark tables.
7. **💾 Database & History:** View past scan runs, query scanned files and deduplication records, and filter results by similarity threshold and duplicate category.

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

# 3. Run near-duplicate image detection only with custom threshold
python unified_detector.py --folder dataset --images --threshold 85.0

# 4. Run document similarity detection only
python unified_detector.py --folder dataset --documents --threshold 75.0

# 5. Run 16-class document classification only
python unified_detector.py --folder dataset --classification

# 6. Run all modules with ORB geometric verification and persist results to SQLite database
python unified_detector.py --folder dataset --all --orb --db duplicate_detection.db
```

---

## Running Test Suites

Execute the comprehensive unit and integration test suites:

```bash
# Run all tests in the repository (63 tests)
python -m unittest discover -s tests -p "test_*.py"
python -m unittest test_image_detection.py
```

All 63 tests pass completely in under 12 seconds on synthetic datasets without external network downloads.

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
### How to Run

1. Clone the repository.
2. Open the project folder in VS Code.
3. Ensure Python is installed.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Place sample files inside the `dataset` folder.
6. Run Module 1 (Exact Duplicates using SHA-256):
   ```bash
   python main.py
   ```
7. Run Module 2 (Near-Duplicate Image Detection using pHash & dHash):
   ```bash
   python image_detection.py dataset
   ```
8. Run the test suite:
   ```bash
   python test_image_detection.py
   ```

### Current Development Status
- **Completed:** Module 1 – Exact Duplicate Detection using SHA-256 hashing.
- **Completed:** Module 2 – Near-Duplicate Image Detection using pHash and dHash.
- **Upcoming:** Document similarity (TF-IDF/cosine) and advanced feature extraction.


## Module 3: Document Similarity Detection using TF-IDF & Cosine Similarity

Module 3 detects **duplicate and near-duplicate documents** by comparing their extracted OCR text.

Unlike exact duplicate detection, where the files must have identical binary content, this module compares the **textual content** of documents. This is useful when two documents contain similar information but may differ in formatting, spacing, scanning quality, or other file-level properties.

### Dataset Used

For document similarity testing, the project uses the following publicly available datasets from Hugging Face:

- `stefan-hf/rvlcdip-duplicates` — provides known duplicate relationships between RVL-CDIP test and training documents.
- `stefan-hf/rvlcdip-redact` — provides OCR text associated with RVL-CDIP document images.

The project uses **streaming** for the OCR dataset so that the complete dataset does not need to be downloaded locally.

### Workflow

```text
Document Dataset
       ↓
Known Duplicate Metadata
       ↓
Find Test Document + Known Training Duplicate
       ↓
Retrieve OCR Text
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Document Similarity Score
```

### 1. OCR Text Retrieval

Each document is identified using its unique document ID.

The duplicate dataset provides the relationship between a test document and its known training duplicate. The corresponding OCR text is then retrieved from the OCR dataset using the same document ID.

Example:

```text
Test document:
test_resume_29d565d2-3664-446c-ab99-5525ebd342a5.jpg

Known training duplicate:
train_resume_cff691b6-8d5b-4ed5-acfa-bfcc26fa9169.jpg
```

The OCR text for both documents is retrieved before calculating similarity.

### 2. Text Preprocessing

Before similarity calculation, OCR text is normalized.

The current preprocessing performs:

- Conversion to lowercase
- Removal of unnecessary extra spaces
- Removal of extra newline characters

Example:

```python
def preprocess_text(text):
    text = text.lower()
    text = " ".join(text.split())

    return text
```

This helps reduce differences caused only by formatting.

### 3. TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts document text into numerical vectors.

It gives higher importance to words that are useful for distinguishing documents and lower importance to words that occur frequently across documents.

The implementation uses:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

The two OCR documents are converted into TF-IDF vectors before comparison.

### 4. Cosine Similarity

After TF-IDF vectorization, **cosine similarity** is used to measure how similar the two document vectors are.

The implementation uses:

```python
from sklearn.metrics.pairwise import cosine_similarity
```

The similarity value ranges from:

```text
0 → Completely different
1 → Identical text representation
```

The project converts this value into a percentage:

```python
similarity * 100
```

For example:

```text
0.85 → 85%
0.95 → 95%
1.00 → 100%
```

### 5. Document Similarity Result

The current implementation prints the compared document IDs and their TF-IDF similarity score.

Example output format:

```text
--------------------------------
DOCUMENT SIMILARITY RESULT
--------------------------------
Test document: <test_document_id>
Training document: <training_document_id>

TF-IDF Similarity: XX.XX%
```

### Files Added for Module 3

```text
download_documents_dataset.py
load_documents_ocr.py
prepare_documents_sample.py
document_detection.py
```

#### `download_documents_dataset.py`

Loads the RVL-CDIP duplicate metadata dataset and checks the available duplicate information.

#### `load_documents_ocr.py`

Retrieves OCR text for a document using its document ID.

#### `prepare_documents_sample.py`

Selects a small sample containing:

- Known duplicate documents
- Non-duplicate documents

This avoids downloading the complete RVL-CDIP dataset.

#### `document_detection.py`

Contains the main document similarity logic using:

- OCR text
- Text preprocessing
- TF-IDF
- Cosine similarity

### How to Run Module 3

Install the required dependencies first:

```bash
pip install datasets scikit-learn
```

Then run:

```bash
python document_detection.py
```

The script retrieves a known duplicate document pair, loads their OCR text, and calculates their TF-IDF cosine similarity.

### Module 3 Objective

The objective of Module 3 is to detect documents that contain highly similar textual information even when the underlying files are not byte-for-byte identical.

This module will later be extended to compare larger document sets and generate a structured similarity report.

### Current Status

- [x] Load duplicate document metadata
- [x] Retrieve OCR text
- [x] Preprocess OCR text
- [x] TF-IDF vectorization
- [x] Cosine similarity calculation
- [x] Calculate similarity percentage
- [ ] Test duplicate and non-duplicate document pairs
- [ ] Define similarity threshold
- [ ] Generate document similarity CSV report
- [ ] Integrate with Modules 1 and 2

### Repository
GitHub: https://github.com/anshika-63/AI-Duplicate-Detection
