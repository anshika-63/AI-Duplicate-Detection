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
