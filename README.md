# AI-Based Duplicate Document & Near-Duplicate Image Detection

## Module 1: Exact Duplicate Detection Using SHA-256

### Project Overview
This project aims to identify duplicate documents and images using Python. The first module implements exact duplicate file detection using the SHA-256 hashing algorithm.

### Features
- Scans files from the dataset folder.
- Calculates SHA-256 hashes for files.
- Groups files with identical hashes.
- Identifies exact duplicate files.
- Generates a CSV report containing duplicate file paths and hashes.

### Technologies Used
- Python
- hashlib – SHA-256 hashing
- csv – CSV report generation
- pathlib – File and folder handling

---

## Module 2: Near-Duplicate Image Detection Using Perceptual Hashing (pHash & dHash)

### Project Overview
The second module implements perceptual image hashing to detect identical, resized, compressed, or brightness-adjusted duplicate and near-duplicate images.

### Features
- Loads images securely using Pillow (supports JPG, PNG, WEBP, BMP, TIFF).
- Computes 64-bit Perceptual Hash (pHash) and Difference Hash (dHash) using ImageHash.
- Computes Hamming distance and similarity percentage: `Similarity (%) = (1 - Hamming Distance / 64) * 100`.
- Categorizes pairs into `Exact Duplicate`, `Near Duplicate`, `Potential Near-Duplicate`, and `Non-Duplicate`.
- Scans directories recursively without loading full-resolution images into memory.
- Safely handles and logs corrupted and unsupported image files without crashing.
- Generates `image_duplicate_report.csv` without modifying or overwriting `duplicate_report.csv`.

### Technologies Used
- Python
- Pillow – Safe image loading and verification
- ImageHash – pHash and dHash generation
- csv – Image duplicate report generation
- pathlib – File and folder handling

---

### Project Structure
```text
AI-Duplicate-Detection/
├── dataset/
├── main.py
├── duplicate_report.csv
├── image_detection.py
├── image_duplicate_report.csv
├── test_image_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

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
