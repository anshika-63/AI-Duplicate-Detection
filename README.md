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

### Repository
GitHub: https://github.com/anshika-63/AI-Duplicate-Detection
