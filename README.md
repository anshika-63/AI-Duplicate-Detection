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

### Project Structure
```text
AI-Duplicate-Detection/
├── dataset/
├── main.py
├── duplicate_report.csv
├── README.md
└── .gitignore
```

### How to Run

1. Clone the repository.
2. Open the project folder in VS Code.
3. Ensure Python is installed.
4. Place sample files inside the `dataset` folder.
5. Run the following command:

```bash
python main.py
```

### Expected Output
The program identifies files having identical SHA-256 hashes and generates a `duplicate_report.csv` file.

### Current Development Status
**Completed:** Module 1 – Exact Duplicate Detection using SHA-256 hashing.

**Upcoming:** Near-duplicate image and document detection.

### Repository
GitHub: https://github.com/anshika-63/AI-Duplicate-Detection