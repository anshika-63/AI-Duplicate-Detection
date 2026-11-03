# T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection
## Project Completion & Engineering Execution Plan

**Author / Maintainer:** T243 Engineering Team
**Git Branch:** `feature/image-detection`
**Execution Environment:** Windows 11, Python 3.13, PyTorch 2.9.1 (CPU / CUDA compatible), RTX 2050 (4 GB VRAM), 16 GB RAM
**Dataset Directory:** `E:\Fake-Detect\datasets`

---

## Stage Progression & Status Tracker

| Stage | Name | Status | Key Deliverables & Milestones |
| :---: | :--- | :---: | :--- |
| **0** | Initial Repository Audit & Baseline Testing | **COMPLETED** | Inventory verified, baseline test suite (52 tests) recorded, environment audited. |
| **1** | Document Classification Validation & Improvement | **COMPLETED** | Per-class official test set evaluation (992 test samples: 87.40% accuracy, 0.8737 weighted F1), confusion matrix, top-5 predictions, multimodal fusion verification, 11 regression tests passing. |
| **2** | Near-Duplicate Image Detection Improvement | **COMPLETED** | Configurable thresholds, multi-criteria decision fusion, robust benchmarking (INRIA Copydays JPEG attacks: 100% pHash/dHash accuracy; 8-category transformation benchmark: 100% accuracy, 0 FP, 0 FN across all transformations). |
| **3** | Document Similarity and OCR Improvement | **IN PROGRESS** | Multi-format text extraction (PDF, DOCX, TXT), table/paragraph preservation, OCR fallback, TF-IDF cosine calibration, cross-format evaluation. |
| **4** | Unified Scanner, Performance & Scalability | **PLANNED** | Robust recursive scanning, incremental caching, memory optimization, error isolation, CLI multi-mode options. |
| **5** | Streamlit Dashboard & SQLite Database | **PLANNED** | 6-page interactive UI, visual diffs, text diffs, live database querying and filtering, export capabilities. |
| **6** | Final Testing, Documentation & Project Finalization | **PLANNED** | Full test discovery, README.md, PROJECT_REPORT.md, TESTING_REPORT.md, BENCHMARK_REPORT.md, KNOWN_LIMITATIONS.md, GitHub clean state. |

---

## Detailed Stage Plans & Acceptance Criteria

### Stage 1: Document Classification Validation and Improvement
- **Objective:** Ensure the 16-class document classification module is reliable, measurable, and suitable for academic evaluation.
- **Tasks:**
  1. Inspect complete preprocessing, training, inference, and evaluation pipeline.
  2. Verify class-to-index mapping consistency (`CLASS_NAMES` vs Parquet metadata).
  3. Validate official untouched 992-image test partition from `test-00000-of-00001.parquet`.
  4. Generate full evaluation metrics: Accuracy, Macro/Weighted Precision, Recall, F1-score, per-class support, confusion matrix (`reports/confusion_matrix.png`), and classification report (`reports/document_classification_report.csv`).
  5. Provide calibrated confidence scores, predicted document category, and top-3 / top-5 candidate rankings.
  6. Validate all 16 genuine categories using dataset samples.
  7. Maintain strict separation between synthetic DOCX/PDF documents and official RVL-CDIP evaluation.
  8. Verify multimodal lexical prior fusion for native text documents.
  9. Add regression tests covering class mapping, determinism, missing/corrupted files, and unsupported formats.
- **Acceptance Criteria:** 16 classes reliably predicted, 0 class index mismatches, official evaluation reproducible on genuine data, synthetic DOCX invoices classified as `invoice` without compromising official image accuracy.

---

### Stage 2: Near-Duplicate Image Detection Improvement
- **Objective:** Deliver robust image comparison that detects transformed copies while rejecting unrelated similar-background images.
- **Tasks:**
  1. Enforce strict detection hierarchy:
     - Priority 1: Byte-identical files (SHA-256 match -> 100% exact duplicate; immutable).
     - Priority 2: Visually identical / re-encoded images (zero pHash/dHash distance).
     - Priority 3: Near-duplicate images (both hashes >= threshold, or geometric ORB match).
     - Priority 4: Non-duplicate images (unrelated, or gradient divergence).
  2. Eliminate dHash gradient false positives through cross-hash divergence checks.
  3. Feature matching with ORB descriptors, Lowe's ratio test (0.75), and RANSAC homography geometric verification.
  4. Make similarity thresholds configurable (CLI, function parameters, UI).
  5. Benchmark on INRIA Copydays (157 originals, 1,413 JPEG quality attacks) and synthetic crop/rotation benchmarks.
  6. Handle corrupted, grayscale, truncated, and unsupported image files safely.
- **Acceptance Criteria:** 100% precision and recall on JPEG attacks at optimal threshold, ORB recall >= 90% on 50-70% crops/rotations, unrelated similar-background images rejected as non-duplicates.

---

### Stage 3: Document Similarity and OCR Improvement
- **Objective:** Reliable cross-format text comparison across PDF, DOCX, TXT, and scanned inputs.
- **Tasks:**
  1. Table- and paragraph-aware DOCX text extraction via native XML parser and `python-docx`.
  2. Multi-engine PDF extraction (`pdfplumber` with fallback to `PyPDF2` and OCR).
  3. Robust text normalization: Unicode NFKD normalization, whitespace cleanup, ligature resolution.
  4. Optional OCR support for scanned documents via `pytesseract` with graceful fallback when uninstalled.
  5. Calibrate TF-IDF cosine similarity thresholds on labelled test pairs:
     - Exact duplicate: 100.0%
     - Mostly same: >= 85.0%
     - Moderately same: >= 75.0%
     - Lightly same: < 75.0%
  6. Strict monotonic ranking preservation: Mostly same > Moderately same > Lightly same.
- **Acceptance Criteria:** Equivalent text matches across TXT, DOCX, and PDF with 100% similarity; cross-format duplicates detected accurately; graceful degradation when OCR is absent.

---

### Stage 4: Unified Scanner, Performance and Scalability
- **Objective:** Efficient multi-modal directory scanning, incremental processing, and unified reporting.
- **Tasks:**
  1. Standardized input/output interfaces via `DuplicateRecord` and `ScanInventory`.
  2. Memory-efficient chunked SHA-256 hashing (4096-8192 bytes) and on-the-fly image closing.
  3. Incremental scanning / caching support to avoid redundant computations on unchanged files.
  4. Robust handling of nested directories, duplicate filenames, corrupted files, and permission errors.
  5. Multi-modal combined CSV report (`combined_duplicate_report.csv`, 10 columns) and execution summaries.
- **Acceptance Criteria:** Mixed folders with images, PDFs, DOCX, and TXT scanned without crashing, O(N^2) memory bloat, or reverse/self-comparison duplicates.

---

### Stage 5: Streamlit Dashboard and SQLite Database
- **Objective:** Deliver an intuitive, professional, demonstrable web application and robust persistence layer.
- **Tasks:**
  1. 6 functional dashboard tabs:
     - Tab 1: Near-Duplicate Image Detection (side-by-side comparison, difference heatmap, ORB inliers).
     - Tab 2: Document Classification (16-class RVL-CDIP prediction, top-5 confidence ranking).
     - Tab 3: Document Similarity (inline text diff, cross-format comparison, metrics).
     - Tab 4: Unified Scanner (batch folder scan, live threshold controls, CSV export).
     - Tab 5: Benchmark & Evaluation (confusion matrix, training loss curves, algorithm metrics).
     - Tab 6: Database & History (SQLite query interface, scan history, category filters).
  2. SQLite database schema (`files`, `detection_results`, `classification_results`, `scan_history`).
  3. Thread-safe connection handling, transaction rollback, and query filters.
- **Acceptance Criteria:** Complete end-to-end user workflows executable through the browser without CLI commands.

---

### Stage 6: Final Testing, Documentation and Project Finalization
- **Objective:** Prepare project for college demonstration, evaluation, and GitHub submission.
- **Tasks:**
  1. Full test discovery across all unit, regression, and integration suites (50+ tests).
  2. Generate comprehensive documentation:
     - `README.md`: System overview, setup, usage, architecture diagram, benchmark summary.
     - `PROJECT_REPORT.md`: Academic project report with problem statement, methodology, evaluation, and conclusions.
     - `TESTING_REPORT.md`: Exhaustive test suite logs and pass/fail metrics.
     - `BENCHMARK_REPORT.md`: Empirical evaluation numbers on INRIA Copydays, RVL-CDIP, and synthetic attacks.
     - `KNOWN_LIMITATIONS.md`: Transparent analysis of hardware constraints, missing public datasets (California-ND 0 files), and future enhancements.
  3. Verify `.gitignore` excludes large checkpoints, datasets, local caches, and temporary uploads.
- **Acceptance Criteria:** Clean git status, all documentation complete, 100% test pass rate, and ready for evaluation.
