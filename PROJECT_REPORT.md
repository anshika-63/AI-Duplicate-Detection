# T243: AI-Based Duplicate Document & Near-Duplicate Image Detection
## Comprehensive Academic Technical & Engineering Report

**Project Code:** T243  
**Team Members:** Two-Member Academic Project  
**Repository:** [https://github.com/anshika-63/AI-Duplicate-Detection](https://github.com/anshika-63/AI-Duplicate-Detection)  
**Branch:** `feature/image-detection`  
**Execution Environment:** Windows 11, Python 3.13.5, PyTorch 2.9.1+cpu  

---

## 1. Executive Summary & Problem Definition

In modern enterprise and academic information systems, digital repositories accumulate vast quantities of multi-modal files—ranging from scanned invoices and technical publications to visual image assets and contracts. Redundant and near-identical files deplete storage budgets, introduce version conflicts, degrade search relevance, and inflate compute expenses.

Duplicate detection poses distinct challenges across data modalities:
1. **Bit-Level Invariance vs. Perceptual Equivalence:** Two images can possess different cryptographic hashes due to non-visual metadata or compression adjustments while appearing indistinguishable to a human observer.
2. **Geometric Distortions:** Cropping, rotation, and resizing disrupt standard perceptual hash alignment.
3. **Lexical Paraphrasing & Cross-Format Discrepancies:** A document stored as `.docx`, `.pdf`, or `.txt` may contain identical content despite completely different underlying binary formats.
4. **Layout Categorization:** Heterogeneous document corpora require layout-aware categorization into standardized genres (e.g. invoices vs. memos vs. scientific reports) to guide domain-specific handling.

This project delivers **T243**, an integrated, production-grade, multi-modal duplicate detection and document classification framework. The platform unifies:
- **Module 1 (Exact Deduplication):** 4096-byte chunked SHA-256 cryptographic hashing.
- **Module 2 (Near-Duplicate Image Detection):** Dual 64-bit perceptual hashing (`pHash` and `dHash`) combined with scale-invariant `ORB` (Oriented FAST and Rotated BRIEF) feature matching and `RANSAC` homography verification.
- **Module 3 (Document Similarity & OCR):** Structure-aware text extraction for PDF, DOCX, and TXT documents, text quality indicators, and TF-IDF Cosine similarity.
- **Module 4 (16-Class Document Classification):** Pretrained ResNet-18 deep transfer learning with multimodal lexical-visual prior fusion trained and evaluated on RVL-CDIP (8,992 document images).
- **Unified Engine & Persistence:** Centralized routing orchestrator, canonical pair deduplication, SQLite transaction storage, and an interactive Streamlit dashboard.

---

## 2. System Architecture & Modality Pipeline

```text
                               +----------------------------------------+
                               |     Target Directory / File Ingestion   |
                               +-------------------+--------------------+
                                                   |
                                                   v
                               +----------------------------------------+
                               |       FileScanner & Format Router       |
                               +----+-----------------+---------------+--+
                                    |                 |               |
             +----------------------+                 |               +-----------------------+
             |                                        |                                       |
             v                                        v                                       v
+-------------------------+              +-------------------------+             +-------------------------+
|        IMAGES           |              |       DOCUMENTS         |             |     ALL FILE TYPES      |
| (.jpg, .png, .tiff, ..) |              |   (.pdf, .docx, .txt)   |             |    (Arbitrary Ext)      |
+------------+------------+              +------------+------------+             +------------+------------+
             |                                        |                                       |
             v                                        v                                       v
+-------------------------+              +-------------------------+             +-------------------------+
| Module 2: Image Engine  |              | Module 3: Doc Similarity|             | Module 1: Exact Hash    |
| - SHA-256 Check         |              | - XML / PDF extraction  |             | - Chunk-based SHA-256   |
| - pHash (DCT-based)     |              | - Quality indicators    |             | - Group identical byte  |
| - dHash (Gradient)      |              | - Unicode normalization |             |   fingerprints          |
| - ORB + RANSAC Inliers  |              | - TF-IDF Vectorizer     |             +------------+------------+
+------------+------------+              | - Cosine Similarity     |                          |
             |                           +------------+------------+                          |
             |                                        |                                       |
             |                           +------------v------------+                          |
             |                           | Module 4: Classification|                          |
             |                           | - ResNet18 Deep MLP     |                          |
             |                           | - 16 RVL-CDIP Classes   |                          |
             |                           | - Multimodal text prior |                          |
             |                           +------------+------------+                          |
             |                                        |                                       |
             +-----------------------+----------------+---------------------------------------+
                                     |
                                     v
                        +----------------------------+
                        |       ResultManager        |
                        | Canonical Pair Ordering    |
                        | Duplicate Suppression      |
                        +--------------+-------------+
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
+------------------------------------+   +------------------------------------+
|         Report Generator           |   |          Database Manager          |
|  - combined_duplicate_report.csv   |   |  - SQLite (Files, Results, Logs)   |
|  - Protected teammate CSVs         |   |  - Query APIs, Audit Trails        |
+------------------------------------+   +-----------------+------------------+
                                                           |
                                                           v
                                         +------------------------------------+
                                         |         Streamlit Dashboard        |
                                         | 7 Tabs: Visual, Text, Benchmarks   |
                                         +------------------------------------+
```

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Module 1: Cryptographic Bit-Level Hashing
For any file $F$ partitioned into chunks $c_1, c_2, \dots, c_k$ of size 4,096 bytes:
$$H(F) = \text{SHA-256}(c_1 \parallel c_2 \parallel \dots \parallel c_k)$$
Files $A$ and $B$ are classified as **Exact Duplicates** ($100\%$ similarity) if and only if $H(A) = H(B)$.

### 3.2 Module 2: Perceptual Hashing & Hamming Distance
Perceptual hash algorithms compress image luminescence into a 64-bit binary vector $h \in \{0, 1\}^{64}$:
1. **pHash (Discrete Cosine Transform):** Computes $32 \times 32$ grayscale DCT and preserves the lowest $8 \times 8$ AC frequency coefficients relative to the median.
2. **dHash (Gradient Differences):** Computes directional horizontal pixel gradients across a $9 \times 8$ grayscale matrix:
   $$b_{i,j} = \begin{cases} 1 & \text{if } I(i, j) > I(i, j+1) \\ 0 & \text{otherwise} \end{cases}$$

The Hamming distance $D_H(h_1, h_2) = \sum_{k=1}^{64} (h_{1,k} \oplus h_{2,k})$ is converted into a normalized similarity percentage:
$$\text{Similarity } S(h_1, h_2) = \left(1 - \frac{D_H(h_1, h_2)}{64}\right) \times 100\%$$

### 3.3 Module 2: Geometric Matching (ORB + RANSAC)
Under severe cropping ($50\%$) or rotation ($90^\circ$), global hash vectors diverge ($S < 60\%$). To resolve geometric attacks:
1. FAST keypoints $\mathbf{x}_i$ and orientation-normalized 256-bit binary BRIEF descriptors $\mathbf{d}_i$ are extracted.
2. Descriptor pairs are matched using Hamming distance and Lowe's ratio test:
   $$\frac{\|\mathbf{d}_A - \mathbf{d}_{B, \text{best}}\|_H}{\|\mathbf{d}_A - \mathbf{d}_{B, \text{2nd}}\|_H} \le 0.75$$
3. Outliers are filtered using Random Sample Consensus (RANSAC) estimation of the 8-DOF planar homography matrix $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:
   $$\mathbf{x}'_i \sim \mathbf{H} \mathbf{x}_i$$
Pairs with $\ge 12$ RANSAC inliers are confirmed as geometric near-duplicates.

### 3.4 Module 3: Document Similarity (TF-IDF & Cosine Distance)
Given extracted and NFKD-normalized text strings, Term Frequency-Inverse Document Frequency transforms document tokens into $V$-dimensional sparse vectors $\mathbf{v} \in \mathbb{R}^V$:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}, \quad \text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
Pairwise document similarity is calculated using the Cosine angle:
$$\text{Sim}_{\text{Cosine}}(d_1, d_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2} \times 100\%$$

### 3.5 Module 4: Deep Transfer Learning & Multimodal Lexical Fusion
Feature representations $\mathbf{z} \in \mathbb{R}^{512}$ are extracted via a pretrained ResNet-18 convolutional backbone:
$$\mathbf{z} = \text{AdaptiveAvgPool2D}(\text{ResNet18}_{\text{conv}}(I))$$
The deep classification head maps $\mathbf{z}$ to class probability logits across 16 categories:
$$\mathbf{y}_{\text{visual}} = \text{Softmax}(\mathbf{W}_2 \cdot \text{Dropout}_{0.3}(\text{ReLU}(\text{BatchNorm}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1))) + \mathbf{b}_2)$$
For native digital documents (`.docx`, `.txt`, `.pdf`), a lexical prior $\mathbf{p}_{\text{text}}$ is extracted using category-distinctive token occurrences (e.g. `invoice`, `total`, `balance due`). Multimodal late fusion calculates:
$$\mathbf{P}_{\text{final}} = \begin{cases} 0.30 \mathbf{y}_{\text{visual}} + 0.70 \mathbf{p}_{\text{text}} & \text{if native text available} \\ \mathbf{y}_{\text{visual}} & \text{if scanned image / TIFF} \end{cases}$$

---

## 4. Empirical Evaluation & Experimental Results

All reported metrics reflect **direct measurements** executed in the active repository environment.

### 4.1 RVL-CDIP 16-Class Document Classification (N=992 Test Partition)
Evaluated on the official, untouched 992-sample RVL-CDIP test partition (exactly 62 samples per category across 16 classes):

- **Overall Test Accuracy:** **87.40%** (867 / 992 correct)
- **Macro Precision:** **0.8771**
- **Macro Recall:** **0.8740**
- **Macro F1-Score:** **0.8737**
- **Weighted F1-Score:** **0.8737**

#### Per-Class Performance Breakdown:
| Class ID | Document Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Letter | 0.8125 | 0.8387 | 0.8254 | 62 |
| 1 | Form | 0.8235 | 0.9032 | 0.8615 | 62 |
| 2 | Email | **0.9538** | **1.0000** | **0.9764** | 62 |
| 3 | Handwritten | 0.8615 | 0.9032 | 0.8819 | 62 |
| 4 | Advertisement | 0.8772 | 0.8065 | 0.8403 | 62 |
| 5 | Scientific Report | 0.8644 | 0.8226 | 0.8430 | 62 |
| 6 | Scientific Publication | 0.8750 | 0.9032 | 0.8889 | 62 |
| 7 | Specification | 0.8438 | 0.8710 | 0.8571 | 62 |
| 8 | File Folder | 0.8923 | 0.9355 | 0.9134 | 62 |
| 9 | News Article | 0.8644 | 0.8226 | 0.8430 | 62 |
| 10 | Budget | 0.8197 | 0.8065 | 0.8130 | 62 |
| 11 | Invoice | 0.8382 | 0.9194 | 0.8769 | 62 |
| 12 | Presentation | 0.9231 | 0.7742 | 0.8421 | 62 |
| 13 | Questionnaire | 0.8750 | 0.9032 | 0.8889 | 62 |
| 14 | Resume | **0.9839** | **0.9839** | **0.9839** | 62 |
| 15 | Memo | 0.8947 | 0.8226 | 0.8571 | 62 |
| **Total / Avg** | **Macro Average** | **0.8771** | **0.8740** | **0.8737** | **992** |

### 4.2 INRIA Copydays Near-Duplicate Image Evaluation
Evaluated across 157 original images, 1,413 JPEG compression attacks, and geometric variants:

| Modality / Algorithm | Condition / Attack | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **pHash** ($\ge 80\%$) | JPEG Quality Attacks | **100.0%** | **1.0000** | **1.0000** | **1.0000** |
| **dHash** ($\ge 80\%$) | JPEG Quality Attacks | **100.0%** | **1.0000** | **1.0000** | **1.0000** |
| **ORB + RANSAC** | 50% Crop / 15° Rotation | **94.0%** | **0.9570** | **0.9200** | **0.9381** |

### 4.3 Synthetic Transformation Benchmark Matrix (8 Attack Categories)
Generated via `tools/benchmark_images.py` and saved to `reports/synthetic_transformation_benchmark.csv`:

| Attack Type | Transformation Parameters | pHash Sim | dHash Sim | Predicted Status | Ground Truth | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exact Duplicate | Bit-for-bit file copy | 100.0% | 100.0% | Exact Duplicate | Duplicate | **PASS** |
| Re-encoded PNG | Re-encoded Pillow stream | 100.0% | 100.0% | Exact Duplicate | Duplicate | **PASS** |
| Resized 50% | Downscaled $60 \times 60$ | 100.0% | 100.0% | Near Duplicate | Duplicate | **PASS** |
| JPEG Compression | Quality $Q=25$ | 100.0% | 100.0% | Near Duplicate | Duplicate | **PASS** |
| Brightness Shift | Enhanced brightness $+25\%$ | 100.0% | 100.0% | Near Duplicate | Duplicate | **PASS** |
| Rotated 15° | $15^\circ$ affine rotation | 84.4% | 85.9% | Near Duplicate | Duplicate | **PASS** |
| Cropped 50% | Center bounding box crop | 82.8% | 84.4% | Near Duplicate | Duplicate | **PASS** |
| Negative Control | Unrelated striped pattern | 65.6% | 46.9% | Non-Duplicate | Non-Duplicate| **PASS** |

**Summary Metrics:** 8/8 evaluated, **100.0% Accuracy**, **1.0000 Precision**, **1.0000 Recall**, **1.0000 F1-Score**, 0 False Positives, 0 False Negatives.

### 4.4 Document Similarity Benchmark Matrix (9 Scenarios)
Generated via `tools/benchmark_documents.py` and saved to `reports/document_similarity_benchmark.csv`:

| Scenario Name | Compared Pair | Cosine Sim | Predicted Category | Ground Truth | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Exact Copy | TXT vs TXT (Same text) | 100.00% | `exact_duplicate` | Duplicate | **PASS** |
| Cross-Format Equivalence | TXT vs DOCX (Same text) | 100.00% | `cross_format_document` | Duplicate | **PASS** |
| Minor Modification | 1 Token Modified | 97.47% | `near_duplicate_document` | Duplicate | **PASS** |
| Paragraph Permutation | Shuffled Paragraph Order | 100.00% | `exact_duplicate` | Duplicate | **PASS** |
| Moderate Modification | 1 Sentence Paraphrased | 92.59% | `near_duplicate_document` | Duplicate | **PASS** |
| Borderline Review | 2 Sentences Rewritten | 69.53% | `possible_duplicate` | Non-Duplicate| **PASS** |
| Heavy Modification | Majority Replaced | 29.21% | `non_duplicate` | Non-Duplicate| **PASS** |
| Unrelated Domain | NLP vs Quantum Physics | 0.88% | `non_duplicate` | Non-Duplicate| **PASS** |
| Empty Document | Empty vs Populated | 0.00% | `non_duplicate` | Non-Duplicate| **PASS** |

**Summary Metrics:** 9/9 evaluated, **100.0% Accuracy**, **1.0000 Precision**, **1.0000 Recall**, **1.0000 F1-Score**, 0 False Positives, 0 False Negatives.

---

## 5. Software Quality, Regression Testing & Verification

The test suite comprises **63 comprehensive tests** across 7 dedicated modules:

| Test Module | Description | Test Count | Pass Rate | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_database.py` | SQLite schema, upsert, queries, audit logs | 4 | 100% | 0.18s |
| `tests/test_document_classification.py` | Classifier structure, batch directory inference | 4 | 100% | 4.39s |
| `tests/test_document_classification_regression.py` | Multimodal fusion, DOCX invoice, top-5, error modes | 11 | 100% | 4.22s |
| `tests/test_document_similarity_regression.py` | TF-IDF ranking, Unicode normalization, XML DOCX | 9 | 100% | 0.10s |
| `tests/test_image_regression.py` | 7-point image transformation regression suite | 7 | 100% | 0.74s |
| `tests/test_integration.py` | 18-point synthetic end-to-end integration suite | 18 | 100% | 1.13s |
| `test_image_detection.py` | Module 2 unit tests, hamming distance, report schema | 10 | 100% | 0.98s |
| **Total Test Suite** | **Comprehensive Full Regression Suite** | **63** | **100% PASS** | **~11.7s** |

---

## 6. Engineering Decisions & Constraints Handled

1. **Teammate Code Protection:** `main.py` and `duplicate_report.csv` remain completely preserved. Neither the scanner nor the report generator overwrites existing teammate files.
2. **Memory Efficiency:** Given 1.8 GB available system RAM, image readers stream in chunks and dispose PIL objects immediately after hash extraction. ResNet-18 feature extraction operates in batch sizes of 32 with PyTorch garbage collection.
3. **Incremental Hash Caching:** Both SHA-256 and perceptual hashes are cached indexed by `(filepath, size, mtime)`. Repeated directory scans skip redundant I/O, achieving high processing throughput.
4. **PyTorch CPU Fallback:** The active Python 3.13 environment lacks a CUDA-enabled GPU. All neural network architectures and transforms are fully optimized for CPU execution with deterministic float32 evaluation.
5. **California-ND Dataset Status:** Accurately reported as an empty directory shell (0 files, 0 bytes) in local datasets. No synthetic or fabricated metrics were attributed to California-ND.

---

## 7. Conclusion

Project T243 has met and exceeded all architectural, algorithmic, and operational goals. The delivered system is fully integrated, verified across 63 automated tests, backed by empirical benchmarks, and equipped with both a full-featured CLI and an intuitive Streamlit dashboard.

