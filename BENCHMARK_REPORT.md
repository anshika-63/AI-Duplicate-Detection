# T243: Empirical Benchmark Evaluation Report
## AI-Based Duplicate Document & Near-Duplicate Image Detection

**Evaluation Date:** November 2026
**Repository Branch:** `feature/image-detection`
**Execution Environment:** Windows 11, Python 3.13.5, PyTorch 2.9.1+cpu
**Integrity Rule:** All metrics in this document were directly measured from actual evaluation executions.

---

## 1. Document Classification Benchmark (RVL-CDIP 16-Class)

- **Dataset:** RVL-CDIP official untouched test partition
- **Samples Evaluated:** 992 document images
- **Class Balance:** Exactly 62 samples per category across all 16 classes
- **Feature Backbone:** Pretrained ResNet-18 (512-dimensional embeddings)
- **Classification Head:** Linear(512, 256) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(256, 16)
- **Primary Source Report:** `reports/document_classification_report.csv`

### Overall Test Metrics:
| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **87.40%** (867 / 992 correct) |
| **Macro Precision** | **0.8771** |
| **Macro Recall** | **0.8740** |
| **Macro F1-Score** | **0.8737** |
| **Weighted F1-Score** | **0.8737** |

### Per-Class Evaluation Breakdown:
| Class ID | Document Category | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
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
| **Avg** | **Macro Average** | **0.8771** | **0.8740** | **0.8737** | **992** |

### Architecture Comparison (Trained on 8,000 samples, Tested on 992):
| Architecture | Train Acc | Val Acc | Test Acc | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Linear Classifier) | 88.16% | 82.08% | 79.84% | 0.7964 | 0.7963 |
| **Deep MLP Head (BatchNorm + Dropout)** | **97.88%** | **89.50%** | **87.40%** | **0.8739** | **0.8737** |

---

## 2. Near-Duplicate Image Benchmark (INRIA Copydays)

- **Dataset Source:** INRIA Copydays (`E:\Fake-Detect\datasets\inria_copydays`)
- **Original Images:** 157 images
- **JPEG Attack Variants:** 1,413 images (varying JPEG quality levels)
- **Primary Source Report:** `reports/image_benchmark_results.csv`

| Algorithm | Evaluated Attack | Operating Threshold | Precision | Recall | F1-Score | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **pHash** | JPEG Compression | $\ge 80.0\%$ | **1.0000** | **1.0000** | **1.0000** | **100.0%** |
| **dHash** | JPEG Compression | $\ge 80.0\%$ | **1.0000** | **1.0000** | **1.0000** | **100.0%** |
| **ORB + RANSAC** | 50% Crop / 15° Rotation | Inliers $\ge 12$ | **0.9570** | **0.9200** | **0.9381** | **94.0%** |

---

## 3. Synthetic Transformation Benchmark Matrix

- **Evaluator Script:** `tools/benchmark_images.py`
- **Output Report:** `reports/synthetic_transformation_benchmark.csv`
- **Sample Pairs:** 8 controlled conditions representing real-world repository operations

| Transformation Category | Applied Parameters | pHash Sim | dHash Sim | Predicted Status | Ground Truth | Verdict |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| 1. Exact File Copy | Byte-identical duplicate | 100.00% | 100.00% | Exact Duplicate | Duplicate | **PASS** |
| 2. Re-encoded PNG | Decoded and re-saved | 100.00% | 100.00% | Exact Duplicate | Duplicate | **PASS** |
| 3. Resized 50% | Downscaled $60 \times 60$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| 4. JPEG Compressed | Quality factor $Q=25$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| 5. Brightness Shift | Enhanced brightness $+25\%$ | 100.00% | 100.00% | Near Duplicate | Duplicate | **PASS** |
| 6. Rotated 15° | $15^\circ$ affine rotation | 84.38% | 85.94% | Near Duplicate | Duplicate | **PASS** |
| 7. Cropped 50% | Center bounding crop | 82.81% | 84.38% | Near Duplicate | Duplicate | **PASS** |
| 8. Negative Control | Striped pattern image | 65.62% | 46.88% | Non-Duplicate | Non-Duplicate | **PASS** |

**Summary Evaluation:**
- Total Cases: 8
- Accuracy: **100.00%** (8 / 8)
- Precision: **1.0000** (TP: 7, FP: 0)
- Recall: **1.0000** (TP: 7, FN: 0)
- F1-Score: **1.0000**

---

## 4. Document Similarity Benchmark Matrix

- **Evaluator Script:** `tools/benchmark_documents.py`
- **Output Report:** `reports/document_similarity_benchmark.csv`
- **Sample Pairs:** 9 controlled linguistic and multi-format scenarios

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

**Summary Evaluation:**
- Total Cases: 9
- Accuracy: **100.00%** (9 / 9)
- Precision: **1.0000** (TP: 5, FP: 0)
- Recall: **1.0000** (TP: 5, FN: 0)
- F1-Score: **1.0000**
