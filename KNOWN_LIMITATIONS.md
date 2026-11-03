# T243: Known Limitations & Environment Constraints
## AI-Based Duplicate Document & Near-Duplicate Image Detection

**Document Purpose:** In accordance with rigorous academic and scientific standards, this document explicitly discloses known environment constraints, hardware parameters, dataset anomalies, and operational boundaries.

---

## 1. Dataset Transparency: California-ND

- **Observed State in Repository:** The directory `E:\Fake-Detect\datasets\california-nd` is an empty folder shell containing zero usable media files (0 bytes, 0 photos).
- **Handling & Policy:**
  - California-ND is **NOT** evaluated in this project.
  - No synthetic scores, simulated counts, or fabricated performance numbers are attributed to California-ND.
  - All empirical image benchmarks are strictly conducted on **INRIA Copydays** (157 originals, 1,413 JPEG quality attacks) and our controlled **8-category synthetic transformation matrix**.

---

## 2. Hardware Constraints & PyTorch CPU Execution

- **Host Environment:** Windows 11, Intel/AMD multi-core host.
- **CUDA Availability:** `torch.cuda.is_available()` returns `False`. PyTorch is running as `2.9.1+cpu`.
- **System Memory:** Available system RAM is approximately $1.8\text{ GB}$ out of $15.3\text{ GB}$.
- **Design Mitigations Implemented:**
  - **Chunked Feature Extraction:** Documents are pre-extracted in memory-efficient batches of 32 samples.
  - **Immediate Image Teardown:** Pillow images are closed immediately after perceptual hash or tensor conversion.
  - **Cached Embeddings:** 512-dimensional ResNet-18 features are cached in `cache/rvlcdip_resnet18_features.npz` to avoid repeated forward passes during classifier training.
  - **Inference Latency:** Classification on CPU achieves ~50-80 ms per document, suitable for interactive use and batch processing without requiring an external GPU.

---

## 3. Optical Character Recognition (OCR) Dependency

- **Engine:** `pytesseract` adapter for scanned documents and TIFF images.
- **Environment Status:** Tesseract OCR binary is not installed in the default system PATH.
- **Graceful Degradation:**
  - When processing native digital documents (`.txt`, `.docx`, text-based `.pdf`), text extraction executes through native standard-library XML and `pdfplumber` parsers with 100% fidelity.
  - If a scanned document or image-only PDF is processed with OCR enabled and Tesseract is unavailable, the pipeline logs an informative warning and degrades gracefully without terminating the execution.
  - To enable full OCR for scanned paper documents, users can install Google Tesseract OCR and set the binary path in their environment.

---

## 4. Algorithmic Boundaries

1. **Perceptual Hashing vs. Cropping:**
   - Standard 64-bit `pHash` and `dHash` algorithms are invariant to compression, format re-encoding, and moderate resizing ($100\%$ accuracy).
   - However, when images are cropped beyond $50\%$ or rotated by arbitrary angles ($> 30^\circ$), global hash vectors diverge.
   - For geometric attacks, users must enable the `--orb` flag (`UnifiedDetector(..., enable_orb=True)`) to invoke keypoint descriptor matching with RANSAC homography.
2. **Bag-of-Words vs. Semantic Paraphrasing:**
   - Module 3 utilizes TF-IDF with Cosine Similarity, which excels at verbatim and near-verbatim duplicate text, cross-format equivalence, and paragraph permutation ($100\%$ accuracy).
   - Highly abstract, conceptual paraphrasing without shared vocabulary falls below the duplicate threshold ($< 75\%$) and is classified as `possible_duplicate` or `non_duplicate`.
