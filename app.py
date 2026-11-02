"""
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection Dashboard
Interactive Web Interface powered by Streamlit

Pages:
1. Overview – System metrics, storage stats, scan audit history.
2. Exact Duplicates – Cryptographic SHA-256 duplicate clusters.
3. Image Detection – Visual side-by-side image comparison with pHash, dHash, and ORB.
4. Document Classification – Interactive 16-class document classifier with confidence scores.
5. Document Similarity – Text extraction, TF-IDF Cosine scores, and diff previews.
6. Reports & Evaluation – Confusion matrix, training curves, benchmark comparisons, CSV downloads.
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Page setup
st.set_page_config(
    page_title="T243 – AI Duplicate Detection & Document Classification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; border-radius: 8px; padding: 15px; border-left: 5px solid #3B82F6; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: #F9FAFB; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #EFF6FF !important; border-bottom: 2px solid #2563EB !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Header
st.markdown('<div class="main-header">T243 – AI Duplicate Detection & Document Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-modal platform for cryptographic hashing, perceptual image fingerprints, ORB feature matching, and RVL-CDIP 16-class classification.</div>', unsafe_allow_html=True)

# Navigation
tabs = st.tabs([
    "📊 Overview",
    "🔒 Exact Duplicates",
    "🖼️ Image Detection",
    "📑 Document Classification",
    "📝 Document Similarity",
    "📈 Reports & Evaluation",
])

# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================
with tabs[0]:
    st.subheader("System Overview & Live Audit")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Detection Modalities", "3 Types", "SHA-256 / Hash / NLP")
    with col2:
        st.metric("Document Classes", "16 Classes", "RVL-CDIP Benchmark")
    with col3:
        st.metric("INRIA Images", "1,799 Files", "Original + Attacks")
    with col4:
        st.metric("Parquet Training Size", "8,992 Docs", "Balanced 500/class")
    with col5:
        st.metric("Target Storage Budget", "10 GB Limit", "Optimal VRAM/RAM")

    st.markdown("---")
    st.markdown("#### Quick Directory Scan")

    scan_folder = st.text_input("Target Directory to Scan", value="dataset")
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_scan = st.button("🚀 Run Unified Scan", use_container_width=True)

    if run_scan:
        with st.spinner(f"Scanning directory '{scan_folder}'..."):
            try:
                from unified_detector import UnifiedDetector
                detector = UnifiedDetector(scan_folder)
                records, summary, _ = detector.run(run_exact=True, run_images=True, run_documents=True)
                st.success(f"Scan Completed in {summary.scan_duration_seconds:.2f}s! Found {summary.total_detected_pairs} duplicate pairs.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files Scanned", summary.total_files_scanned)
                c2.metric("Exact Pairs", summary.exact_duplicate_pairs)
                c3.metric("Near Images", summary.near_duplicate_image_pairs)
                c4.metric("Near Documents", summary.near_duplicate_document_pairs)

                if records:
                    df = pd.DataFrame([r.to_csv_row() for r in records], columns=[
                        "File 1", "File 2", "Name 1", "Name 2", "Type", "Category", "Score", "Algorithm", "Details", "Timestamp"
                    ])
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Scan failed: {e}")

# ==========================================================
# TAB 2: EXACT DUPLICATES
# ==========================================================
with tabs[1]:
    st.subheader("Module 1: Exact Cryptographic Duplicate Detection (SHA-256)")
    st.write("Identifies bit-for-bit identical files across arbitrary extensions using 4096-byte chunked SHA-256 streaming.")

    if Path("duplicate_report.csv").exists():
        df_exact = pd.read_csv("duplicate_report.csv")
        st.dataframe(df_exact, use_container_width=True)
    else:
        st.info("No 'duplicate_report.csv' found. Run a scan from the Overview tab or execute 'python main.py'.")

# ==========================================================
# TAB 3: IMAGE DETECTION
# ==========================================================
with tabs[2]:
    st.subheader("Module 2: Perceptual Hashing & ORB Feature Matching")
    st.write("Detects identical, compressed, resized, and rotated/cropped images.")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        img_upload_1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], key="img1")
    with col_img2:
        img_upload_2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], key="img2")

    if img_upload_1 and img_upload_2:
        pil1 = Image.open(img_upload_1).convert("RGB")
        pil2 = Image.open(img_upload_2).convert("RGB")

        c_view1, c_view2 = st.columns(2)
        with c_view1:
            st.image(pil1, caption=img_upload_1.name, use_container_width=True)
        with c_view2:
            st.image(pil2, caption=img_upload_2.name, use_container_width=True)

        if st.button("Compare Images", key="btn_cmp_img"):
            import hashlib
            from image_detection import compute_phash, compute_dhash, calculate_hamming_distance, calculate_similarity
            from orb_detection import compare_images_orb

            bytes1 = img_upload_1.getvalue()
            bytes2 = img_upload_2.getvalue()
            sha1 = hashlib.sha256(bytes1).hexdigest()
            sha2 = hashlib.sha256(bytes2).hexdigest()
            is_byte_identical = (sha1 == sha2)

            p1, p2 = compute_phash(pil1), compute_phash(pil2)
            d1, d2 = compute_dhash(pil1), compute_dhash(pil2)
            p_dist = calculate_hamming_distance(p1, p2)
            d_dist = calculate_hamming_distance(d1, d2)
            p_sim = calculate_similarity(p_dist)
            d_sim = calculate_similarity(d_dist)

            orb_res = compare_images_orb(pil1, pil2)

            st.markdown("#### Comparison Results")
            m0, m1, m2, m3 = st.columns(4)
            m0.metric("SHA-256 Match", "YES (100%)" if is_byte_identical else "NO (Different)", f"{sha1[:8]}...")
            m1.metric("pHash Similarity", f"{p_sim:.1f}%", f"Dist: {p_dist}/64")
            m2.metric("dHash Similarity", f"{d_sim:.1f}%", f"Dist: {d_dist}/64")
            m3.metric("ORB Similarity", f"{orb_res.similarity_score:.1f}%", f"Inliers: {orb_res.ransac_inliers}")

            if is_byte_identical:
                st.success("Verdict: EXACT DUPLICATE (Byte-Identical File Copy)")
                st.info(f"Identical SHA-256 Digest: `{sha1}`. Byte-identical files are confirmed exact duplicates and cannot be overridden.")
            elif p_dist == 0 and d_dist == 0:
                st.success("Verdict: VISUALLY IDENTICAL (Re-encoded / Metadata Variant)")
                st.info("Different file hashes/metadata, but bit-for-bit identical perceptual and difference fingerprints.")
            elif p_sim >= 85.0 and d_sim >= 85.0:
                st.success("Verdict: NEAR-DUPLICATE (Perceptual Match: Resized / Compressed / Lighting Adjusted)")
            elif orb_res.is_duplicate:
                st.success(f"Verdict: NEAR-DUPLICATE (Geometric Match: Cropped / Rotated – {orb_res.ransac_inliers} RANSAC Inliers)")
            elif d_sim >= 85.0 and p_sim < 75.0 and not orb_res.is_duplicate:
                st.warning("Verdict: NON-DUPLICATE (Different Images with Similar Background / Gradient)")
                st.caption("Notice: High dHash similarity is an artifact of monotonic gradient or layout alignment. Low pHash and low ORB inliers confirm distinct foreground content.")
            else:
                st.warning("Verdict: NON-DUPLICATE (Different Images)")

# ==========================================================
# TAB 4: DOCUMENT CLASSIFICATION
# ==========================================================
with tabs[3]:
    st.subheader("Module 4: 16-Class Document Classification (RVL-CDIP)")
    st.write("Classifies documents into 16 categories using ResNet18 Deep Transfer Learning.")

    doc_file = st.file_uploader(
        "Upload Document (PDF, Scanned Image, DOCX, TXT)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "docx", "txt"],
        key="doc_classify_upload"
    )

    if doc_file:
        tmp_path = Path("temp_upload") / doc_file.name
        tmp_path.parent.mkdir(exist_ok=True)
        tmp_path.write_bytes(doc_file.read())

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            if doc_file.name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
                st.image(str(tmp_path), caption="Uploaded Document Preview", use_container_width=True)
            else:
                st.info(f"Loaded document file: {doc_file.name} ({doc_file.size} bytes)")

        with col_d2:
            if st.button("Classify Document", use_container_width=True):
                with st.spinner("Analyzing document visual and layout features..."):
                    from document_classification import DocumentClassifier
                    classifier = DocumentClassifier()
                    res = classifier.classify_document(tmp_path)

                    st.markdown(f"### Predicted Class: **{res.predicted_category.upper()}**")
                    st.progress(res.confidence)
                    st.write(f"Confidence: **{res.confidence * 100:.2f}%** | Latency: **{res.processing_time_ms:.1f} ms**")

                    st.markdown("#### Top Alternative Candidates:")
                    for cat, conf in res.alternative_predictions:
                        st.write(f"- **{cat.capitalize()}**: {conf * 100:.2f}%")

# ==========================================================
# TAB 5: DOCUMENT SIMILARITY
# ==========================================================
with tabs[4]:
    st.subheader("Module 3: Document Near-Duplicate & Cross-Format Similarity")
    st.write("Evaluates textual content across PDF, DOCX, and TXT using TF-IDF and Cosine Similarity.")

    c_doc1, c_doc2 = st.columns(2)
    with c_doc1:
        doc1 = st.file_uploader("Document A (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="doc_a")
    with c_doc2:
        doc2 = st.file_uploader("Document B (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="doc_b")

    if doc1 and doc2:
        p1 = Path("temp_upload") / doc1.name
        p2 = Path("temp_upload") / doc2.name
        p1.parent.mkdir(exist_ok=True)
        p1.write_bytes(doc1.read())
        p2.write_bytes(doc2.read())

        if st.button("Calculate Document Similarity", use_container_width=True):
            from document_detection import compare_two_documents
            res = compare_two_documents(p1, p2)

            st.metric("TF-IDF Cosine Similarity", f"{res.similarity_score:.2f}%")
            st.write(f"Category: **{res.duplicate_type}** | Duplicate: **{res.is_duplicate}**")

# ==========================================================
# TAB 6: REPORTS & EVALUATION
# ==========================================================
with tabs[5]:
    st.subheader("Academic Evaluation & Benchmark Artifacts")

    rep_tab1, rep_tab2, rep_tab3 = st.tabs(["Document Classification Evaluation", "Image Detection Evaluation", "Download CSV Reports"])

    with rep_tab1:
        st.markdown("#### RVL-CDIP 16-Class Model Evaluation")
        if Path("reports/model_comparison.csv").exists():
            st.markdown("##### Model Comparison")
            st.dataframe(pd.read_csv("reports/model_comparison.csv"), use_container_width=True)

        if Path("reports/confusion_matrix.png").exists():
            st.markdown("##### Confusion Matrix (Untouched Test Set, N=992)")
            st.image("reports/confusion_matrix.png", use_container_width=True)

        if Path("reports/training_curves.png").exists():
            st.markdown("##### Training & Validation Curves")
            st.image("reports/training_curves.png", use_container_width=True)

    with rep_tab2:
        st.markdown("#### INRIA Copydays & Synthetic Transformation Benchmarks")
        if Path("reports/image_benchmark_results.csv").exists():
            st.markdown("##### Compression Robustness (JPEG Attacks)")
            st.dataframe(pd.read_csv("reports/image_benchmark_results.csv"), use_container_width=True)

        if Path("reports/synthetic_crop_benchmark.csv").exists():
            st.markdown("##### Geometric Transformation Robustness (ORB vs pHash/dHash)")
            st.dataframe(pd.read_csv("reports/synthetic_crop_benchmark.csv"), use_container_width=True)

    with rep_tab3:
        st.markdown("#### Downloadable CSV Reports")
        for rep_path in Path("reports").glob("*.csv"):
            with open(rep_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download {rep_path.name}",
                    data=f.read(),
                    file_name=rep_path.name,
                    mime="text/csv",
                    key=f"dl_{rep_path.name}"
                )
