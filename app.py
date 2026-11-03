"""
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection Dashboard
Interactive Web Interface powered by Streamlit

Pages:
1. Overview – System metrics, database stats, unified batch directory scanner.
2. Exact Duplicates – Cryptographic SHA-256 duplicate clusters.
3. Image Detection – Visual side-by-side image comparison with pHash, dHash, and ORB.
4. Document Classification – Interactive 16-class document classifier with top-5 confidence bars.
5. Document Similarity – Text extraction, quality indicators, TF-IDF Cosine scores, text diff previews.
6. Reports & Evaluation – Confusion matrix, training curves, transformation matrices, benchmark tables.
7. Database & History – Query persisted records, filter by duplicate type, review scan audits.
"""

import hashlib
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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 42px; white-space: pre-wrap; background-color: #F9FAFB; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #EFF6FF !important; border-bottom: 2px solid #2563EB !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Header
st.markdown('<div class="main-header">T243 – AI Duplicate Detection & Document Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-modal platform for cryptographic hashing, perceptual image fingerprints, ORB feature matching, and RVL-CDIP 16-class classification.</div>', unsafe_allow_html=True)

# Database instance
from database_manager import DatabaseManager
db = DatabaseManager()

# Navigation
tabs = st.tabs([
    "📊 Overview",
    "🔒 Exact Duplicates",
    "🖼️ Image Detection",
    "📑 Document Classification",
    "📝 Document Similarity",
    "📈 Reports & Evaluation",
    "💾 Database & History",
])

# ==========================================================
# TAB 1: OVERVIEW
# ==========================================================
with tabs[0]:
    st.subheader("System Overview & Live Audit")

    stats = db.get_database_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Detection Modalities", "3 Types", "SHA-256 / Hash / NLP")
    with col2:
        st.metric("Document Classes", "16 Classes", "RVL-CDIP Benchmark")
    with col3:
        st.metric("INRIA Images", "1,799 Files", "Original + Attacks")
    with col4:
        st.metric("Indexed DB Files", f"{stats.get('files', 0):,}", "SQLite Files")
    with col5:
        st.metric("DB Duplicate Pairs", f"{stats.get('detection_results', 0):,}", "Identified Relations")

    st.markdown("---")
    st.markdown("#### Quick Directory Scan")

    scan_folder = st.text_input("Target Directory to Scan", value="dataset")
    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1:
        threshold_val = st.slider("Similarity Threshold (%)", min_value=50.0, max_value=99.0, value=80.0, step=1.0)
    with c_opt2:
        enable_orb_chk = st.checkbox("Enable ORB Geometric Verification", value=True)
    with c_opt3:
        save_db_chk = st.checkbox("Persist Results to SQLite Database", value=True)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_scan = st.button("🚀 Run Unified Scan", use_container_width=True)

    if run_scan:
        with st.spinner(f"Scanning directory '{scan_folder}'..."):
            try:
                from unified_detector import UnifiedDetector
                detector = UnifiedDetector(
                    scan_folder,
                    threshold=threshold_val,
                    enable_orb=enable_orb_chk,
                )
                records, summary, class_results = detector.run(
                    run_exact=True,
                    run_images=True,
                    run_documents=True,
                    run_classification_flag=False,
                )

                if save_db_chk:
                    db.save_results(records)
                    db.log_scan(
                        scanned_folder=str(detector.target_folder),
                        total_files=summary.total_files_scanned,
                        total_matches=summary.total_detected_pairs,
                    )

                st.success(f"Scan Completed in {summary.scan_duration_seconds:.2f}s! Found {summary.total_detected_pairs} duplicate pairs ({summary.throughput_fps:.1f} files/sec).")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files Scanned", summary.total_files_scanned)
                c2.metric("Exact Pairs (SHA-256)", summary.exact_duplicate_pairs)
                c3.metric("Near Images (Hash/ORB)", summary.near_duplicate_image_pairs)
                c4.metric("Near Documents (TF-IDF)", summary.near_duplicate_document_pairs)

                if records:
                    df = pd.DataFrame([r.to_csv_row() for r in records], columns=[
                        "File 1", "File 2", "Name 1", "Name 2", "Type", "Category", "Score", "Algorithm", "Details", "Timestamp"
                    ])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No duplicates detected at the current similarity threshold.")
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
            from image_detection import (
                calculate_hamming_distance,
                calculate_similarity,
                compute_dhash,
                compute_phash,
            )
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
                    res = classifier.classify_document(tmp_path, top_k=5)

                    st.markdown(f"### Predicted Class: **{res.predicted_category.upper()}**")
                    st.progress(min(1.0, max(0.0, res.confidence)))
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
            from document_detection import compare_two_documents, extract_document_text_with_metadata
            res = compare_two_documents(p1, p2)
            meta1 = extract_document_text_with_metadata(p1)
            meta2 = extract_document_text_with_metadata(p2)

            st.metric("TF-IDF Cosine Similarity", f"{res.similarity_score:.2f}%")
            st.write(f"Category: **{res.duplicate_type}** | Duplicate: **{res.is_duplicate}**")

            st.markdown("---")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"**Document A Metrics:** {meta1.word_count} words | {meta1.char_count} chars | Quality: {meta1.quality_score * 100:.1f}%")
                with st.expander("Preview Text Content (Doc A)"):
                    st.text(meta1.text[:600] + ("..." if len(meta1.text) > 600 else ""))
            with col_t2:
                st.markdown(f"**Document B Metrics:** {meta2.word_count} words | {meta2.char_count} chars | Quality: {meta2.quality_score * 100:.1f}%")
                with st.expander("Preview Text Content (Doc B)"):
                    st.text(meta2.text[:600] + ("..." if len(meta2.text) > 600 else ""))

# ==========================================================
# TAB 6: REPORTS & EVALUATION
# ==========================================================
with tabs[5]:
    st.subheader("Academic Evaluation & Benchmark Artifacts")

    rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs([
        "Document Classification (RVL-CDIP)",
        "Image Detection (Copydays & Attacks)",
        "Document Similarity Benchmark",
        "Download CSV Reports",
    ])

    with rep_tab1:
        st.markdown("#### RVL-CDIP 16-Class Model Evaluation")
        if Path("reports/document_classification_report.csv").exists():
            st.markdown("##### Per-Class Precision / Recall / F1 (Untouched Test Partition, N=992)")
            st.dataframe(pd.read_csv("reports/document_classification_report.csv"), use_container_width=True)

        if Path("reports/model_comparison.csv").exists():
            st.markdown("##### Model Architecture Comparison")
            st.dataframe(pd.read_csv("reports/model_comparison.csv"), use_container_width=True)

        if Path("reports/confusion_matrix.png").exists():
            st.markdown("##### Confusion Matrix (Untouched Test Set, N=992)")
            st.image("reports/confusion_matrix.png", use_container_width=True)

        if Path("reports/training_curves.png").exists():
            st.markdown("##### Training & Validation Curves")
            st.image("reports/training_curves.png", use_container_width=True)

    with rep_tab2:
        st.markdown("#### Image Detection Robustness Benchmarks")
        if Path("reports/synthetic_transformation_benchmark.csv").exists():
            st.markdown("##### 8-Category Transformation Matrix (Exact, Resize, JPEG, Brightness, Rotation, Crop, Negative Control)")
            st.dataframe(pd.read_csv("reports/synthetic_transformation_benchmark.csv"), use_container_width=True)

        if Path("reports/image_benchmark_results.csv").exists():
            st.markdown("##### INRIA Copydays Compression Robustness (JPEG Attacks)")
            st.dataframe(pd.read_csv("reports/image_benchmark_results.csv"), use_container_width=True)

        if Path("reports/synthetic_crop_benchmark.csv").exists():
            st.markdown("##### Geometric Transformation Robustness (ORB vs pHash/dHash)")
            st.dataframe(pd.read_csv("reports/synthetic_crop_benchmark.csv"), use_container_width=True)

    with rep_tab3:
        st.markdown("#### Document Similarity Benchmark")
        if Path("reports/document_similarity_benchmark.csv").exists():
            st.markdown("##### 9-Category NLP Evaluation Matrix")
            st.dataframe(pd.read_csv("reports/document_similarity_benchmark.csv"), use_container_width=True)

    with rep_tab4:
        st.markdown("#### Downloadable CSV Reports")
        for rep_path in sorted(Path("reports").glob("*.csv")):
            with open(rep_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download {rep_path.name}",
                    data=f.read(),
                    file_name=rep_path.name,
                    mime="text/csv",
                    key=f"dl_{rep_path.name}"
                )

# ==========================================================
# TAB 7: DATABASE & HISTORY
# ==========================================================
with tabs[6]:
    st.subheader("Persistent SQLite Database & Audit Logs")

    db_stats = db.get_database_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed Files", db_stats.get("files", 0))
    c2.metric("Detected Pairs", db_stats.get("detection_results", 0))
    c3.metric("Classification Records", db_stats.get("classification_results", 0))
    c4.metric("Scan Audit Logs", db_stats.get("scan_history", 0))

    st.markdown("---")
    st.markdown("#### Query Duplicate Detection Records")

    q_col1, q_col2 = st.columns([1, 1])
    with q_col1:
        min_sim_filter = st.slider("Filter by Minimum Similarity (%)", 0.0, 100.0, 75.0, step=5.0)
    with q_col2:
        dup_type_filter = st.selectbox(
            "Filter by Duplicate Category",
            ["All", "exact_duplicate", "near_duplicate_image", "near_duplicate_document", "cross_format_document", "possible_duplicate"],
        )

    db_records = db.get_detection_results(min_similarity=min_sim_filter, duplicate_type=dup_type_filter)
    if db_records:
        df_db = pd.DataFrame(db_records)
        st.dataframe(df_db, use_container_width=True)
    else:
        st.info("No records matching the filter criteria found in database.")

    st.markdown("---")
    st.markdown("#### Recent Scan History")
    history_records = db.get_scan_history(limit=20)
    if history_records:
        df_hist = pd.DataFrame(history_records)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No scan history recorded yet.")
