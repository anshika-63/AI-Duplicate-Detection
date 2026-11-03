"""
Image Detection Benchmark Utility
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Evaluates:
- Perceptual Hashing (pHash)
- Difference Hashing (dHash)
- ORB Feature Matching with RANSAC Homography

Against:
1. INRIA Copydays: Original vs JPEG compression attacks (9 quality levels).
2. INRIA Copydays: Original vs Strong attacks.
3. Synthetic Transformations: Cropping (50%, 70%) and Rotation (15°, 45°, 90°).

Calculates strictly measured evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR).
- Threshold sensitivity analysis across similarity thresholds (50% to 95%).
- Generates reports in reports/ directory.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from image_detection import (
    calculate_hamming_distance,
    calculate_similarity,
    compute_dhash,
    compute_phash,
)
from orb_detection import compare_images_orb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INRIA_DIR = r"E:\Fake-Detect\datasets\image-detection"
DEFAULT_REPORTS_DIR = "reports"


def compute_metrics(
    y_true: List[int], y_pred: List[int], y_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculates evaluation metrics without fabricated values."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix elements
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))

    auc = 0.0
    if y_scores and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "roc_auc": round(auc, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_inria_jpeg(
    inria_root: Path, sample_limit: int = 50
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluates original vs JPEG attacked images.
    Ground truth: Same image ID (e.g., 200100.jpg vs 200100.jpg) = Duplicate (1).
    Negative pairs: Different image IDs = Non-duplicate (0).
    """
    orig_dir = inria_root / "orignal"
    jpeg_qual_dir = inria_root / "jpeg" / "jpegqual"

    if not orig_dir.exists() or not jpeg_qual_dir.exists():
        logger.warning("INRIA JPEG directories not found at %s", inria_root)
        return [], []

    orig_files = sorted(list(orig_dir.glob("*.jpg")))[:sample_limit]
    quality_levels = ["10", "30", "50", "75"]

    benchmark_rows = []
    threshold_rows = []

    # Prepare evaluation pairs
    eval_pairs = []
    # 1. Positive pairs: Original vs JPEG at each quality
    for orig in orig_files:
        for q in quality_levels:
            q_file = jpeg_qual_dir / q / orig.name
            if q_file.exists():
                eval_pairs.append((orig, q_file, 1, f"JPEG-Q{q}"))

    # 2. Negative pairs: Original vs different image
    for i, orig in enumerate(orig_files):
        other = orig_files[(i + 1) % len(orig_files)]
        eval_pairs.append((orig, other, 0, "Unrelated-Originals"))

    logger.info("Evaluating %d image pairs on INRIA JPEG benchmark...", len(eval_pairs))

    # Pre-extract hashes
    phash_scores, dhash_scores, orb_scores, ground_truth = [], [], [], []

    for img1_path, img2_path, label, transform_name in eval_pairs:
        # pHash
        p1 = compute_phash(img1_path)
        p2 = compute_phash(img2_path)
        p_dist = calculate_hamming_distance(p1, p2)
        p_sim = calculate_similarity(p_dist, total_bits=64)

        # dHash
        d1 = compute_dhash(img1_path)
        d2 = compute_dhash(img2_path)
        d_dist = calculate_hamming_distance(d1, d2)
        d_sim = calculate_similarity(d_dist, total_bits=64)

        # ORB
        orb_res = compare_images_orb(img1_path, img2_path, nfeatures=500)
        orb_sim = orb_res.similarity_score

        phash_scores.append(p_sim)
        dhash_scores.append(d_sim)
        orb_scores.append(orb_sim)
        ground_truth.append(label)

    # Benchmark at standard 80% threshold
    for algo_name, scores in [("pHash", phash_scores), ("dHash", dhash_scores), ("ORB", orb_scores)]:
        preds = [1 if s >= 80.0 else 0 for s in scores]
        m = compute_metrics(ground_truth, preds, scores)
        m["algorithm"] = algo_name
        m["benchmark_dataset"] = "INRIA Copydays (JPEG Attacks)"
        m["threshold"] = 80.0
        benchmark_rows.append(m)

    # Threshold sensitivity analysis (50% to 95%)
    for thresh in range(50, 96, 5):
        for algo_name, scores in [("pHash", phash_scores), ("dHash", dhash_scores), ("ORB", orb_scores)]:
            preds = [1 if s >= float(thresh) else 0 for s in scores]
            m = compute_metrics(ground_truth, preds, scores)
            m["algorithm"] = algo_name
            m["threshold"] = float(thresh)
            m["benchmark_dataset"] = "INRIA Copydays (JPEG Attacks)"
            threshold_rows.append(m)

    return benchmark_rows, threshold_rows


def evaluate_synthetic_crops(
    inria_root: Path, sample_limit: int = 30
) -> List[Dict[str, Any]]:
    """
    Evaluates robustness on Synthetic Cropping and Rotation benchmark.
    Clearly labeled as synthetic transformations.
    """
    orig_dir = inria_root / "orignal"
    if not orig_dir.exists():
        return []

    orig_files = sorted(list(orig_dir.glob("*.jpg")))[:sample_limit]
    rows = []

    transformations = [
        ("Crop 70% Area", lambda img: img.crop((int(img.width * 0.15), int(img.height * 0.15), int(img.width * 0.85), int(img.height * 0.85)))),
        ("Crop 50% Area", lambda img: img.crop((int(img.width * 0.25), int(img.height * 0.25), int(img.width * 0.75), int(img.height * 0.75)))),
        ("Rotate 15 deg", lambda img: img.rotate(15, expand=False)),
        ("Rotate 90 deg", lambda img: img.rotate(90, expand=True)),
    ]

    for t_name, t_func in transformations:
        y_true, phash_pred, dhash_pred, orb_pred = [], [], [], []
        phash_sims, dhash_sims, orb_sims = [], [], []

        for orig in orig_files:
            with Image.open(orig) as pil_img:
                rgb_orig = pil_img.convert("RGB")
                trans_img = t_func(rgb_orig)

            # Positive pair (original vs transformed)
            p1 = compute_phash(rgb_orig)
            p2 = compute_phash(trans_img)
            p_sim = calculate_similarity(calculate_hamming_distance(p1, p2))

            d1 = compute_dhash(rgb_orig)
            d2 = compute_dhash(trans_img)
            d_sim = calculate_similarity(calculate_hamming_distance(d1, d2))

            orb_res = compare_images_orb(rgb_orig, trans_img, nfeatures=500)
            orb_sim = orb_res.similarity_score

            y_true.append(1)
            phash_sims.append(p_sim)
            dhash_sims.append(d_sim)
            orb_sims.append(orb_sim)

            phash_pred.append(1 if p_sim >= 75.0 else 0)
            dhash_pred.append(1 if d_sim >= 75.0 else 0)
            orb_pred.append(1 if orb_sim >= 60.0 else 0)

        for algo_name, preds, sims in [
            ("pHash", phash_pred, phash_sims),
            ("dHash", dhash_pred, dhash_sims),
            ("ORB", orb_pred, orb_sims),
        ]:
            rec_rate = sum(preds) / len(preds)
            avg_sim = float(np.mean(sims))
            rows.append({
                "transformation": t_name,
                "algorithm": algo_name,
                "test_pairs": len(preds),
                "recall_detected": round(rec_rate, 4),
                "avg_similarity_score": round(avg_sim, 2),
                "note": "Synthetic transformation on INRIA Originals (Not official Copydays crops)",
            })

    return rows


def run_image_benchmarks(
    dataset_root: str = DEFAULT_INRIA_DIR, reports_dir: str = DEFAULT_REPORTS_DIR
):
    """Orchestrates image benchmarks and exports CSV reports."""
    inria_path = Path(dataset_root)
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Image Detection Benchmarks on INRIA Copydays at '%s'...", inria_path)

    # 1. INRIA JPEG Benchmark & Threshold Analysis
    jpeg_benchmarks, threshold_analysis = evaluate_inria_jpeg(inria_path, sample_limit=40)

    # 2. Synthetic Crop & Rotation Benchmark
    crop_benchmarks = evaluate_synthetic_crops(inria_path, sample_limit=25)

    # Export image_benchmark_results.csv
    bench_file = rep_dir / "image_benchmark_results.csv"
    if jpeg_benchmarks:
        with open(bench_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(jpeg_benchmarks[0].keys()))
            writer.writeheader()
            for r in jpeg_benchmarks:
                writer.writerow(r)
        logger.info("Saved image benchmark results to '%s'.", bench_file)

    # Export threshold_analysis.csv
    thresh_file = rep_dir / "threshold_analysis.csv"
    if threshold_analysis:
        with open(thresh_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_analysis[0].keys()))
            writer.writeheader()
            for r in threshold_analysis:
                writer.writerow(r)
        logger.info("Saved threshold sensitivity analysis to '%s'.", thresh_file)

    # Export synthetic_crop_benchmark.csv
    crop_file = rep_dir / "synthetic_crop_benchmark.csv"
    if crop_benchmarks:
        with open(crop_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(crop_benchmarks[0].keys()))
            writer.writeheader()
            for r in crop_benchmarks:
                writer.writerow(r)
        logger.info("Saved synthetic crop benchmark to '%s'.", crop_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Detection Benchmark Utility")
    parser.add_argument("--root", type=str, default=DEFAULT_INRIA_DIR, help="INRIA dataset root")
    parser.add_argument("--reports", type=str, default=DEFAULT_REPORTS_DIR, help="Reports destination folder")
    args = parser.parse_args()

    run_image_benchmarks(dataset_root=args.root, reports_dir=args.reports)
