"""
ORB (Oriented FAST and Rotated BRIEF) Feature Matching Module
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Features:
- Fast multi-scale keypoint detection and orientation computation.
- Binary descriptor extraction (32 bytes per keypoint).
- Brute-Force Hamming matching with Lowe's ratio test.
- RANSAC Homography geometric verification to eliminate false matches.
- Robust detection against severe cropping, rotation, scaling, and perspective distortion.
- Clean integration with the unified pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_ORB_FEATURES = 1000
DEFAULT_LOWE_RATIO = 0.75
DEFAULT_RANSAC_REPROJ_THRESH = 5.0
DEFAULT_MIN_INLIERS = 10


@dataclass
class ORBComparisonResult:
    """Stores feature matching metrics between two images."""
    image_path_1: str
    image_path_2: str
    keypoints_1: int
    keypoints_2: int
    raw_matches: int
    good_matches: int
    ransac_inliers: int
    inlier_ratio: float
    similarity_score: float
    is_duplicate: bool
    detection_algorithm: str = "ORB + RANSAC Homography"


def load_image_cv(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Safely loads an image into grayscale NumPy array for OpenCV processing."""
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        return image_input

    if isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        arr = np.array(rgb_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Read using cv2 with Unicode path support
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
        return img
    except Exception as e:
        # Fallback to Pillow
        with Image.open(path) as pil_img:
            return np.array(pil_img.convert("L"))


def extract_orb_features(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    nfeatures: int = DEFAULT_ORB_FEATURES,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Extracts ORB keypoints and descriptors from an image.
    """
    gray = load_image_cv(image_input)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_orb_descriptors(
    des1: Optional[np.ndarray],
    des2: Optional[np.ndarray],
    ratio_threshold: float = DEFAULT_LOWE_RATIO,
) -> List[cv2.DMatch]:
    """
    Matches binary descriptors using Brute-Force Hamming distance and Lowe's ratio test.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def verify_geometric_homography(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESH,
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Applies RANSAC Homography verification to validate spatial consistency.
    Returns (num_inliers, homography_matrix).
    """
    if len(good_matches) < 4:
        return 0, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
        if mask is None:
            return 0, None
        inliers = int(np.sum(mask))
        return inliers, H
    except Exception as e:
        logger.debug("Homography computation error: %s", e)
        return 0, None


def compare_images_orb(
    img1_path: Union[str, Path, Image.Image, np.ndarray],
    img2_path: Union[str, Path, Image.Image, np.ndarray],
    nfeatures: int = DEFAULT_ORB_FEATURES,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    similarity_threshold: float = 60.0,
) -> ORBComparisonResult:
    """
    Compares two images using ORB keypoint matching + RANSAC verification.
    """
    kp1, des1 = extract_orb_features(img1_path, nfeatures=nfeatures)
    kp2, des2 = extract_orb_features(img2_path, nfeatures=nfeatures)

    good_matches = match_orb_descriptors(des1, des2)
    inliers, H = verify_geometric_homography(kp1, kp2, good_matches)

    min_kps = max(1, min(len(kp1), len(kp2)))
    inlier_ratio = float(inliers) / float(min_kps)

    # Scale similarity: inliers >= 30 or inlier_ratio >= 0.15 indicates strong near-duplicate
    # Calibrate normalized similarity score (0 - 100)
    score_from_inliers = min(100.0, (inliers / 30.0) * 100.0)
    score_from_ratio = min(100.0, (inlier_ratio / 0.15) * 100.0)
    similarity_score = round(max(score_from_inliers, score_from_ratio), 2)

    is_duplicate = inliers >= min_inliers and similarity_score >= similarity_threshold

    p1_str = str(img1_path) if isinstance(img1_path, (str, Path)) else "Image 1"
    p2_str = str(img2_path) if isinstance(img2_path, (str, Path)) else "Image 2"

    return ORBComparisonResult(
        image_path_1=p1_str,
        image_path_2=p2_str,
        keypoints_1=len(kp1),
        keypoints_2=len(kp2),
        raw_matches=len(good_matches),
        good_matches=len(good_matches),
        ransac_inliers=inliers,
        inlier_ratio=round(inlier_ratio, 4),
        similarity_score=similarity_score,
        is_duplicate=is_duplicate,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        res = compare_images_orb(sys.argv[1], sys.argv[2])
        print(f"\nORB Comparison:")
        print(f" - Keypoints 1: {res.keypoints_1}")
        print(f" - Keypoints 2: {res.keypoints_2}")
        print(f" - Good matches: {res.good_matches}")
        print(f" - RANSAC inliers: {res.ransac_inliers} (Ratio: {res.inlier_ratio:.4f})")
        print(f" - Similarity Score: {res.similarity_score}%")
        print(f" - Duplicate: {res.is_duplicate}\n")
    else:
        print("Usage: python orb_detection.py <image1> <image2>")
