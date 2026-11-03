"""
RVL-CDIP 16-Class Document Classification Training & Evaluation
T243 – AI-Based Duplicate Document & Near-Duplicate Image Detection

Features:
- Reads RVL-CDIP 8,992 document images directly from Parquet files without disk extraction.
- 16 document classes: letter, form, email, handwritten, advertisement, scientific report,
  scientific publication, specification, file folder, news article, budget, invoice,
  presentation, questionnaire, resume, memo.
- Stratified 85% Train (6,800) / 15% Validation (1,200) split; untouched 992 official test set.
- ResNet18 transfer learning feature extraction with on-disk caching (cache/rvlcdip_resnet18_features.npz).
- Multi-model comparison: Linear SVM / Logistic Classifier vs Deep Neural Classification Head.
- Overfitting / underfitting monitoring with training curves and early stopping.
- Complete evaluation metrics: Macro/Weighted Precision, Recall, F1-score, Per-Class metrics, Confusion Matrix.
- Saves model checkpoint and CSV reports to checkpoints/ and reports/.
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = r"E:\Fake-Detect\datasets\document_classification"
CACHE_DIR = "cache"
CHECKPOINTS_DIR = "checkpoints"
REPORTS_DIR = "reports"

# Canonical 16-class label mapping from RVL-CDIP dataset metadata
CLASS_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


class DeepClassificationHead(nn.Module):
    """Deep neural classification head with Dropout and BatchNorm to prevent overfitting."""

    def __init__(self, in_features: int = 512, num_classes: int = 16, dropout_rate: float = 0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def extract_features_from_parquet(
    parquet_path: Path,
    model: nn.Module,
    transform: transforms.Compose,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts ResNet18 feature vectors from Parquet table in batches.
    Returns (features_array, labels_array).
    """
    tbl = pq.read_table(parquet_path)
    num_rows = tbl.num_rows
    logger.info("Extracting visual features from %s (%d samples)...", parquet_path.name, num_rows)

    all_features = []
    all_labels = tbl.column("label").to_numpy()

    image_col = tbl.column("image")

    for i in range(0, num_rows, batch_size):
        batch_end = min(i + batch_size, num_rows)
        batch_tensors = []

        for row_idx in range(i, batch_end):
            img_bytes = image_col[row_idx].as_py()["bytes"]
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(255, 255, 255))
            batch_tensors.append(transform(img))

        batch_stack = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = model(batch_stack)
            all_features.append(feats.cpu().numpy())

        if (i // batch_size) % 25 == 0 and i > 0:
            logger.info("Processed %d / %d samples...", i, num_rows)

    features_matrix = np.vstack(all_features)
    return features_matrix, all_labels


def get_or_create_features(
    dataset_dir: Path, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts or loads cached ResNet18 features for train and test sets.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "rvlcdip_resnet18_features.npz"

    if cache_file.exists():
        logger.info("Loading cached ResNet18 features from '%s'...", cache_file)
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    logger.info("Cached features not found. Initializing ResNet18 feature extractor...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()

    # Train files
    train1_p = dataset_dir / "train-00000-of-00002.parquet"
    train2_p = dataset_dir / "train-00001-of-00002.parquet"
    test_p = dataset_dir / "test-00000-of-00001.parquet"

    X_train1, y_train1 = extract_features_from_parquet(train1_p, resnet, transform)
    X_train2, y_train2 = extract_features_from_parquet(train2_p, resnet, transform)
    X_train = np.vstack([X_train1, X_train2])
    y_train = np.concatenate([y_train1, y_train2])

    X_test, y_test = extract_features_from_parquet(test_p, resnet, transform)

    logger.info("Saving extracted features to cache '%s'...", cache_file)
    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return X_train, y_train, X_test, y_test


def train_and_evaluate(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    max_epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    """Orchestrates end-to-end model training, validation, baseline comparison, and evaluation."""
    data_dir = Path(dataset_dir)
    check_dir = Path(CHECKPOINTS_DIR)
    rep_dir = Path(REPORTS_DIR)
    check_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or extract features
    X_train_full, y_train_full, X_test, y_test = get_or_create_features(
        data_dir, Path(CACHE_DIR)
    )

    # 2. Stratified Train / Validation Split (85% Train = 6800, 15% Val = 1200)
    logger.info("Splitting training data into 85%% Train and 15%% Validation (Stratified)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(X_train_full, y_train_full))

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    logger.info("Split sizes: Train = %d, Validation = %d, Test = %d", len(X_train), len(X_val), len(X_test))

    # Save class distribution report
    dist_file = rep_dir / "class_distribution.csv"
    with open(dist_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "train_count", "val_count", "test_count"])
        for cid, cname in enumerate(CLASS_NAMES):
            tr_cnt = int(np.sum(y_train == cid))
            vl_cnt = int(np.sum(y_val == cid))
            ts_cnt = int(np.sum(y_test == cid))
            writer.writerow([cid, cname, tr_cnt, vl_cnt, ts_cnt])
    logger.info("Exported class distribution to '%s'.", dist_file)

    # ==========================================
    # MODEL 1: Logistic Regression / Linear Baseline
    # ==========================================
    logger.info("Training Model 1: Linear Softmax / Logistic Regression Baseline...")
    linear_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    linear_clf.fit(X_train, y_train)

    lin_val_preds = linear_clf.predict(X_val)
    lin_val_acc = np.mean(lin_val_preds == y_val)
    lin_test_preds = linear_clf.predict(X_test)
    lin_test_acc = np.mean(lin_test_preds == y_test)
    lin_test_f1 = f1_score(y_test, lin_test_preds, average="weighted")
    logger.info("Model 1 (Linear Baseline) - Val Acc: %.4f, Test Acc: %.4f, Weighted F1: %.4f",
                lin_val_acc, lin_test_acc, lin_test_f1)

    # ==========================================
    # MODEL 2: Deep MLP Classification Head with Dropout & Early Stopping
    # ==========================================
    logger.info("Training Model 2: Deep MLP Classification Head (PyTorch AdamW)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DeepClassificationHead(in_features=512, num_classes=16, dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # Training loop
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation loop
        model.eval()
        val_loss_running = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss_running += loss.item() * len(batch_y)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == batch_y).sum().item()
                total_val += len(batch_y)

        val_loss = val_loss_running / total_val
        val_acc = correct_val / total_val

        scheduler.step(val_acc)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
            epoch, max_epochs, train_loss, train_acc, val_loss, val_acc
        )

        # Early stopping and best model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d (no validation improvement for %d epochs).", epoch, patience)
                break

    # Save training history
    hist_file = rep_dir / "training_history.csv"
    with open(hist_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])
    logger.info("Saved training history to '%s'.", hist_file)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(history["epoch"], history["val_acc"], label="Val Accuracy", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_file = rep_dir / "training_curves.png"
    plt.savefig(curve_file, dpi=150)
    plt.close()
    logger.info("Saved training curves plot to '%s'.", curve_file)

    # Load best checkpoint
    model.load_state_dict(best_weights)

    # Save best checkpoint
    checkpoint_path = check_dir / "best_document_classifier.pth"
    torch.save({
        "state_dict": best_weights,
        "class_names": CLASS_NAMES,
        "input_dim": 512,
        "num_classes": 16,
        "best_val_acc": best_val_acc,
    }, checkpoint_path)
    logger.info("Saved best model checkpoint to '%s'.", checkpoint_path)

    # Save model metadata JSON
    meta_path = check_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "ResNet18_TransferLearning_MLP",
            "backbone": "resnet18",
            "feature_dim": 512,
            "num_classes": 16,
            "classes": CLASS_NAMES,
            "best_validation_accuracy": round(best_val_acc, 4),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        }, f, indent=2)

    # ==========================================
    # FINAL EVALUATION ON UNTOUCHED TEST SET (992 IMAGES)
    # ==========================================
    logger.info("Evaluating Best Model on untouched official RVL-CDIP Test Set (992 samples)...")
    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        test_preds = np.argmax(probs, axis=1)

    test_acc = float(np.mean(test_preds == y_test))
    macro_prec = float(precision_score(y_test, test_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))

    logger.info("Final Test Performance:")
    logger.info(" - Test Accuracy:     %.4f (%.2f%%)", test_acc, test_acc * 100)
    logger.info(" - Macro Precision:   %.4f", macro_prec)
    logger.info(" - Macro Recall:      %.4f", macro_rec)
    logger.info(" - Macro F1-Score:    %.4f", macro_f1)
    logger.info(" - Weighted F1-Score: %.4f", weighted_f1)

    # Model comparison report
    comp_file = rep_dir / "model_comparison.csv"
    with open(comp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "architecture", "val_accuracy", "test_accuracy", "weighted_f1", "notes"])
        writer.writerow([
            "Baseline 1", "ResNet18 Features + Logistic Regression",
            round(lin_val_acc, 4), round(lin_test_acc, 4), round(lin_test_f1, 4),
            "Linear baseline with L2 regularization"
        ])
        writer.writerow([
            "Baseline 2 (Best)", "ResNet18 Features + Deep MLP (Dropout & BatchNorm)",
            round(best_val_acc, 4), round(test_acc, 4), round(weighted_f1, 4),
            "Non-linear multi-layer representation with AdamW early stopping"
        ])
    logger.info("Saved model comparison to '%s'.", comp_file)

    # Per-Class Classification Report
    clf_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)
    rep_csv = rep_dir / "document_classification_report.csv"
    with open(rep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for cname in CLASS_NAMES:
            c_dict = clf_report[cname]
            writer.writerow([
                cname,
                round(c_dict["precision"], 4),
                round(c_dict["recall"], 4),
                round(c_dict["f1-score"], 4),
                int(c_dict["support"]),
            ])
        writer.writerow([
            "macro avg",
            round(clf_report["macro avg"]["precision"], 4),
            round(clf_report["macro avg"]["recall"], 4),
            round(clf_report["macro avg"]["f1-score"], 4),
            int(clf_report["macro avg"]["support"]),
        ])
        writer.writerow([
            "weighted avg",
            round(clf_report["weighted avg"]["precision"], 4),
            round(clf_report["weighted avg"]["recall"], 4),
            round(clf_report["weighted avg"]["f1-score"], 4),
            int(clf_report["weighted avg"]["support"]),
        ])
    logger.info("Saved per-class classification report to '%s'.", rep_csv)

    # Sample Predictions CSV
    pred_csv = rep_dir / "document_classification_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_class", "predicted_class", "confidence", "is_correct"])
        for idx in range(min(len(y_test), 200)):
            p_class = CLASS_NAMES[test_preds[idx]]
            t_class = CLASS_NAMES[y_test[idx]]
            conf = round(float(probs[idx][test_preds[idx]]), 4)
            writer.writerow([idx, t_class, p_class, conf, p_class == t_class])
    logger.info("Saved sample predictions to '%s'.", pred_csv)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted Document Class")
    plt.ylabel("True Document Class")
    plt.title("RVL-CDIP 16-Class Confusion Matrix (Test Set, N=992)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_file = rep_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to '%s'.", cm_file)



def evaluate_model(
    checkpoint_path: Path = Path("checkpoints/best_document_classifier.pth"),
    cache_dir: Path = Path("cache"),
    reports_dir: Path = Path("reports"),
    dataset_dir: Path = Path(DEFAULT_DATASET_DIR),
) -> Dict[str, float]:
    """
    Evaluates saved model checkpoint on untouched official RVL-CDIP Test Set (992 samples).
    Computes Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix, and CSV reports.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running Official Evaluation using device: %s", device)

    # 1. Load features
    _, _, X_test, y_test = get_or_create_features(dataset_dir, cache_dir)
    logger.info("Loaded test set with %d samples across %d classes.", len(y_test), len(CLASS_NAMES))

    # 2. Load model
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    model = DeepClassificationHead(in_features=512, num_classes=16).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        test_preds = np.argmax(probs, axis=1)

    test_acc = float(np.mean(test_preds == y_test))
    macro_prec = float(precision_score(y_test, test_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))

    logger.info("Official Test Performance:")
    logger.info(" - Test Accuracy:     %.4f (%.2f%%)", test_acc, test_acc * 100)
    logger.info(" - Macro Precision:   %.4f", macro_prec)
    logger.info(" - Macro Recall:      %.4f", macro_rec)
    logger.info(" - Macro F1-Score:    %.4f", macro_f1)
    logger.info(" - Weighted F1-Score: %.4f", weighted_f1)

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Classification Report
    clf_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)
    rep_csv = reports_dir / "document_classification_report.csv"
    with open(rep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for cname in CLASS_NAMES:
            c_dict = clf_report[cname]
            writer.writerow([
                cname,
                round(c_dict["precision"], 4),
                round(c_dict["recall"], 4),
                round(c_dict["f1-score"], 4),
                int(c_dict["support"]),
            ])
        writer.writerow([
            "macro avg",
            round(clf_report["macro avg"]["precision"], 4),
            round(clf_report["macro avg"]["recall"], 4),
            round(clf_report["macro avg"]["f1-score"], 4),
            len(y_test),
        ])
        writer.writerow([
            "weighted avg",
            round(clf_report["weighted avg"]["precision"], 4),
            round(clf_report["weighted avg"]["recall"], 4),
            round(clf_report["weighted avg"]["f1-score"], 4),
            len(y_test),
        ])
    logger.info("Saved per-class classification report to '%s'.", rep_csv)

    # Predictions CSV
    pred_csv = reports_dir / "document_classification_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_class", "predicted_class", "confidence", "is_correct"])
        for idx in range(min(len(y_test), 200)):
            p_class = CLASS_NAMES[test_preds[idx]]
            t_class = CLASS_NAMES[y_test[idx]]
            conf = round(float(probs[idx][test_preds[idx]]), 4)
            writer.writerow([idx, t_class, p_class, conf, p_class == t_class])
    logger.info("Saved sample predictions to '%s'.", pred_csv)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted Document Class")
    plt.ylabel("True Document Class")
    plt.title("RVL-CDIP 16-Class Confusion Matrix (Test Set, N=992)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_file = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to '%s'.", cm_file)

    return {
        "test_accuracy": test_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVL-CDIP Document Classifier Trainer & Evaluator")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to Parquet files")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only on official test partition")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_model(dataset_dir=Path(args.dataset_dir))
    else:
        train_and_evaluate(
            dataset_dir=args.dataset_dir,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
