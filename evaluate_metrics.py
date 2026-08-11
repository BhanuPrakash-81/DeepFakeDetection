import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import torch
from typing import Tuple

from model_and_train import MultimodalGatedAttentionAdapter, MultimodalDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateMetrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Explicit mathematical calculation of Equal Error Rate (EER) where FPR == FNR.
    Returns:
        eer: Equal Error Rate percentage
        threshold: Optimal decision threshold where FPR == FNR
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("y_true contains only 1 class. Equal Error Rate calculation defaults to 0.0.")
        return 0.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr

    diffs = np.abs(fpr - fnr)
    if np.all(np.isnan(diffs)):
        return 0.0, 0.5

    idx = np.nanargmin(diffs)
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    optimal_threshold = float(thresholds[idx])

    return eer, optimal_threshold


def evaluate_model(
    npz_path: str = "features_out/extracted_features.npz",
    model_path: str = "attention_adapter.pth",
    output_roc_path: str = "roc_curve.png"
):
    """
    Evaluates trained MultimodalGatedAttentionAdapter model on test/validation .npz features,
    calculates academic metrics (AUC-ROC, EER, F1, Precision, Recall), plots ROC curve,
    and benchmarks throughput latency/FPS.
    """
    if not os.path.exists(npz_path):
        logger.warning(f"File '{npz_path}' not found. Generating synthetic test dataset for metric evaluation.")
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        from model_and_train import create_dummy_npz
        create_dummy_npz(npz_path, num_samples=120)

    # Load dataset
    dataset = MultimodalDataset(npz_path)
    X_tensor = dataset.X.to(device)
    y_true = dataset.y.numpy()

    # Load model
    model = MultimodalGatedAttentionAdapter(
        spatial_dim=dataset.spatial_dim,
        temporal_dim=dataset.temporal_dim,
        biological_dim=dataset.biological_dim
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Successfully loaded model checkpoint from '{model_path}'")
    else:
        logger.warning(f"Checkpoint '{model_path}' not found. Evaluating on raw initialized weights.")

    model.eval()

    # Warmup pass
    with torch.no_grad():
        _ = model(X_tensor[:min(5, len(X_tensor))])

    # Benchmark Throughput Latency
    num_samples = len(X_tensor)
    start_t = time.time()
    with torch.no_grad():
        logits, attn_weights = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    end_t = time.time()

    total_time_sec = end_t - start_t
    latency_per_sample_ms = (total_time_sec / max(num_samples, 1)) * 1000.0
    throughput_fps = num_samples / max(total_time_sec, 1e-6)

    # 1. AUC-ROC Curve Calculation & Plotting
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        roc_auc = 1.0

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Gated Attention Model (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_roc_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve plot to '{output_roc_path}'")

    # 2. Equal Error Rate (EER) Calculation
    eer, eer_threshold = compute_eer(y_true, probs)

    # 3. Standard Classification Metrics
    y_pred = (probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Mean Modality Attention Weights
    mean_weights = attn_weights.mean(dim=0).cpu().numpy()

    # Print Academic Summary Report
    print("\n" + "="*70)
    print("      ACADEMIC EVALUATION REPORT: MULTIMODAL DEEPFAKE DETECTION      ")
    print("="*70)
    print(f" Dataset Size        : {num_samples} video feature samples")
    print(f" AUC-ROC Score       : {roc_auc:.4f}")
    print(f" Equal Error Rate    : {eer*100:.2f}% (at threshold tau = {eer_threshold:.4f})")
    print(f" F1-Score            : {f1:.4f}")
    print(f" Precision           : {precision:.4f}")
    print(f" Recall              : {recall:.4f}")
    print("-"*70)
    print(" Confusion Matrix    :")
    print(f"   [ [TN={cm[0,0] if cm.shape==(2,2) else 0:<4} FP={cm[0,1] if cm.shape==(2,2) else 0:<4}]")
    print(f"     [FN={cm[1,0] if cm.shape==(2,2) else 0:<4} TP={cm[1,1] if cm.shape==(2,2) else 0:<4}] ]")
    print("-"*70)
    print(" Mean Attention Weights Allocation (Dynamic Gating):")
    print(f"   Spatial Branch (EfficientNet-B0) : {mean_weights[0]:.4f}")
    print(f"   Temporal Branch (TSM)            : {mean_weights[1]:.4f}")
    print(f"   Biological Branch (POS-rPPG)     : {mean_weights[2]:.4f}")
    print("-"*70)
    print(" Inference Throughput Benchmark:")
    print(f"   Average Latency per Sample      : {latency_per_sample_ms:.2f} ms")
    print(f"   Inference Throughput (FPS)       : {throughput_fps:.1f} samples/sec")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 4: Academic Metrics Evaluation Script")
    parser.add_argument("--features", type=str, default="features_out/extracted_features.npz", help="Path to test .npz dataset")
    parser.add_argument("--model", type=str, default="attention_adapter.pth", help="Path to trained PyTorch checkpoint")
    parser.add_argument("--roc_plot", type=str, default="roc_curve.png", help="Path to save ROC curve image")
    args = parser.parse_args()

    evaluate_model(
        npz_path=args.features,
        model_path=args.model,
        output_roc_path=args.roc_plot
    )
