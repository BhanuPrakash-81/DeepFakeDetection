import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, save_path: str = "roc_curve.png"):
    """Plots and saves ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_confusion_matrix(cm: List[List[int]], save_path: str = "confusion_matrix.png"):
    """Plots and saves Confusion Matrix heatmap."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["REAL", "DEEPFAKE"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(j, i, format(cm_arr[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm_arr[i, j] > thresh else "black")
                     
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_feature_importances(importances: np.ndarray, top_k: int = 15, save_path: str = "feature_importances.png"):
    """Plots top K feature importances."""
    indices = np.argsort(importances)[::-1][:top_k]
    vals = importances[indices]
    
    plt.figure(figsize=(8, 5))
    plt.barh(range(top_k), vals[::-1], align="center", color="skyblue")
    plt.yticks(range(top_k), [f"Feature #{idx}" for idx in indices[::-1]])
    plt.xlabel("Feature Importance Score")
    plt.title(f"Top {top_k} Most Important Features (XGBoost)")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
