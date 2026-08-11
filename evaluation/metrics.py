import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Any, Tuple
from utils.device import get_memory_usage

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    latency_ms: float = 0.0,
    fps: float = 0.0
) -> Dict[str, Any]:
    """
    Computes classification performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    along with benchmark latency, FPS, and peak memory usage.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    mem_stats = get_memory_usage()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "latency_ms": latency_ms,
        "fps": fps,
        "memory": mem_stats
    }
