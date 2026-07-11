"""Classification metrics. Jaundice (label 1) is the positive/clinical class."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix


def classification_metrics(y_true, prob_pos, thresh: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_pos, dtype=float)
    pred = (p >= thresh).astype(int)
    out = {
        "acc": float((pred == y).mean()),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")   # recall on jaundice
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return out
