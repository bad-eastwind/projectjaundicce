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
        "threshold": float(thresh),
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


def pick_threshold(y_true, prob_pos, mode: str = "youden",
                   target_sensitivity: float = 0.95) -> float:
    """Choose an operating threshold on a (val) split.
      youden        - maximize sensitivity+specificity-1 (balanced operating point)
      target_sens   - smallest threshold whose sensitivity >= target (screening: recall-first)
    Falls back to 0.5 when a split has a single class.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_pos, dtype=float)
    if y.min() == y.max():
        return 0.5
    cand = np.unique(np.concatenate([[0.0, 1.0], p]))
    pos, neg = (y == 1), (y == 0)
    best_t, best_score = 0.5, -1.0
    chosen = None
    for t in cand:
        pred = p >= t
        sens = pred[pos].mean() if pos.any() else 0.0
        spec = (~pred[neg]).mean() if neg.any() else 0.0
        if mode == "target_sens":
            if sens >= target_sensitivity and spec > best_score:   # highest specificity meeting recall
                best_score, chosen = spec, float(t)
        else:  # youden
            j = sens + spec - 1.0
            if j > best_score:
                best_score, best_t = j, float(t)
    return chosen if (mode == "target_sens" and chosen is not None) else best_t


# ITA (Individual Typology Angle) skin-tone strata, dermatology convention (light -> dark).
ITA_EDGES = [55.0, 41.0, 28.0, 10.0, -30.0]
ITA_NAMES = ["very_light", "light", "intermediate", "tan", "brown", "dark"]


def ita_stratum(ita: float) -> str:
    for name, edge in zip(ITA_NAMES, ITA_EDGES):
        if ita > edge:
            return name
    return ITA_NAMES[-1]


def fairness_report(y_true, prob_pos, ita, thresh: float = 0.5) -> dict:
    """Per-skin-tone-stratum metrics + worst-group gaps. `ita` = per-sample Individual Typology Angle.
    Substantiates the skin-tone-fairness claim (otherwise unmeasured)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_pos, dtype=float)
    ita = np.asarray(ita, dtype=float)
    strata = np.array([ita_stratum(v) for v in ita])
    per = {}
    for name in ITA_NAMES:
        m = strata == name
        n = int(m.sum())
        if n == 0:
            continue
        s = classification_metrics(y[m], p[m], thresh)
        per[name] = {"n": n, "pos": int(y[m].sum()), **{k: s[k] for k in
                     ("balanced_acc", "sensitivity", "specificity", "auc")}}
    def gap(key):
        vals = [v[key] for v in per.values() if v.get(key) == v.get(key)]  # drop nan
        return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    return {"per_stratum": per, "bacc_gap": gap("balanced_acc"),
            "sensitivity_gap": gap("sensitivity")}
