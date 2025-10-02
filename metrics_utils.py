# metrics_utils.py

import numpy as np
from sklearn.metrics import roc_auc_score

def ecdf_fit(errors_normal: np.ndarray):
    xs = np.sort(errors_normal.astype(np.float64))
    return xs

def ecdf_cdf(xs_sorted: np.ndarray, x: np.ndarray):
    cdf_vals = np.searchsorted(xs_sorted, x, side="right") / float(len(xs_sorted) + 1e-9)
    return cdf_vals

def prob_normal_from_error(errors: np.ndarray, xs_sorted: np.ndarray):
    cdf_vals = ecdf_cdf(xs_sorted, errors)
    p_norm = 1.0 - cdf_vals
    return np.clip(p_norm, 0.0, 1.0)

def aggregate_errors_per_study(errors_per_slice: np.ndarray, mode: str="mean"):
    if mode == "mean":
        return float(np.mean(errors_per_slice))
    elif mode == "median":
        return float(np.median(errors_per_slice))
    elif mode == "p75":
        return float(np.percentile(errors_per_slice, 75))
    else:
        return float(np.mean(errors_per_slice))

_Z_95 = 1.959963984540054

def _wilson_ci(k: int, n: int, z: float = _Z_95):
    """Wilson score interval для биномиальной доли (устойчивее Вальда).
    Возвращает (low, high). Если n==0 -> (nan, nan).
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    half_width = z * np.sqrt((p*(1.0 - p) + (z**2)/(4*n)) / n) / denom
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return float(lo), float(hi)

def _auc_ci_hanley_mcneil(y_true: np.ndarray, scores: np.ndarray, z: float = _Z_95):
    """95% ДИ AUC по Hanley & McNeil (1982).
    Возвращает (auc, low, high). Если класс один -> (nan, nan, nan).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan"), float("nan"), float("nan")

    auc = float(roc_auc_score(y_true, scores))

    q1 = auc / (2.0 - auc + 1e-12)
    q2 = (2.0 * auc * auc) / (1.0 + auc + 1e-12)
    se = np.sqrt((auc*(1.0 - auc) + (pos - 1.0)*(q1 - auc*auc) + (neg - 1.0)*(q2 - auc*auc)) / (pos*neg))
    lo = max(0.0, auc - z*se)
    hi = min(1.0, auc + z*se)
    return auc, float(lo), float(hi)

def compute_metrics(y_true: np.ndarray, p_pathology: np.ndarray, threshold: float=0.5, alpha: float=0.95):
    """
    Возвращает метрики и 95% ДИ:
      - AUC (+ ДИ Hanley–McNeil)
      - Accuracy (+ Wilson ДИ)
      - Sensitivity, Specificity (+ Wilson ДИ по соответствующим подвыборкам)
    """
    y_true = y_true.astype(int)
    p_pathology = p_pathology.astype(np.float64)

    auc, auc_lo, auc_hi = _auc_ci_hanley_mcneil(y_true, p_pathology)

    y_pred = (p_pathology >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    acc = (tp + tn) / max(n, 1)
    acc_ci = _wilson_ci(tp + tn, n)

    pos = tp + fn
    neg = tn + fp
    sens = tp / max(pos, 1)
    spec = tn / max(neg, 1)
    sens_ci = _wilson_ci(tp, pos)
    spec_ci = _wilson_ci(tn, neg)

    return {
        "AUC": float(auc),
        "AUC_CI95": [auc_lo, auc_hi],
        "Accuracy": float(acc),
        "Accuracy_CI95": [acc_ci[0], acc_ci[1]],
        "Specificity": float(spec),
        "Specificity_CI95": [spec_ci[0], spec_ci[1]],
        "Sensitivity": float(sens),
        "Sensitivity_CI95": [sens_ci[0], sens_ci[1]],
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }
