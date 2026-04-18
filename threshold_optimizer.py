import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, y_scores, metric='micro_f1', thresholds=None):
    """
    Find the best global threshold for multi-label classification.

    Args:
        y_true: (N, L) binary ground truth
        y_scores: (N, L) predicted scores/probabilities
        metric: 'micro_f1', 'macro_f1', or 'exact_match'
        thresholds: custom threshold range, default 0.1~0.9 step 0.01

    Returns:
        best_threshold, best_score
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.01)

    best_score = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        if metric == 'micro_f1':
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        elif metric == 'macro_f1':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'exact_match':
            score = (y_true == y_pred).all(axis=1).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def find_per_label_threshold(y_true, y_scores, metric='f1'):
    """
    Find the best threshold for each label independently.
    Useful for highly imbalanced multi-label scenarios.

    Args:
        y_true: (N, L) binary ground truth
        y_scores: (N, L) predicted scores/probabilities
        metric: 'f1' (per-label F1)

    Returns:
        best_thresholds: (L,) array of best thresholds per label
    """
    num_labels = y_true.shape[1]
    best_thresholds = np.zeros(num_labels)

    for i in range(num_labels):
        y_true_col = y_true[:, i]
        y_scores_col = y_scores[:, i]

        # Skip if all zeros or all ones
        if y_true_col.sum() == 0 or y_true_col.sum() == len(y_true_col):
            best_thresholds[i] = 0.5
            continue

        best_score = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            y_pred = (y_scores_col >= t).astype(int)
            score = f1_score(y_true_col, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t
        best_thresholds[i] = best_t

    return best_thresholds
