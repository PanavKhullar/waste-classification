"""
metrics.py — Metric computation utilities.

Uses scikit-learn for proper multi-class metric computation.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute overall accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Compute macro-averaged F1 score.

    WHY MACRO F1?
    Accuracy can be misleading on imbalanced datasets.
    If 'trash' is 5.4% of the data, a model that never predicts 'trash'
    still gets ~94.6% accuracy. Macro F1 averages F1 across ALL classes
    equally, so failing on the minority class tanks the score.
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Compute a full set of classification metrics.

    Returns:
        Dict with accuracy, macro_f1, macro_precision, macro_recall.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> str:
    """Generate a formatted per-class classification report."""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Compute the confusion matrix."""
    return confusion_matrix(y_true, y_pred)
