"""Classification metrics: accuracy, macro / weighted F1, per-class precision/recall/F1, confusion matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np



@dataclass(frozen=True)
class ClassificationReport:
    summary: dict[str, float]
    per_class: list[dict[str, float | int | str]]
    confusion_matrix: list[list[int]]


def classification_report(
    targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> ClassificationReport:
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    size = len(class_names)
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (targets_array, predictions_array), 1)
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    total = int(matrix.sum())
    summary = {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()) if support.sum() else 0.0,
    }
    per_class = [
        {
            "class": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    ]
    return ClassificationReport(summary, per_class, matrix.tolist())


