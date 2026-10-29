import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from schema import CLASS_ORDER


def score_predictions(y_true, y_pred, y_proba, proba_classes):
    proba = _align_probabilities(y_proba, proba_classes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(
            roc_auc_score(
                y_true,
                proba,
                multi_class="ovr",
                average="weighted",
                labels=CLASS_ORDER,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                labels=CLASS_ORDER,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=CLASS_ORDER,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=CLASS_ORDER,
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def report_frame(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in CLASS_ORDER + ["macro avg", "weighted avg"]:
        row = report[label]
        rows.append(
            {
                "label": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": row["support"],
            }
        )
    return rows


def matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)


def _align_probabilities(y_proba, proba_classes):
    class_to_index = {label: index for index, label in enumerate(proba_classes)}
    missing = [label for label in CLASS_ORDER if label not in class_to_index]
    if missing:
        raise ValueError(f"Model is missing classes: {missing}")
    indices = [class_to_index[label] for label in CLASS_ORDER]
    return np.asarray(y_proba)[:, indices]
