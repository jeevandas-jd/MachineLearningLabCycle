import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def _validate_inputs(y_true, y_pred):
    """
    Internal validation for evaluation inputs.
    """

    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred cannot be None.")

    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a numpy array.")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a numpy array.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")


# ---------------------------------------------------------
# 1️⃣ Compute Core Metrics
# ---------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for classification.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels.

    y_pred : ndarray
        Predicted labels.

    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics.
    """

    _validate_inputs(y_true, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    return metrics


# ---------------------------------------------------------
# 2️⃣ Generate Classification Report
# ---------------------------------------------------------

def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a detailed classification report.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels.

    y_pred : ndarray
        Predicted labels.

    target_names : list, optional
        Class names for readability.

    Returns
    -------
    report : str
        Text classification report.
    """

    _validate_inputs(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
    )

    return report


# ---------------------------------------------------------
# 3️⃣ Complete Evaluation Pipeline
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.

    X_test : ndarray
        Test feature matrix.

    y_test : ndarray
        Ground truth labels.

    Returns
    -------
    results : dict
        Dictionary containing predictions and metrics.
    """

    if model is None:
        raise ValueError("Model cannot be None.")

    if not hasattr(model, "predict"):
        raise TypeError("Model must implement a predict() method.")

    if not isinstance(X_test, np.ndarray):
        raise TypeError("X_test must be a numpy array.")

    if not isinstance(y_test, np.ndarray):
        raise TypeError("y_test must be a numpy array.")

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    results = {
        "predictions": y_pred,
        "metrics": metrics,
    }

    return results