import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def _validate_training_inputs(X, y):
    """
    Internal helper to validate training inputs.
    """

    if X is None or y is None:
        raise ValueError("X and y cannot be None.")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must match.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix.")


# ---------------------------------------------------------
# 1️⃣ Train SVM Model
# ---------------------------------------------------------

def train_svm(X_train, y_train, kernel="linear", C=1.0):
    """
    Train a Support Vector Machine classifier.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training labels.

    kernel : str
        Kernel type for SVM. Default is 'linear'.

    C : float
        Regularization parameter.

    Returns
    -------
    model : SVC
        Trained SVM model.
    """

    _validate_training_inputs(X_train, y_train)

    model = SVC(
        kernel=kernel,
        C=C,
        probability=False
    )

    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------
# 2️⃣ Train Logistic Regression Model
# ---------------------------------------------------------

def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training labels.

    max_iter : int
        Maximum iterations for optimization.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    """

    _validate_training_inputs(X_train, y_train)

    model = LogisticRegression(
        max_iter=max_iter,
        multi_class="auto"
    )

    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------
# 3️⃣ Prediction Function
# ---------------------------------------------------------

def predict(model, X):
    """
    Generate predictions using a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier.

    X : ndarray
        Feature matrix.

    Returns
    -------
    predictions : ndarray
        Predicted class labels.
    """

    if model is None:
        raise ValueError("Model cannot be None.")

    if not hasattr(model, "predict"):
        raise TypeError("Model must implement a predict() method.")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    predictions = model.predict(X)

    return predictions