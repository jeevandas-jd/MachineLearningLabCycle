import numpy as np
from sklearn.model_selection import train_test_split
import os
import joblib


# ---------------------------------------------------------
# Internal validation utilities
# ---------------------------------------------------------

def _validate_arrays(X, y=None):
    """
    Internal helper to validate numpy arrays.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if y is not None:
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")


# ---------------------------------------------------------
# 1️⃣ Train/Test Split
# ---------------------------------------------------------

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    X : ndarray
        Feature matrix.

    y : ndarray
        Target labels.

    test_size : float
        Fraction of dataset used for testing.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    _validate_arrays(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 2️⃣ Apply Feature Selector Safely
# ---------------------------------------------------------

def apply_feature_selector(selector, X_train, X_test):
    """
    Apply a fitted feature selector to training and test data.

    Parameters
    ----------
    selector : fitted selector object
        Must implement transform().

    X_train : ndarray
        Training features.

    X_test : ndarray
        Test features.

    Returns
    -------
    X_train_selected, X_test_selected
    """

    if selector is None:
        raise ValueError("Selector cannot be None.")

    if not hasattr(selector, "transform"):
        raise TypeError("Selector must implement transform().")

    _validate_arrays(X_train)
    _validate_arrays(X_test)

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected


# ---------------------------------------------------------
# 3️⃣ Get Selected Feature Names
# ---------------------------------------------------------

def get_feature_names(selected_indices, feature_names):
    """
    Convert selected feature indices into readable names.

    Parameters
    ----------
    selected_indices : array-like
        Indices of selected features.

    feature_names : list
        List of original feature names.

    Returns
    -------
    selected_features : list
        Names of selected features.
    """

    if selected_indices is None:
        raise ValueError("selected_indices cannot be None.")

    if feature_names is None:
        raise ValueError("feature_names cannot be None.")

    selected_features = [feature_names[i] for i in selected_indices]

    return selected_features


# ---------------------------------------------------------
# 4️⃣ Save Text Report
# ---------------------------------------------------------

def save_report(filepath, content):
    """
    Save experiment results to a text file.

    Parameters
    ----------
    filepath : str
        Path where report should be saved.

    content : str
        Text content to write.
    """

    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string.")

    if not isinstance(content, str):
        raise TypeError("content must be a string.")

    with open(filepath, "w") as f:
        f.write(content)

# ---------------------------------------------------------
# 5️⃣ Save Model
# ---------------------------------------------------------

def save_model(model, filename):
    """
    Save a trained model inside the models directory.

    Parameters
    ----------
    model : sklearn estimator
        Trained model to save.

    filename : str
        Name of the model file (e.g., "svm_model.pkl").

    Returns
    -------
    filepath : str
        Full path where the model was saved.
    """

    if model is None:
        raise ValueError("Model cannot be None.")

    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")

    models_dir = "../models"

    # create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    filepath = os.path.join(models_dir, filename)

    joblib.dump(model, filepath)

    return filepath


# ---------------------------------------------------------
# 6️⃣ Load Model
# ---------------------------------------------------------

def load_model(filepath):
    """
    Load a saved model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model file.

    Returns
    -------
    model : sklearn estimator
        Loaded model.
    """

    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")

    model = joblib.load(filepath)

    return model