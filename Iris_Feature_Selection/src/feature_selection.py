import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def _validate_inputs(X, y):
    """
    Internal utility to validate feature selection inputs.
    """
    if X is None or y is None:
        raise ValueError("X and y cannot be None.")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must match.")
    

    # ---------------------------------------------------------
#    2 Univariate Feature Selection
# ---------------------------------------------------------

def select_univariate(X_train, y_train, k=2):
    """
    Select top-k features using ANOVA F-test.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training labels.

    k : int
        Number of features to select.

    Returns
    -------
    selector : SelectKBest
        Fitted feature selector.

    selected_indices : ndarray
        Indices of selected features.

    scores : ndarray
        ANOVA F-test scores for each feature.
    """

    _validate_inputs(X_train, y_train)

    if k <= 0 or k > X_train.shape[1]:
        raise ValueError("k must be between 1 and number of features.")

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_

    return selector, selected_indices, scores


# ---------------------------------------------------------
#  2 Random Forest Feature Importance
# ---------------------------------------------------------

def select_rf_importance(X_train, y_train, n_estimators=200, random_state=42):
    """
    Compute feature importance using Random Forest.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training labels.

    n_estimators : int
        Number of trees in the forest.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest model.

    importances : ndarray
        Importance score of each feature.

    ranked_indices : ndarray
        Features ranked by importance.
    """

    _validate_inputs(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    return model, importances, ranked_indices

# 3️⃣ Recursive Feature Elimination (RFE) with SVM


def select_rfe(X_train, y_train, n_features=2):
    """
    Perform Recursive Feature Elimination using a linear SVM.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training labels.

    n_features : int
        Number of features to select.

    Returns
    -------
    selector : RFE
        Fitted RFE selector.

    selected_indices : ndarray
        Indices of selected features.

    ranking : ndarray
        Ranking of all features.
    """

    _validate_inputs(X_train, y_train)

    if n_features <= 0 or n_features > X_train.shape[1]:
        raise ValueError("n_features must be between 1 and number of features.")

    estimator = SVC(kernel="linear")

    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X_train, y_train)

    selected_indices = selector.get_support(indices=True)
    ranking = selector.ranking_

    return selector, selected_indices, ranking