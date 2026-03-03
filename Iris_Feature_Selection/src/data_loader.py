from sklearn.datasets import load_iris


def load_data():
    """
    Loads the Iris dataset.

    Returns:
        X (ndarray): Feature matrix (150 samples, 4 features)
        y (ndarray): Target labels (150,)
        feature_names (list): Names of input features
        target_names (list): Names of target classes
    """
    
    iris = load_iris()
    
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names