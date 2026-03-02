"""
Tuning Architecture 

Preprocessed Training Data
        ↓
Flatten images
        ↓
Grid of hyperparameters
        ↓
Cross-validation (k-fold)
        ↓
Best Logistic Regression model
        ↓
Save tuned model + report
"""

import numpy as np

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
if __name__=="__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path=f"{os.path.join(BASE_DIR,"..","data/processed")}/mnist_preprocessed.npz"

    data=np.load(data_path)

    X_train=data["X_train"]

    y_train=data["y_train"]

    # flatten image

    X_train=X_train.reshape(X_train.shape[0],-1)

    y_train_lbl = np.argmax(y_train, axis=1)

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500, 1000]
    }

    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train_lbl)

    best_model = grid.best_estimator_

    joblib.dump(best_model,os.path.join(BASE_DIR,"..","models/logistic_model.pkl"))

    result_path=os.path.join(BASE_DIR,"..","results")
    with open (f"{result_path}/tuning_report.txt","w") as f:
        f.write("Logistic Regression Hyperparameter Tuning Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Best Parameters:\n{grid.best_params_}\n\n")
        f.write(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}\n")

    print("\nHyperparameter tuning completed ✅")
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")