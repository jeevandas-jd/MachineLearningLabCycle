"""
High-dimensional MNIST (784D)
            ↓
PCA (2 components)
            ↓
Logistic Regression (re-trained on PCA space)
            ↓
Meshgrid + contour plot

"""

import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if __name__=="__main__":

    BASE_DIR=os.path.dirname(os.path.abspath(__file__))

    print(BASE_DIR)

    data_path=f"{os.path.join(BASE_DIR,"..","data/processed")}/mnist_preprocessed.npz"

    data=np.load(data_path)

    # get X_train and y_train

    X_train=data["X_train"]

    y_train=data["y_train"]


    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train_lbl = np.argmax(y_train, axis=1)

    pca=PCA(n_components=2)

    X_pca=pca.fit_transform(X_train)

    model_2d = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model_2d.fit(X_pca,y_train_lbl)

    joblib.dump(model_2d,os.path.join(BASE_DIR,"..","models/logistic_model_pca.pkl"))

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ==============================
    # 6. Plot decision boundary
    # ==============================
    plt.figure(figsize=(10, 8))

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train_lbl,
        cmap="tab10",
        s=10
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Decision Boundary of Logistic Regression (PCA-reduced MNIST)")

    plt.colorbar(scatter, label="Digit Class")

    # ==============================
    # 7. Save figure
    # ==============================
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/decision_boundary.png", dpi=300)
    plt.show()

    print("Decision boundary visualization saved successfully ✅")