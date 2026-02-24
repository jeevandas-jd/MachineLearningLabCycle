import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from loader import load_images,load_labels


if __name__=="__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR= os.path.join(BASE_DIR,"..","data/raw")
    print("dir is \n ",DATA_DIR)
    X_train=load_images(os.path.join(DATA_DIR,"train-images.idx3-ubyte"))
    y_train = load_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))

    X_test  = load_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"))
    y_test  = load_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))
    # ==============================
    # 2. Normalize pixel values
    # ==============================
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    # ==============================
    # 3. One-hot encode labels
    # ==============================
    encoder = OneHotEncoder(sparse_output=False)

    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh  = encoder.transform(y_test.reshape(-1, 1))

    # ==============================
    # 4. Train–Validation split
    # ==============================
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train,
        y_train_oh,
        test_size=0.2,
        random_state=42
    )

    # ==============================
    # 5. Sanity checks
    # ==============================
    print("Training set   :", X_train.shape, y_train_oh.shape)
    print("Validation set :", X_val.shape, y_val_oh.shape)
    print("Test set       :", X_test.shape, y_test_oh.shape)

    assert X_train.shape[0] == y_train_oh.shape[0]
    assert X_val.shape[0] == y_val_oh.shape[0]
    assert X_test.shape[0] == y_test_oh.shape[0]

    # ==============================
    # 6. Save preprocessed data
    # ==============================
    os.makedirs(os.path.join(BASE_DIR,"..","data/processed"), exist_ok=True)
    data_save=os.path.join(BASE_DIR,"..","data/processed")
    np.savez(
        f"{data_save}/mnist_preprocessed.npz",
        X_train=X_train,
        y_train=y_train_oh,
        X_val=X_val,
        y_val=y_val_oh,
        X_test=X_test,
        y_test=y_test_oh
    )

    print("\nPreprocessing completed successfully")