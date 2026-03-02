import numpy as np

import joblib
import os
from sklearn.linear_model import LogisticRegression

if __name__=="__main__":
    ############################
    
    # step 1: loading the preprocessed data

    ############################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path=f"{os.path.join(BASE_DIR,"..","data/processed")}/mnist_preprocessed.npz"

    data=np.load(data_path)

    # step 2: get the X and Y for trining 

    X_train=data["X_train"]

    y_train=data["y_train"]

    print(f"x train shape = {X_train.shape}\ny train shape = {y_train.shape}")

    # step 3: flatten image shape

    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train_lbl = np.argmax(y_train, axis=1)


    # step 4: build a logistic regression model

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    # step 5: train model

    model.fit(X_train,y_train_lbl)

    joblib.dump(model,os.path.join(BASE_DIR,"..","models/logistic_model.pkl"))

    print(f"logistic regression saved successfully {os.path.join(BASE_DIR,"..","models")}")