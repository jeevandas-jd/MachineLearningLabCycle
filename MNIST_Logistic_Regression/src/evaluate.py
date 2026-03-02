"""
++++ Evaluation Architecture +++++
Saved model (.pkl)
        ↓
Test data (.npz)
        ↓
Flatten images
        ↓
Predict labels
        ↓
Metrics computation
        ↓
Save evaluation report

"""

import joblib

import numpy as np

import os

from sklearn.metrics import ( accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report)

if __name__=="__main__":

    #load test data

    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    
    model_path=os.path.join(BASE_DIR,"..","models/logistic_model.pkl")

    data_path=os.path.join(BASE_DIR,"..","data/processed/mnist_preprocessed.npz")

    data=np.load(data_path)

    # load test data X and Y
    X_test=data["X_test"]
    y_test=data["y_test"]

    #flatten data

    X_test=X_test.reshape(X_test.shape[0],-1)

    y_test_lbl = np.argmax(y_test, axis=1)

    #load model

    model=joblib.load(model_path)

    y_pred=model.predict(X_test)


    # compute metrices

    acc  = accuracy_score(y_test_lbl, y_pred)
    prec = precision_score(y_test_lbl, y_pred, average="macro")
    rec  = recall_score(y_test_lbl, y_pred, average="macro")
    f1   = f1_score(y_test_lbl, y_pred, average="macro")
    report = classification_report(y_test_lbl, y_pred)
    result_path=os.path.join(BASE_DIR,"..","results")

    #save report

    with open (f"{result_path}/report.txt","w") as f:

        f.write("MNIST Logistic Regression Evaluation Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    # 8. Print summary
    # ==============================
    print("Evaluation completed successfully ✅\n")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nDetailed Classification Report:\n")
    print(report)

