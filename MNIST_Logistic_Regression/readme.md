# 📘 MNIST Digit Classification using Logistic Regression

## 📌 Project Overview

This project implements a complete machine learning pipeline to classify handwritten digits from the MNIST dataset using Logistic Regression (scikit-learn).

The pipeline includes:

- Data preprocessing  
- Model training  
- Model evaluation  
- Hyperparameter tuning (GridSearchCV)  
- Decision boundary visualization using PCA  

The implementation uses NumPy and scikit-learn only (no deep learning frameworks).

---

## 📂 Project Structure

.
├── data  
│   ├── processed  
│   │   └── mnist_preprocessed.npz  
│   └── raw  
│       ├── train-images.idx3-ubyte  
│       ├── train-labels.idx1-ubyte  
│       ├── t10k-images.idx3-ubyte  
│       └── t10k-labels.idx1-ubyte  
│  
├── models  
│   ├── logistic_model_pca.pkl  
│   └── tuned_logistic_model.pkl  
│  
├── results  
│   ├── decision_boundary.png  
│   ├── report.txt  
│   └── tuning_report.txt  
│  
└── src  
    ├── loader.py  
    ├── preprocess.py  
    ├── train.py  
    ├── evaluate.py  
    ├── tune.py  
    └── visualize.py  

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

- Raw MNIST IDX files are loaded using a custom loader.
- Pixel values are normalized to the range [0, 1].
- Labels are one-hot encoded.
- Training data is split into:
  - Training set (80%)
  - Validation set (20%)
- Processed data is saved as:

data/processed/mnist_preprocessed.npz

---

### 2️⃣ Model Training

- Images are flattened from 28×28 → 784 features.
- One-hot labels are converted back to class labels.
- A multi-class Logistic Regression model is trained using:

Solver: LBFGS  
Max iterations: 1000  
Multi-class: Auto  

The trained model is saved in:

models/tuned_logistic_model.pkl

---

### 3️⃣ Model Evaluation

The model is evaluated using:

- Accuracy  
- Precision (macro average)  
- Recall (macro average)  
- F1-score (macro average)  
- Detailed classification report  

Results are saved in:

results/report.txt

---

### 4️⃣ Hyperparameter Tuning

GridSearchCV is used to optimize:

- Regularization strength (C)  
- Solver  
- Maximum iterations  

Cross-validation is performed using 3-fold CV.

Best model and tuning results are stored in:

models/tuned_logistic_model.pkl  
results/tuning_report.txt  

---

### 5️⃣ Decision Boundary Visualization

Since MNIST has 784 dimensions, direct visualization of the decision boundary is not possible.

Therefore:

- PCA reduces dimensionality to 2D
- Logistic Regression is retrained in PCA space
- Decision boundary is plotted using contour plots

Output:

results/decision_boundary.png

Note: PCA is used only for visualization and does not affect the original trained model.

---

## ▶ How to Run the Project

### Step 1: Preprocess Data

python src/preprocess.py

### Step 2: Train Model

python src/train.py

### Step 3: Evaluate Model

python src/evaluate.py

### Step 4: Hyperparameter Tuning

python src/tune.py

### Step 5: Visualize Decision Boundary

python src/visualize.py

---

##  Expected Performance

Logistic Regression on MNIST typically achieves:

- Accuracy ≈ 92–93%
- Strong macro-averaged precision, recall, and F1-score

---

##  Technologies Used

- Python 3  
- NumPy  
- scikit-learn  
- Matplotlib  
- Joblib  

---

##  Learning Outcomes

This project demonstrates:

- End-to-end machine learning pipeline design  
- Multi-class classification using logistic regression  
- Model evaluation with standard metrics  
- Hyperparameter tuning using cross-validation  
- Dimensionality reduction using PCA  
- Visualization of decision boundaries  

---

##  Academic Summary

This project builds and evaluates a multi-class Logistic Regression model for handwritten digit classification on the MNIST dataset. The model is optimized using cross-validation-based hyperparameter tuning and interpreted using PCA-based visualization.