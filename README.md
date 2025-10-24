# 🧠 Support Vector Machine (SVM) from Scratch using Python

This project implements **Support Vector Machine (SVM)** completely from scratch using **Python + NumPy** — without using any machine learning libraries like scikit-learn.

The goal is to deeply understand how SVM works mathematically and programmatically behind the scenes 🔍

---

## 🧩 Project Overview

Support Vector Machine (SVM) is a **supervised learning algorithm** used for **classification tasks**.

It tries to find the **best separating hyperplane** that divides the data into different classes with the **maximum margin** — meaning the largest possible distance between the decision boundary and the nearest data points of each class.

Those nearest points are called **Support Vectors**.

---

## ⚙️ Implementation Details

We built an `SVM` class that contains three main methods:

### 1️⃣ `__init__()`
Initializes the hyperparameters:
- `learning_rate` → how fast the model updates weights.
- `lambda_param` → regularization term to avoid overfitting.
- `n_iters` → number of iterations for training.

### 2️⃣ `fit(X, y)`
Trains the model using **Gradient Descent**:
- Computes the margin condition for each sample.
- Updates weights `w` and bias `b` to maximize the margin between classes.
- Penalizes samples that violate the margin condition.

### 3️⃣ `predict(X)`
Predicts the class of each sample:
- Calculates the linear function `np.dot(X, w) - b`
- Returns `+1` or `-1` based on which side of the hyperplane the sample lies on.

---

## 🧮 Mathematical Intuition

The main objective of SVM is to **find `w` and `b`** that minimize the following cost function:

1/2 * ||w||² + C * Σ(max(0, 1 - yᵢ(w·xᵢ - b)))

yaml
Copy code

Where:
- `||w||²` → controls the margin width  
- `C` (or `1/λ`) → controls the penalty for misclassified samples  
- `yᵢ` → true labels (+1 or -1)  
- `xᵢ` → feature vector  

---

## 💻 Example Code

```python
import numpy as np

# Sample data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [4, 5],
    [1, 0],
    [2, 1]
])
y = np.array([-1, -1, -1, 1, 1, 1])

# Train model
model = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print("Predictions:", predictions)
Output:

makefile
Copy code
Predictions: [-1 -1 -1  1  1  1]
✅ The model successfully separates the two classes!

📊 Intuitive Summary
SVM finds the most confident decision boundary,
not just any line — but the one that maximizes the margin between different classes.

That’s why it’s known as a maximum-margin classifier.

🚀 Why Build It From Scratch?
Because when you code it yourself:

You understand how SVM actually learns.

You see how mathematical optimization happens behind the scenes.

You’re better prepared to debug or fine-tune real-world ML models.

🧱 Next Steps
This project is part of the "Machine Learning From Scratch" educational series,
where we rebuild core algorithms step by step using pure Python and NumPy.

Coming next in the series:

Logistic Regression

Decision Tree

Random Forest

K-Means

PCA

Naive Bayes

Perceptron

AdaBoost

LDA

🏷️ Hashtags
#MachineLearning #Python #SVM #FromScratch #AI #DataScience #DeepLearning #MLBeginners #Education #LearningJourney


