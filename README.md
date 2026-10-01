# 🌸 ANN for Iris Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/TensorFlow-ANN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

<p align="center">
  <b>A practical Artificial Neural Network project for multiclass Iris classification.</b>
</p>

---

## 🌱 Project Overview

I built this project to understand the practical workflow of an **Artificial Neural Network (ANN)** for multiclass classification.

Instead of directly training the neural network, I first established a simple **Perceptron baseline**, then built an ANN, analyzed its training behavior, identified a generalization gap, and introduced regularization using **Dropout** and **Early Stopping**.

### 🔄 Project Flow

```text
Iris Dataset
     ↓
Data Preparation
     ↓
Label Encoding
     ↓
Train / Test Split
     ↓
Feature Scaling
     ↓
Perceptron Baseline
     ↓
ANN Development
     ↓
Training & Validation Analysis
     ↓
Generalization Gap Observed
     ↓
Dropout + Early Stopping
     ↓
Final Evaluation
```

---

## 🌼 Dataset

The project uses the **Iris dataset**.

| Property | Details |
|---|---|
| 🌸 Samples | 150 |
| 📊 Input Features | 4 |
| 🎯 Classes | 3 |
| 🧮 Problem Type | Multiclass Classification |

### Input Features

- 🌿 Sepal Length
- 🌿 Sepal Width
- 🌺 Petal Length
- 🌺 Petal Width

### Target Classes

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

The `Id` column is removed because it is only an identifier and does not provide meaningful information for classification.

---

## 🧹 Data Preparation

### 1. Feature & Target Separation

The four flower measurements are used as input features, while `Species` is used as the target.

### 2. Label Encoding

The species names are converted into integer class labels using **Scikit-learn's `LabelEncoder`**.

```text
Iris-setosa      → 0
Iris-versicolor  → 1
Iris-virginica   → 2
```

### 3. Train-Test Split

The dataset is divided into:

- 🟦 **80% Training**
- 🟩 **20% Testing**

A `random_state` of `42` is used for reproducibility, with stratification applied to preserve class proportions.

### 4. Feature Scaling

The input features are standardized using **Scikit-learn's `StandardScaler`**.

The scaler is fitted only on the training data and then used to transform both training and test data.

---

## ⚡ Perceptron Baseline

Before building the ANN, I trained a **Scikit-learn Perceptron** as a baseline.

```python
Perceptron(
    max_iter=1000,
    random_state=42
)
```

### 📌 Baseline Result

> **93.33% Test Accuracy**

This provides a simple reference point for evaluating the neural network.

---

## 🧠 Artificial Neural Network

The initial ANN uses a:

```text
4 → 16 → 8 → 3
```

architecture.

### 🏗️ Architecture

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 4 | — |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output | 3 | Softmax |

```text
             ┌──────────────┐
             │  4 Inputs    │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Dense: 16    │
             │    ReLU      │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Dense: 8     │
             │    ReLU      │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Dense: 3     │
             │   Softmax    │
             └──────────────┘
```

The four input neurons correspond to the four Iris measurements, while the three output neurons represent the three species.

The hidden-layer sizes are hyperparameters selected as the initial architecture for this experiment.

---

## ⚙️ Model Compilation

The neural network was compiled using:

| Component | Configuration |
|---|---|
| 🔧 Optimizer | Adam |
| 📉 Loss | Categorical Cross-Entropy |
| 🎯 Metric | Accuracy |

Because this is a three-class classification problem with one-hot encoded target labels, **categorical cross-entropy** is used as the loss function.

---

## 🏋️ Initial ANN Training

The initial ANN was trained with:

- 🔁 **Epochs:** 100
- 📦 **Batch Size:** 8
- 🔍 **Validation Split:** 20%
- 🔧 **Optimizer:** Adam
- 📉 **Loss:** Categorical Cross-Entropy

The training history was stored to analyze training and validation accuracy and loss.

### 📌 Initial ANN Result

> **96.67% Test Accuracy**

---

## ⚠️ Observing Overfitting

After training the initial ANN, I compared the training and validation curves.

The training loss continued to decrease while the validation loss remained considerably higher and fluctuated across epochs.

Training accuracy also reached approximately **99%**, while validation accuracy remained around **95.8%**.

This indicated a noticeable **generalization gap** and suggested mild overfitting.

Rather than increasing model complexity, I introduced regularization techniques to improve generalization.

---

## 🛡️ Reducing Overfitting

I used two techniques:

### 💧 Dropout

A dropout rate of `0.2` was added after each hidden layer.

Dropout randomly deactivates a fraction of neurons during training. This encourages the network to learn more robust patterns instead of relying heavily on individual neurons.

### ⏹️ Early Stopping

Early Stopping was configured to monitor validation loss:

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

This allows training to stop when validation performance stops improving and restores the weights from the best validation-loss point.

---

## 🔐 Regularized ANN

The modified architecture became:

```text
Input
  ↓
Dense(16, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(8, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(3, Softmax)
```

### Training Configuration

| Parameter | Value |
|---|---:|
| Maximum Epochs | 100 |
| Batch Size | 8 |
| Validation Split | 20% |
| Dropout | 0.2 |
| Early Stopping | Enabled |

The regularized model achieved **95.83% validation accuracy**, while validation loss decreased substantially during training.

---

## 🏆 Final Results

All models were evaluated on the same held-out test set.

| Model | Test Accuracy |
|---|---:|
| 🔵 Perceptron | 93.33% |
| 🟠 Original ANN | 96.67% |
| 🟢 Regularized ANN | **100.00%** |

### 🥇 Final Regularized ANN

| Metric | Result |
|---|---:|
| 🎯 Test Accuracy | **100.00%** |
| 📉 Test Loss | **0.0790** |

The final model correctly classified all **30 samples** in the test set.

> **Note:** The test set contains only 30 samples, so the 100% accuracy represents performance on this particular test split.

---

## 📊 Model Progression

```text
Perceptron
   │
   └── 93.33% Test Accuracy
              ↓
        Original ANN
   │
   └── 96.67% Test Accuracy
              ↓
    Dropout + Early Stopping
              ↓
       Regularized ANN
   │
   └── 100.00% Test Accuracy
```

The regularization step was introduced after observing the generalization gap in the initial ANN.

---

## 💡 Key Learning

This project helped me understand the practical workflow of building and improving an Artificial Neural Network.

### Concepts Covered

- 🧹 Dataset preparation
- 🔤 Label encoding
- ✂️ Train-test splitting
- 📏 Feature standardization
- 📌 Perceptron baseline
- 🧠 Neural network architecture
- ⚡ ReLU activation
- 🎯 Softmax activation
- 🔢 One-hot encoding
- 🔧 Adam optimization
- 📉 Categorical cross-entropy
- 📈 Training and validation analysis
- ⚠️ Generalization gap
- 🛡️ Dropout regularization
- ⏹️ Early Stopping
- 🧪 Test-set evaluation

---

## 🧰 Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

---

## 📁 Project Structure

```text
ANN-Iris-Classification/
│
├── 📓 ANN_implementation.ipynb
├── 📄 Iris.csv
└── 📘 README.md
```

---

## 👩‍💻 Author

### Pallavi Dahiya

**B.Tech — Computer Science & Engineering (Data Science)**

> Building practical machine learning projects and strengthening my understanding of ML concepts through implementation.
