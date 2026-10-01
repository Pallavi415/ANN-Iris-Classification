# ANN for Iris Classification

A practical implementation of an Artificial Neural Network for multiclass classification using the Iris dataset.

In this project, I follow a complete machine learning workflow: preparing the dataset, establishing a Perceptron baseline, building a neural network with TensorFlow/Keras, analyzing training and validation performance, identifying a generalization gap, and applying regularization to improve generalization.

---

## Project Overview

The goal of this project is to classify Iris flowers into three species based on their physical measurements.

The workflow includes:

1. Dataset preparation
2. Feature and target separation
3. Label encoding
4. Train-test splitting
5. Feature scaling
6. Perceptron baseline
7. Artificial Neural Network
8. Model training and validation
9. Training and validation analysis
10. Overfitting analysis
11. Dropout and Early Stopping
12. Final test evaluation

---

## Dataset

The project uses the **Iris dataset**.

The dataset contains:

- **150 samples**
- **4 input features**
- **3 target classes**

### Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The `Id` column is removed because it is only an identifier and does not provide meaningful information for classification.

---

## Data Preparation

### Feature and Target Separation

The four flower measurements are used as input features, while `Species` is used as the target variable.

The identifier column is removed before model training.

### Label Encoding

The categorical species names are converted into integer class labels using Scikit-learn's `LabelEncoder`.

```text
Iris-setosa      → 0
Iris-versicolor  → 1
Iris-virginica   → 2
```

### Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

A `random_state` of `42` is used for reproducibility, and stratification is applied to preserve the class distribution.

### Feature Scaling

The input features are standardized using Scikit-learn's `StandardScaler`.

The scaler is fitted only on the training data and then used to transform both the training and test sets. This prevents information from the test set from influencing the scaling process.

---

## Perceptron Baseline

Before building the neural network, I trained a Scikit-learn Perceptron as a baseline model.

```python
Perceptron(
    max_iter=1000,
    random_state=42
)
```

### Baseline Result

**Test Accuracy: 93.33%**

The Perceptron provides a simple baseline against which the neural network can be compared.

---

## Building the Neural Network

The ANN uses the following architecture:

```text
4 → 16 → 8 → 3
```

### Architecture

- **4 input neurons** — one for each Iris feature
- **16 neurons** — first hidden layer
- **8 neurons** — second hidden layer
- **3 output neurons** — one for each Iris species

ReLU activation is used in the hidden layers, while Softmax is used in the output layer.

The hidden-layer sizes are hyperparameters selected as the initial architecture for this experiment.

### Model

```python
model = Sequential(
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
)
```

---

## Model Compilation

The neural network is compiled using:

| Component | Configuration |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |

Since this is a three-class classification problem with one-hot encoded target labels, categorical cross-entropy is used as the loss function.

---

## Initial ANN Training

The initial ANN was trained with:

- **Epochs:** 100
- **Batch Size:** 8
- **Validation Split:** 20%
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy

The training history was stored and used to visualize training and validation accuracy and loss.

### Initial ANN Result

**Test Accuracy: 96.67%**

---

## Observing Overfitting

After training the initial ANN, I analyzed the training and validation curves.

The training loss continued to decrease while the validation loss remained considerably higher and fluctuated across epochs. Training accuracy also reached approximately 99%, while validation accuracy remained around 95.8%.

This indicated a noticeable generalization gap and suggested mild overfitting.

Instead of increasing the model complexity, I introduced regularization techniques to improve generalization.

---

## Reducing Overfitting

Two techniques were introduced:

### Dropout

A dropout rate of `0.2` was added after each hidden layer.

Dropout randomly deactivates a fraction of neurons during training. This encourages the network to learn more robust patterns instead of relying heavily on individual neurons.

### Early Stopping

Early Stopping was configured to monitor validation loss.

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

This allows training to stop when validation performance stops improving and restores the weights from the best validation-loss point.

---

## Regularized ANN

The modified architecture is:

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

The regularized model was trained using:

- **Maximum Epochs:** 100
- **Batch Size:** 8
- **Validation Split:** 20%
- **Dropout:** 0.2
- **Early Stopping:** Enabled

The validation accuracy reached **95.83%**, while the validation loss decreased substantially during training.

---

## Final Results

The models were evaluated on the same held-out test set.

| Model | Test Accuracy |
|---|---:|
| Perceptron | 93.33% |
| Original ANN | 96.67% |
| Regularized ANN | **100.00%** |

### Regularized ANN

- **Test Loss:** 0.0790
- **Test Accuracy:** 100.00%

The final model correctly classified all 30 samples in the test set.

Because the test set contains only 30 samples, this 100% accuracy represents performance on this particular test split.

---

## Key Learning

This project helped me understand the practical workflow of building and improving an Artificial Neural Network.

### Concepts covered

- Dataset preparation
- Feature and target separation
- Label encoding
- Train-test splitting
- Feature standardization
- Perceptron baseline
- Neural network architecture
- ReLU activation
- Softmax activation
- One-hot encoding
- Adam optimization
- Categorical cross-entropy
- Training and validation analysis
- Generalization gap
- Dropout regularization
- Early Stopping
- Test-set evaluation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Google Colab
- Jupyter Notebook

---

## Project Structure

```text
ANN-Iris-Classification/
│
├── ANN_implementation.ipynb
├── Iris.csv
└── README.md
```

---

## Author

**Pallavi Dahiya**

B.Tech — Computer Science & Engineering (Data Science)
