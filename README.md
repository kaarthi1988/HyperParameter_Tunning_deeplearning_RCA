# Root Cause Analysis (RCA) Neural Network

This project implements a neural network to predict root causes based on symptom data. The focus is on tuning and optimizing various aspects of the neural network, such as layers, nodes, optimizers, regularizers, and dropout rates to improve the model's performance.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data](#data)
- [Model Architecture and Hyperparameters](#model-architecture-and-hyperparameters)
  - [Hyperparameter Explanations](#hyperparameter-explanations)
- [Tuning the Network](#tuning-the-network)
  - [Layers in the Network](#layers-in-the-network)
  - [Nodes in Each Layer](#nodes-in-each-layer)
  - [Optimizer Tuning](#optimizer-tuning)
  - [Regularizer and Dropout](#regularizer-and-dropout)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Project Overview

The goal of this project is to analyze and optimize a neural network model to predict root causes from symptom data. The network is built using TensorFlow/Keras and is tuned by experimenting with various configurations, such as:

1. **Number of Hidden Layers**
2. **Number of Nodes per Layer**
3. **Optimizer Selection**
4. **Regularization and Dropout** to mitigate overfitting.

---

## Installation

To run this project, you need the following dependencies:

1. Python 3.x
2. TensorFlow/Keras
3. Pandas
4. NumPy
5. Scikit-learn
6. Matplotlib (for plotting accuracy graphs)

You can install the required libraries using the following command:


---

## Data

The dataset used in this project is `root_cause_analysis.csv`, which contains symptom-related features and corresponding root cause labels. The target variable (`ROOT_CAUSE`) is transformed into categorical values for neural network classification.

Ensure the CSV file is available in the project directory.

---

## Model Architecture and Hyperparameters

The neural network architecture consists of the following components:

### **Base Model Architecture**:

1. **Input Layer**: Accepts the 7 input features from the dataset.
2. **Hidden Layers**: A configurable number of dense layers, with tunable nodes and activation functions.
3. **Output Layer**: A softmax output layer for multi-class classification of root causes (based on `ROOT_CAUSE`).

### **Hyperparameter Explanations**:

Each hyperparameter in the model plays a key role in the network's performance. Below is an explanation of each one:

#### **1. Number of Hidden Layers**:
- **Purpose**: Hidden layers help the model learn complex representations from the data. More layers typically mean that the model can capture deeper patterns, but it also increases the risk of overfitting.
- **Tuning Range**: 1 to 5 hidden layers.
- **Default**: 2 hidden layers.

#### **2. Number of Nodes per Layer**:
- **Purpose**: The number of nodes (or neurons) per layer determines the capacity of that layer to learn features from the input. More nodes increase the layer's ability to model complex patterns, but too many nodes may lead to overfitting.
- **Tuning Range**: 8 to 64 nodes per layer.
- **Default**: 64 nodes per layer.

#### **3. Activation Function**:
- **Purpose**: The activation function introduces non-linearity to the model, which allows it to learn and model complex data patterns. The **ReLU (Rectified Linear Unit)** activation function is most commonly used in hidden layers to avoid vanishing gradient problems. The **softmax** activation is used in the output layer for multi-class classification.
- **Options**: `relu` (for hidden layers), `softmax` (for output layer).
- **Default**: `relu` in hidden layers, `softmax` in the output layer.

#### **4. Optimizer**:
- **Purpose**: The optimizer defines how the model’s weights are updated based on the gradients calculated during backpropagation. Different optimizers can have varying convergence speeds and stability.
- **Tuning Options**:
  - `SGD` (Stochastic Gradient Descent): Slower but may find more generalizable solutions.
  - `RMSprop`: Adaptively changes the learning rate, which can improve training speed.
  - `Adam`: Combines the best properties of both `RMSprop` and `SGD` and is widely used.
  - `Adagrad`: Useful when the learning rate needs to adapt based on feature frequency.
- **Default**: `Adam`, as it often achieves faster convergence in deep learning.

#### **5. Loss Function**:
- **Purpose**: The loss function quantifies the difference between the predicted output and the actual output. For multi-class classification problems, **categorical cross-entropy** is widely used to penalize incorrect predictions.
- **Default**: `categorical_crossentropy`.

#### **6. Regularization**:
- **Purpose**: Regularization techniques are used to prevent overfitting by adding a penalty for larger weights. This can help the model generalize better to new, unseen data.
- **Tuning Options**:
  - `None`: No regularization applied.
  - `l1`: Adds a penalty equal to the absolute value of the weights.
  - `l2`: Adds a penalty equal to the square of the weights (also known as weight decay).
- **Default**: `None`.

#### **7. Dropout**:
- **Purpose**: Dropout randomly ignores (drops) a subset of neurons during training to make the network more robust and avoid overfitting. Dropout is typically applied to hidden layers.
- **Tuning Range**: 0.1 to 0.5 (representing the percentage of neurons dropped).
- **Default**: 0.2 (20% of neurons dropped).

#### **8. Batch Size**:
- **Purpose**: The batch size defines how many samples are processed before the model’s weights are updated. Smaller batch sizes can provide more fine-grained updates, while larger batch sizes can speed up computation.
- **Tuning Range**: 16 to 128 samples per batch.
- **Default**: 32 samples per batch.

#### **9. Learning Rate**:
- **Purpose**: The learning rate controls how much the model's weights are adjusted with respect to the loss gradient. If the learning rate is too high, the model might converge too quickly to a suboptimal solution. If it is too low, training may be slow and might get stuck in local minima.
- **Tuning Range**: 0.0001 to 0.1.
- **Default**: 0.001 (for `Adam` optimizer).

---

## Tuning the Network

### Layers in the Network
The network is evaluated by increasing the number of layers from 1 to 5. Each layer is populated with 32 nodes. The performance is measured by tracking accuracy across epochs.

### Nodes in Each Layer
The number of nodes is tuned by incrementing the node count by 8, ranging from 8 to 64 in each layer.

### Optimizer Tuning
The effect of different optimizers on model performance is evaluated by experimenting with `SGD`, `RMSprop`, `Adam`, and `Adagrad`.

### Regularizer and Dropout
To avoid overfitting, we explore different regularization techniques (`None`, `l1`, `l2`) and dropout rates (`0.2`, `0.5`).

---

## Usage

To run the project, follow these steps:

1. Clone the repository or download the project files.
2. Ensure you have the required dependencies installed.
3. Run the script to train the model based on the configurations set and display the accuracy for various tuning experiments.

---

## Results

The results from tuning experiments, including the impact of layers, nodes, optimizers, and regularization on model accuracy, will be logged and plotted for analysis.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
