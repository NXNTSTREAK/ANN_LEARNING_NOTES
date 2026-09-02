### 1. Big Data

**What is it for?**  
Big Data provides a large amount of information that AI models use for **training and learning patterns**.

**Examples:**

- Labelled datasets
    
- Image datasets
    
- Text and video datasets
    

**Purpose:** More and better data helps AI models learn more accurately.

---

### 2. Compute Power

**What is it for?**  
Compute Power provides the **processing resources needed to train and run AI models**, especially large models.

**Examples:**

- GPUs
    
- Cloud computing
    
- AWS
    
- Microsoft Azure
    
- Google Cloud (GCP)
    

**Purpose:** GPUs and cloud computing make AI training **faster and possible at large scale**.

---

### 3. Better Algorithms

**What is it for?**  
Better algorithms improve how AI models **learn, process information, and make predictions**.

**Examples:**

- **ReLU activation function** — helps reduce the vanishing-gradient problem and allows neural networks to learn effectively.
    
- **Transformers** — efficiently process sequences and capture relationships between different parts of the input.
    
- **Attention mechanism** — helps the model focus on the most relevant information.
    

**Purpose:** Better algorithms make AI models **more efficient, powerful, and accurate**.

---

## Bias and Activation Function

### What is Bias for?

**Bias** is a parameter in a neural network that helps the model **shift the activation function** and fit the training data better.

### What is an Activation Function for?

An **activation function** determines whether and how strongly a neuron should be activated. It also introduces **non-linearity**, allowing neural networks to learn complex patterns.

### Neural Network

```text
                  I/P
                   │
                   ▼
            ┌─────────────┐
            │     H1      │
            │ Weighted Sum│
            │   + Bias    │
            │    ReLU     │ ← Activation Function
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │     H2      │
            │ Weighted Sum│
            │   + Bias    │
            │    ReLU     │ ← Activation Function
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │     O/P     │
            └─────────────┘
```

### In simple words:

**Big Data → provides information**  
**Compute Power → provides processing power**  
**Better Algorithms → provide better ways to learn**  
**Bias → helps adjust the neuron's output**  
**Activation Function → helps the network learn complex patterns**


## Common Activation Functions

### 1. Sigmoid

- **Range:** `[0, 1]`
    
- **Used for:** Binary classification
    
- **Purpose:** Converts output into a probability between 0 ansd 1.
    
- **Example:** Spam or Not Spam
    

### 2. Tanh (Hyperbolic Tangent)

- **Range:** `[-1, 1]`
    
- **Used for:** Hidden layers in some neural networks
    
- **Purpose:** Produces both positive and negative values and is zero-centered.
    

### 3. ReLU (Rectified Linear Unit)

- **Range:** `[0, ∞)`
    
- **Used for:** Hidden layers
    
- **Purpose:** Introduces non-linearity and helps reduce the vanishing-gradient problem.
    
- **Formula:** `f(x) = max(0, x)`
    

### 4. Leaky ReLU

- **Range:** `(-∞, ∞)`
    
- **Used for:** Hidden layers
    
- **Purpose:** Allows a small negative value, helping reduce the **Dying ReLU problem**.
    
- **Formula:** `f(x) = max(αx, x)`
    

### 5. Softmax

- **Range:** `(0, 1)` for each class
    
- **Used for:** **Multi-class classification**
    
- **Purpose:** Converts multiple output scores into **probabilities whose total equals 1**.
    
- **Example:** Classifying an image as **Cat, Dog, or Bird**.
    

### Quick Summary

|Activation Function|Range|Common Use|Main Purpose|
|---|---|---|---|
|**Sigmoid**|`[0, 1]`|Binary classification|Probability of one class|
|**Tanh**|`[-1, 1]`|Hidden layers|Zero-centered output|
|**ReLU**|`[0, ∞)`|Hidden layers|Non-linearity & faster learning|
|**Leaky ReLU**|`(-∞, ∞)`|Hidden layers|Reduces Dying ReLU|
|**Softmax**|`(0, 1)`|Multi-class classification|Class probabilities|

### Easy Way to Remember

**Sigmoid → 2 classes**  
**Softmax → Multiple classes**  
**Tanh → −1 to +1**  
**ReLU → Positive values**  
**Leaky ReLU → ReLU with a small negative slope**


## Computational graph
It represents computations as operations and the flow of data.

Tensorflow supports eager execution where operations are executed immediately and their result can be inspected directly.

## Compile the Model

Before training, we **configure the model** by specifying an **optimizer** and a **loss function**.

### 1. Optimizer

**What is it?**  
The optimizer controls **how the model's trainable parameters (weights and biases) are updated during training**.

In simple words:

> **Optimizer = decides how to improve the model.**

It looks at the error (loss) and adjusts the **weights and biases** to reduce that error.

```
Prediction
    ↓
Calculate Loss
    ↓
Optimizer
    ↓
Update Weights & Biases
    ↓
Better Prediction
```

Common optimizer:

```
optimizer = "adam"
```

---

### 2. Loss Function

**What is it?**  
The loss function measures the **difference between the model's prediction and the target (correct) value**.

In simple words:

> **Loss = tells us how wrong the model is.**

Example:

```
Target value     = 10
Model prediction = 8

Difference/error = 2
```

The training goal is to **reduce the loss**.

Common loss functions include:

- **Mean Squared Error (MSE)** → commonly used for regression(to measure how wrong the predictions are )
- **Binary Cross-Entropy** → binary classification
- **Categorical Cross-Entropy** → multi-class classification

---

### The relationship between them

This is the part you should remember:

```
              INPUT
                │
                ▼
        ┌───────────────┐
        │ Neural Network│
        └───────┬───────┘
                │
                ▼
           Prediction
                │
                ▼
        ┌───────────────┐
        │  Loss Function│
        └───────┬───────┘
                │
                ▼
              Loss
           "How wrong?"
                │
                ▼
        ┌───────────────┐
        │   Optimizer   │
        └───────┬───────┘
                │
                ▼
       Update Weights & Bias
                │
                ▼
          Train Again
```

## Stochastic Gradient Descent (SGD)

- **SGD** is one of the most fundamental optimization algorithms used to **train neural networks**.
- It updates the model's **weights and biases** to reduce the **loss/error**.
- It calculates the gradient using **one training example (or a small batch)** at a time.
- The process is repeated many times until the model's predictions improve.

### Simple Flow

```
Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Calculate Loss
  ↓
Calculate Gradient
  ↓
SGD updates Weights & Biases
  ↓
Better Prediction
```

### Easy to Remember

> **SGD = Calculate the error → find the direction to improve → update weights & biases.**

### Workflow

        DATA
          │
          ▼
     BUILD MODEL
          │
          ▼
       COMPILE
          │
          ▼
        TRAIN
          │
          ▼
        PREDICT


