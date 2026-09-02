
An **Artificial Neural Network (ANN)** is a machine learning model that consists of **layers of interconnected nodes (neurons)** that process and transform data.

It **learns patterns and relationships** in the data through a process called **training**, by adjusting its **weights and biases**.

**Example:** Speech Recognition, Image Classification, Handwritten data.
### Basic Structure

```
        Input Layer        Hidden Layers       Output Layer

          I/P                 H1     H2             O/P
           │                   │      │              │
           ● ────────────────► ● ───► ● ───────────► ●
           │                 ↗ │    ↗ │
           ● ───────────────► ● ───► ●
           │                 ↘ │    ↘ │
           ● ────────────────► ● ───► ●
```
### Main Components

### 1. Input Layer

- The **input layer receives the input data**.
- Each neuron represents an input feature.
- It passes the data to the hidden layer.

**Example:** For predicting house prices:

```
Area | Bedrooms | Location
```

---

### 2. Hidden Layer

- Hidden layers **process the input data**.
- They learn **patterns and relationships** from the data.
- An ANN can have **one or many hidden layers**.

```
Input → Hidden Layer 1 → Hidden Layer 2 → Output
```

---

### 3. Output Layer

- The output layer produces the **final prediction or result**.
- The number of neurons depends on the problem.

**Examples:**

- Binary classification → usually 1 output
- Multi-class classification → multiple outputs
- Regression → usually 1 output

---

### 4. Neuron

- A **neuron is the basic processing unit** of an ANN.
- It receives inputs, applies weights and bias, and passes the result through an activation function.

```
Inputs → Weighted Sum + Bias → Activation Function → Output
```

---

### 5. Weights

- Weights determine the **importance of each input**.
- During training, the network **adjusts the weights** to improve predictions.
**Example: **
```
For a banking loan system, 
1. Credit Score (Highest priority)
2. Income
3. Age 
```
---

### 6. Bias

- Bias is an additional parameter added to the weighted sum.
- It helps the neuron **adjust or shift its output**.
**Example**: 
```
House Price Prediction
Area =  1000 sq ft
Weight = 0.5
Bias = 10

Z =(1000*5) + 10
  = 510
```

**Formula**: `Z = w1x1 + w2x2 + w3x3 + b`

---

### 7. Activation Function

- It introduces **non-linearity** into the neural network.
- This allows the network to learn **complex patterns**.

We pass Z to activation function and the Activation(Z) and we get the output.

**Examples:**

- ReLU
- Sigmoid
- Tanh
- Leaky ReLU
- Softmax

## Dense Layer

A **Dense layer** is a neural-network layer where **every neuron is connected to every neuron in the previous layer**.

### What is it used for?

- It **processes the input** from the previous layer.
- It learns **weights and biases** during training.
- It is commonly used in **hidden layers and output layers**.

### Example

```
Input Layer          Dense Layer

   I1 ───────────────► H1
    │ ╲             ╱ │
    │  ╲           ╱  │
   I2 ──╲─────────╱──► H2
    │    ╲       ╱    │
    │     ╲     ╱     │
   I3 ─────╲───╱─────► H3
```

Here, **every input neuron is connected to every neuron in the Dense layer**.

### In Keras

```
tf.keras.layers.Dense(10)
```

means:

> Create a Dense layer with **10 neurons**.

Each neuron has its own **weights and bias**.


## Backward Propagation (Backpropagation)

**Backpropagation** is an algorithm used to **train a neural network by minimizing the loss (error)**.

It works by calculating **how much each weight contributed to the error** and using this information to update the **weights and biases**.

### How it works

1. **Forward Propagation** → The input passes through the network to produce a prediction.
2. **Calculate Loss** → Compare the prediction with the actual/target value.
3. **Backward Propagation** → Calculate the gradients by moving backward through the network.
4. **Update Weights & Biases** → The optimizer uses the gradients to reduce the loss.

```
        FORWARD PROPAGATION
              ───────────────►

Input → Hidden 1 → Hidden 2 → Output
                              │
                              ▼
                             Loss
                              │
              ◄───────────────┘
        BACKWARD PROPAGATION
                             
      Gradients flow backward
              ↓
      Update weights & biases
```

### Chain Rule

Backpropagation uses the **chain rule of calculus** to calculate how changes in each weight affect the final loss.

> **Backpropagation = Calculate gradients → Find each parameter's contribution to the error → Update weights and biases**

![393e3466-824a-4c33-ab41-a64c33cde5a2.jpeg]



### Gradient Tape:
**GradientTape** records the operations performed during the forward pass so that TensorFlow can automatically calculate **gradients** during backpropagation

### Why do we need Optimizers?

**Backpropagation** tells us **which weights are wrong and how much they need to change**.

But we need something to **actually update the weights**.

👉 That is the job of the **optimizer**.

### Simple example:

Imagine you are trying to reach a destination:

- **Backpropagation:** Tells you **which direction to go** and how far you are from the correct path.
- **Optimizer:** Decides **how big a step to take**.

### Weight Update Using Gradient Descent

`new_weight = old_weight - learning_rate * gradient`

![[Pasted image 20260901132016.png]]



## Loss Function

A **loss function** measures **how wrong the model's predictions are compared to the actual/target values**.

- It produces a **single scalar value** called the **loss**.
- **Lower loss → better prediction**.
- During training, the model tries to **minimize the loss** by updating its weights and biases.

### Choosing the Loss Function

The loss function depends on the **type of problem**:

|Problem Type|Common Loss Functions|Used For|
|---|---|---|
|**Regression**|MSE, MAE|Predicting continuous values|
|**Binary Classification**|Binary Cross-Entropy|Two classes|
|**Multi-class Classification**|Categorical Cross-Entropy|More than two classes|

### 1. Regression

**MSE (Mean Squared Error)** and **MAE (Mean Absolute Error)** are commonly used.

Example:

```
Actual Price      = ₹500,000
Predicted Price   = ₹480,000
```

The loss measures how far the prediction is from the actual value.

### 2. Binary Classification

**Binary Cross-Entropy**

```
tf.keras.losses.BinaryCrossentropy()
```

Used when there are **two possible classes**.

Example:

```
Spam       → 1
Not Spam   → 0
```

### 3. Multi-class Classification

**Categorical Cross-Entropy**

```
tf.keras.losses.CategoricalCrossentropy()
```

Used when there are **more than two classes**.

Example:

```
Cat   → 0
Dog   → 1
Bird  → 2
```



