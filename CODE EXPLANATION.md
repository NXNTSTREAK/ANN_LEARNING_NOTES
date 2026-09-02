```
model = tf.keras.Sequential([

	tf.keras.layers.Flatten(input_shape=(28, 28)),
	
	tf.keras.layers.Dense(128, activation='relu'),
	
	tf.keras.layers.Dense(64, activation='relu'),
	
	tf.keras.layers.Dense(10, activation='softmax')

])

  

model.summary()
```

**### Explanation:

This code creates a **3-layer ANN model** to recognize handwritten digits from the MNIST dataset.

- **Flatten:** Converts the 28 × 28 image into **784 numbers**.
    
- **Dense 128:** Has **128 neurons** and learns patterns from the image.
    
- **Dense 64:** Has **64 neurons** and learns more features.
    
- **Dense 10:** Has **10 neurons**, one for each digit **0–9**.
    
- **ReLU:** Helps the hidden layers learn patterns.
    
- **Softmax:** Gives the probability of each digit.
    
- **model.summary():** Shows the complete model information.

`MNIST stands for the Modified National Institute of Standards and Technology database, which is a classic dataset of 70,000 small grayscale images of handwritten digits used to train and test machine learning systems`

### Simple flow:

**Image (28×28) → 784 → 128 → 64 → 10 → Predicted Digit**

For example, if the model sees a handwritten **7**, the output layer will give the highest probability to **7**.

### How does the ANN learn?

Suppose we give the model a picture of **7**.

**Step 1 — Give image**

- The model gets a 28×28 image.
- `Flatten` changes it into **784 numbers**.

**Step 2 — Make a guess**

- The network passes these numbers through:  
    **784 → 128 → 64 → 10**
- At first, the model may guess **3** instead of **7**.

**Step 3 — Check the mistake**

- Correct answer = **7**
- Model's answer = **3**
- The model calculates **how wrong it was**.

**Step 4 — Correct itself**

- The network changes its internal **weights**.
- These weights control how strongly each input affects the answer.

**Step 5 — Try again**

- It sees another image.
- Makes a guess.
- Checks the answer.
- Changes the weights again.

This happens **thousands of times**.

---

## Forward Propagation

**Forward Propagation** is the process of **passing input data forward through the neural network to produce a prediction**.

### Flow

```
Input Layer
     │
     ▼
Hidden Layer 1
     │
     ▼
Hidden Layer 2
     │
     ▼
Output Layer
     │
     ▼
Prediction
```

During forward propagation, **each neuron performs two main operations**:

### 1. Linear Transformation

The neuron calculates a **weighted sum of inputs and adds a bias**.

**Formula:**

z=Wx+bz = Wx + b

Where:

- **x** = input
- **W** = weight
- **b** = bias
- **z** = result of linear transformation

Example:

```
Input → × Weight → + Bias → z
```

### 2. Non-Linear Transformation

The result from the linear transformation is passed through an **activation function**.

a=f(z)a = f(z)

Where:

- **z** = linear transformation result
- **f** = activation function
- **a** = output of the neuron

Examples of activation functions:

- ReLU
- Sigmoid
- Tanh
- Leaky ReLU
- Softmax

### Complete Process

```
        Input (x)
            │
            ▼
   ┌─────────────────┐
   │ Linear          │
   │ z = Wx + b      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Non-Linear      │
   │ a = f(z)        │
   └────────┬────────┘
            │
            ▼
       Next Layer
```

---


Dropout & L2 normalization

```


from tensorflow.keras import regularizers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = 'relu',
                          kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation = 'relu',
                          kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
    
])
```


This code builds a **neural network for image classification** (most likely MNIST handwritten digits) and uses **Dropout + L2 regularization** to reduce overfitting.

### Code explanation

```
from tensorflow.keras import regularizers
```

Imports Keras's `regularizers` module so we can apply **L2 regularization** to the neural network weights.

---

```
model = tf.keras.Sequential([
```

Creates a **Sequential model**, meaning the layers are arranged one after another:

**Input → Dense → Dropout → Dense → Dropout → Output**

---

### 1. Flatten layer

```
tf.keras.layers.Flatten(input_shape=(28,28))
```

The input image is **28 × 28 pixels**.

Flatten converts the 2D image into a 1D array:

```
28 × 28 = 784 values
```

So:

```
(28, 28) → (784)
```

This is required because the Dense layer expects a one-dimensional input.

---

### 2. First Dense layer + L2 regularization

```
tf.keras.layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.001)
)
```

Creates a fully connected layer with **128 neurons**.

#### `activation='relu'`

ReLU stands for **Rectified Linear Unit**:

```
ReLU(x) = max(0, x)
```

It introduces non-linearity, allowing the network to learn more complex patterns.

#### `kernel_regularizer=regularizers.l2(0.001)`

This applies **L2 regularization** to the weights.

L2 adds a penalty when weights become too large:

```
Loss = original loss + 0.001 × sum(weights²)
```

The goal is to prevent the model from relying too heavily on particular weights and therefore **reduce overfitting**.

---

### 3. Dropout

```
tf.keras.layers.Dropout(0.3)
```

During training, randomly disables **30% of the neurons** in this layer.

For example:

```
Before dropout:
● ● ● ● ● ● ● ● ● ●

After dropout:
●   ● ●   ●   ● ●
```

The neurons selected for dropout change randomly during each training step.

This forces the network to learn more robust features instead of depending on specific neurons.

**Important:** Dropout is only active during training, not during prediction/testing.

---

### 4. Second Dense layer + L2

```
tf.keras.layers.Dense(
    64,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.001)
)
```

Creates another fully connected layer with **64 neurons**.

It again uses:

- **ReLU** → introduces non-linearity
- **L2 regularization** → discourages excessively large weights

So the network gradually reduces its representation:

```
784 → 128 → 64
```

---

### 5. Second Dropout

```
tf.keras.layers.Dropout(0.2)
```

Randomly disables **20% of the neurons** during training.

This provides another layer of protection against overfitting.

---

### 6. Output layer

```
tf.keras.layers.Dense(10, activation='softmax')
```

There are **10 neurons** because MNIST has 10 possible digit classes:

```
0 1 2 3 4 5 6 7 8 9
```

`softmax` converts the outputs into probabilities.

For example:

```
0 → 0.01
1 → 0.02
2 → 0.03
3 → 0.01
4 → 0.05
5 → 0.82  ← highest
6 → 0.01
7 → 0.02
8 → 0.02
9 → 0.01
```

The model predicts **5** in this example.

---

### Overall architecture

```
28 × 28 image
      ↓
   Flatten
      ↓
    784
      ↓
Dense(128, ReLU + L2)
      ↓
Dropout(30%)
      ↓
Dense(64, ReLU + L2)
      ↓
Dropout(20%)
      ↓
Dense(10, Softmax)
      ↓
  Digit prediction
```

### Why both Dropout and L2?

Both are **regularization techniques**, but they work differently:

|Technique|What it does|
|---|---|
|**Dropout**|Randomly removes neurons during training|
|**L2**|Penalizes large weights|
|**Together**|Helps reduce overfitting from two different directions|

So the main purpose of this architecture is:

> **Build a classifier while controlling overfitting using both Dropout and L2 regularization.**
