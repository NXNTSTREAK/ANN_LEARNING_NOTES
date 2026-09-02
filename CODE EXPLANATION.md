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

