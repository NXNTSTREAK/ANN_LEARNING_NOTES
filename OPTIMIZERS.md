**Optimizers** are algorithms used to **update the weights and biases during training**.

They use the **gradients calculated during backpropagation** to determine:

- **Direction** → which way to change the parameters
- **Magnitude** → how much to change them

---

### SGD (Stochastic Gradient Descent)

**SGD** is one of the simplest and most fundamental optimization algorithms.

It uses the gradients from **backpropagation** to update the model's weights and biases.

### Gradient Descent Update Rule

w_new = w_old - learning_rate * gradient

Where:

- **w_new** → updated weight
- **w_old** → current weight
- **learning_rate** → controls the size of the update
- **gradient** → tells the direction in which the loss increases

### Example

Suppose:

```
Old weight       = 5
Learning rate    = 0.1
Gradient         = 2
```

Then:

```
New weight = 5 - (0.1 × 2)
           = 5 - 0.2
           = 4.8
```

So the weight changes:

```
5  ───────►  4.8
     update
```

## Types of Gradient Descent

Gradient Descent can be classified based on **how much training data is used to calculate the gradient and update the weights**.

### 1. Batch Gradient Descent (Batch GD)

- Uses the **entire training dataset** to calculate the gradient.
- Updates the weights **once after processing the complete dataset**.
- Stable, but can be slow for very large datasets.

```
Entire Dataset
      ↓
Calculate Gradient
      ↓
Update Weights
```

**Example:**  
If you have **1,000 samples**, all 1,000 samples are used before one weight update.

---

### 2. Stochastic Gradient Descent (SGD)

- Uses **one sample at a time** to calculate the gradient.
- Updates the weights after **each individual sample**.
- Faster updates, but the training path can be noisy.

```
Sample 1 → Gradient → Update
Sample 2 → Gradient → Update
Sample 3 → Gradient → Update
      ↓
     ...
```

**Example:**  
For 1,000 samples → potentially **1,000 updates per epoch**.

---

### 3. Mini-Batch Gradient Descent

- Divides the dataset into **small batches**.
- Uses one batch to calculate the gradient.
- Updates the weights after processing each batch.
- It provides a good balance between **Batch GD and SGD**.

```
Dataset
   ↓
┌────────────┐
│ Batch 1    │ → Gradient → Update
├────────────┤
│ Batch 2    │ → Gradient → Update
├────────────┤
│ Batch 3    │ → Gradient → Update
└────────────┘
```

**Example:**  
If you have 1,000 samples and batch size = 32:

```
1000 samples
     ↓
32 → Update
32 → Update
32 → Update
...
```

### Batch Size

A common mini-batch size is **32, 64, 128, or 256 samples**.

> **Note:** 32–256 is a common practical range, not a strict rule. The best batch size depends on the dataset, model, and available hardware..



| Optimizer             | Overview                                                                                      | Example / Use Case       |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------ |
| **1. SGD**            | Uses a **fixed learning rate** to update weights and biases using gradients.                  | Simple problems          |
| **2. SGD + Momentum** | Adds a **momentum/velocity term** to smooth updates and speed up learning.                    | Image classification     |
| **3. Adam**           | Combines ideas from **Momentum and RMSprop** and adapts the learning rate for each parameter. | Most deep-learning tasks |
| **4. RMSprop**        | **Adapts the learning rate for each parameter** based on recent gradients.                    | RNNs, NLP tasks          |


### Momentum :
