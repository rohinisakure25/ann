import numpy as np
import matplotlib.pyplot as plt

# Input values
x = np.linspace(-10, 10, 100)

# Activation Functions

# 1. Linear Function
def linear(x):
    return x

# 2. Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. Tanh Function
def tanh(x):
    return np.tanh(x)

# 4. ReLU Function
def relu(x):
    return np.maximum(0, x)

# 5. Softmax Function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Plotting Graphs

plt.figure(figsize=(12,8))

# Linear
plt.subplot(2,3,1)
plt.plot(x, linear(x))
plt.title("Linear Function")
plt.grid()

# Sigmoid
plt.subplot(2,3,2)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Function")
plt.grid()

# Tanh
plt.subplot(2,3,3)
plt.plot(x, tanh(x))
plt.title("Tanh Function")
plt.grid()

# ReLU
plt.subplot(2,3,4)
plt.plot(x, relu(x))
plt.title("ReLU Function")
plt.grid()

# Softmax
plt.subplot(2,3,5)
plt.plot(x, softmax(x))
plt.title("Softmax Function")
plt.grid()

plt.tight_layout()
plt.show()