import numpy as np
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
Y = np.array([[0],[1],[1],[0]])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Random weight initialization
np.random.seed(1)

# Input layer -> Hidden layer weights
weights_input_hidden = np.random.uniform(size=(2, 2))

# Hidden layer -> Output layer weights
weights_hidden_output = np.random.uniform(size=(2, 1))
learning_rate = 0.5
epochs = 50000

for epoch in range(epochs):
    # ----- Forward Propagation -----
    # Hidden layer
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    # Output layer
    final_input = np.dot(
        hidden_output,
        weights_hidden_output
    )

    predicted_output = sigmoid(final_input)
    error = Y - predicted_output
    # ----- Back Propagation -----

    d_output = error * sigmoid_derivative(
        predicted_output
    )

    hidden_error = d_output.dot(
        weights_hidden_output.T
    )

    d_hidden = hidden_error * sigmoid_derivative(
        hidden_output
    )

    # ----- Update Weights -----

    weights_hidden_output += hidden_output.T.dot(
        d_output
    ) * learning_rate

    weights_input_hidden += X.T.dot(
        d_hidden
    ) * learning_rate

print("Predicted Output:",predicted_output)
