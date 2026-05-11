import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)

        return self.y_hat

    def backward_propagation(self, X, y, y_hat):
        self.error = y - y_hat

        self.delta2 = self.error * self.sigmoid_derivative(y_hat)

        self.a1_error = self.delta2.dot(self.W2.T)

        self.delta1 = self.a1_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(self.delta2)

        self.b2 += np.sum(self.delta2, axis=0, keepdims=True)

        self.W1 += X.T.dot(self.delta1)

        self.b1 += np.sum(self.delta1, axis=0)

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):

            y_hat = self.forward_propagation(X)

            self.backward_propagation(X, y, y_hat)

            if i % 100 == 0:
                print("Error at epoch", i, ":", np.mean(np.abs(self.error)))


# Define the input and output datasets

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])


# Create a neural network with 2 input neurons,
# 4 neurons in the hidden layer,
# and 1 output neuron

nn = NeuralNetwork(2, 4, 1)


# Train the neural network

nn.train(X, y, epochs=10000, learning_rate=0.1)


# Use the trained neural network to make predictions

predictions = nn.forward_propagation(X)


# Print the predictions

print(predictions)