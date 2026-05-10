import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Define model
model = Sequential([
    Flatten(input_shape=(28, 28)),        # flatten 28x28 to 784
    Dense(128, activation='relu'),         # hidden layer
    Dense(10, activation='softmax')        # output 10 classes
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")