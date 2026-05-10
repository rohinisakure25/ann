import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D  
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape to add channel dimension (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
models = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # convolution layer
    MaxPooling2D(2,2),                                             # pooling layer
    Flatten(),                                                      # flatten to 1D
    Dense(64, activation='relu'),                                   # fully connected
    Dense(10, activation='softmax')                                 # output 10 classes
])

# Compile model
models.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
models.fit(X_train, y_train, epochs=5)
test_loss, test_acc = models.evaluate(X_test, y_test)
print("Test Accuracy:", round(test_acc * 100, 2), "%")