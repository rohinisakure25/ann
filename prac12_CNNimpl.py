import tensorflow as tf
from tensorflow.keras import layers, models
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values 0-255 to 0-1
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape to add channel dimension (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # convolution layer
    layers.MaxPooling2D(2,2),                                             # pooling layer
    layers.Flatten(),                                                      # flatten to 1D
    layers.Dense(64, activation='relu'),                                   # fully connected
    layers.Dense(10, activation='softmax')                                 # output 10 classes
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", round(test_acc * 100, 2), "%")