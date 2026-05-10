import tensorflow as tf

# Step 1 - Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 2 - Normalize pixel values 0-255 to 0-1
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Step 3 - Flatten 28x28 images to 784 values
X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

# Step 4 - Build model
# Dense(10) with softmax = Logistic Regression for 10 classes
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # hidden layer = Neural Network
    tf.keras.layers.Dense(10, activation='softmax')                      # output = Logistic Regression
])

# Step 5 - Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#training
model.fit(X_train, y_train, epochs=5)
# Step 7 - Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")