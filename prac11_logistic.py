import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test  / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),              # flatten 28x28 to 784
    Dense(1, activation='sigmoid')              # single neuron = logistic regression
])

model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# Note: MNIST has 10 classes so we convert to binary (0-4=0, 5-9=1)
y_train_binary = (y_train >= 5).astype(int)
y_test_binary  = (y_test  >= 5).astype(int)

model.fit(X_train, y_train_binary,batch_size=64,epochs=5,verbose=1)
test_loss, test_accuracy = model.evaluate(X_test, y_test_binary)
print("Test Loss:    ", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")