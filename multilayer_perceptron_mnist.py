# Multilayer Perceptron (MLP) on MNIST using Keras Sequential API
# ---------------------------------------------------------------
# Uses Flatten and Dense layers only

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1️⃣ Load the MNIST dataset (60,000 training images, 10,000 test images)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2️⃣ Normalize pixel values (0–255) to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3️⃣ Convert labels to one-hot encoding (for 10 output classes)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4️⃣ Build the MLP model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images → 784 vector
        Dense(512, activation="relu"),  # Hidden layer 1
        Dense(256, activation="relu"),  # Hidden layer 2
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),  # Output layer for 10 classes
    ]
)

# 5️⃣ Compile the model
model.compile(
    optimizer="adam",  # or try 'sgd' for comparison
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 6️⃣ Train the model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,  # 10% for validation
    epochs=10,
    batch_size=128,
    verbose=2,
)

# 7️⃣ Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 8️⃣ Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# 9️⃣ Make a single prediction example
import numpy as np

index = np.random.randint(0, len(x_test))
sample = x_test[index]
pred = model.predict(sample.reshape(1, 28, 28))
predicted_label = pred.argmax()

plt.imshow(sample, cmap="gray")
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()
