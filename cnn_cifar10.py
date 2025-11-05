# Convolutional Neural Network (CNN) on CIFAR-10 using Keras Sequential Model
# --------------------------------------------------------------------------
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load CIFAR-10 dataset (60,000 color images, 32x32x3)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2️⃣ Normalize pixel values (0–255) to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3️⃣ Convert labels to one-hot encoding (10 classes)
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4️⃣ Build the CNN model
model = Sequential(
    [
        # Convolution block 1
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        # Convolution block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        # Convolution block 3
        Conv2D(128, (3, 3), activation="relu"),
        # Flatten & Dense layers
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

# 5️⃣ Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# 6️⃣ Define callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("cnn_cifar10_best.h5", save_best_only=True, monitor="val_loss"),
]

# 7️⃣ Train the model
history = model.fit(
    x_train,
    y_train_cat,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=2,
)

# 8️⃣ Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 9️⃣ Plot training curves
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

# 🔟 Single image prediction
idx = np.random.randint(0, len(x_test))
sample = x_test[idx]
pred = model.predict(sample.reshape(1, 32, 32, 3))
predicted_label = np.argmax(pred)
true_label = int(y_test[idx])

plt.imshow(sample)
plt.title(f"True: {true_label} | Predicted: {predicted_label}")
plt.axis("off")
plt.show()
