# MLP on CIFAR-10 using Keras Sequential (Flatten + Dense)
# -------------------------------------------------------
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. One-hot encode labels
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4. Build Sequential MLP
model = Sequential(
    [
        Flatten(input_shape=(32, 32, 3)),  # flatten color images -> 3072 vector
        Dense(1024, activation="relu"),  # hidden layer 1
        Dropout(0.4),
        Dense(512, activation="relu"),  # hidden layer 2
        Dropout(0.3),
        Dense(256, activation="relu"),  # hidden layer 3 (optional)
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),  # output
    ]
)

model.compile(
    optimizer="adam",  # try 'sgd' if you prefer
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint("mlp_cifar10_best.h5", save_best_only=True, monitor="val_loss"),
]

# 6. Train
history = model.fit(
    x_train,
    y_train_cat,
    validation_split=0.1,  # 10% of train as validation
    epochs=30,
    batch_size=128,
    callbacks=callbacks,
    verbose=2,
)

# 7. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# 8. Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Single example prediction demo
idx = np.random.randint(0, x_test.shape[0])
sample = x_test[idx]
pred = model.predict(sample.reshape(1, 32, 32, 3))
pred_label = np.argmax(pred)
true_label = int(y_test[idx])
print(f"True label: {true_label}, Predicted: {pred_label}")

plt.imshow(sample)
plt.title(f"True: {true_label}  Pred: {pred_label}")
plt.axis("off")
plt.show()
