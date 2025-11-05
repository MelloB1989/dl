import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 3 * x + 2 + np.random.randn(100, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential([Dense(1, input_dim=1, activation="linear")])

model.compile(optimizer="sgd", loss="mse", metrics=["mae"])

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=10,
    validation_data=(x_test, y_test),
    verbose=1,
)

loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest MSE: {loss:.4f}, Tesr MAE: {mae:.4f}")

weights, bias = model.layers[0].get_weights()
print(f"Learned weight (slope): {weights[0][0]:.4f}")
print(f"Learned bias (intercept): {bias[0]:.4f}")

y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, color="blue", label="Actual data")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="Model prediction")
plt.title("Linear Regression using Keras Sequential Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
