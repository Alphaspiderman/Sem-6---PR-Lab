import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_cnn = np.expand_dims(x_train, axis=-1)
x_test_cnn = np.expand_dims(x_test, axis=-1)
x_train_kmeans = x_train.reshape(x_train.shape[0], -1)
x_test_kmeans = x_test.reshape(x_test.shape[0], -1)

km = KMeans(n_clusters=10, random_state=42, n_init=10)
km.fit(x_train_kmeans)
km_pred = km.predict(x_test_kmeans)


label_map = np.array(
    [np.bincount(y_train[km.labels_ == i]).argmax() for i in range(10)]
)[km.predict(x_test_kmeans)]

print(f"K-Means Accuracy: {accuracy_score(y_test, label_map):.4f}")

cnn = keras.models.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
cnn.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
cnn.fit(x_train_cnn, y_train, epochs=5, validation_data=(x_test_cnn, y_test))
cnn_loss, cnn_accuracy = cnn.evaluate(x_test_cnn, y_test, verbose=0)
print(f"CNN Accuracy: {cnn_accuracy:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    sample_idx = np.where(km_pred == i)[0][0]
    ax.imshow(x_test[sample_idx], cmap="gray")
    ax.set_title(f"Cluster {i}")
    ax.axis("off")
plt.show()
