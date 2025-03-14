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

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(x_train_kmeans)
kmeans_labels = kmeans.predict(x_test_kmeans)


def map_clusters_to_labels(kmeans_labels, y_true):
    label_map = {}
    for i in range(num_clusters):
        cluster_mask = kmeans_labels == i
        true_labels = y_true[cluster_mask]
        if len(true_labels) > 0:
            most_common_label = np.bincount(true_labels).argmax()
            label_map[i] = most_common_label
    return np.array([label_map[label] for label in kmeans_labels])


mapped_labels = map_clusters_to_labels(kmeans_labels, y_test)
kmeans_accuracy = accuracy_score(y_test, mapped_labels)
print(f"K-Means Accuracy: {kmeans_accuracy:.4f}")

model = keras.models.Sequential(
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
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_train_cnn, y_train, epochs=5, validation_data=(x_test_cnn, y_test))
cnn_loss, cnn_accuracy = model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"CNN Accuracy: {cnn_accuracy:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    sample_idx = np.where(kmeans_labels == i)[0][0]
    ax.imshow(x_test[sample_idx], cmap="gray")
    ax.set_title(f"Cluster {i}")
    ax.axis("off")
plt.show()
