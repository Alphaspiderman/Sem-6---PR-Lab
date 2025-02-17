import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from sklearn.datasets import load_wine

wine = load_wine()
x = wine.data[:, :2]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

fcm = FCM(n_clusters=3, m=2.0, max_iter=150, error=1e-5)
fcm.fit(x_scaled)

centres = fcm.centers
labels = fcm.predict(x_scaled)

plt.figure(figsize=(8, 6))

for i in range(3):
    plt.scatter(
        x_scaled[labels == i, 0], x_scaled[labels == i, 1], label=f"Cluster {i + 1}"
    )

plt.scatter(
    centres[:, 0], centres[:, 1], marker="x", color="black", s=200, label="Centres"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Fuzzy c-means clustering on wine dataset")
plt.show()
