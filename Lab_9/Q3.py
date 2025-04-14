import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw.images
y_names = lfw.target_names[lfw.target]

def guess_gender(name):
    male_names = ['George', 'Tony', 'Donald', 'Gerhard', 'Colin', 'Jean', 'Ariel']
    return 1 if any(male in name for male in male_names) else 0

y = np.array([guess_gender(name) for name in y_names])

X_flat = X.reshape(X.shape[0], -1) / 255.0

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# ANN model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype("int").flatten()

label_map = {0: 'Female', 1: 'Male'}

plt.figure(figsize=(10, 5))
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx].reshape(X.shape[1], X.shape[2])
    true_label = label_map[y_test[idx]]
    pred_label = label_map[y_pred[idx]]
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"T: {true_label}\nP: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()