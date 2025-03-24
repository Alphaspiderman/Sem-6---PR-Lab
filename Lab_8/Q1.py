import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
from keras.utils import to_categorical
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

if k.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(250, activation="sigmoid")(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adadelta(), loss=categorical_crossentropy, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=12, batch_size=500)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {score[0]}")
print(f"Accuracy: {score[1]}")

predictions = model.predict(x_test)
num_images = 10

plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(x_test[i].reshape(img_rows, img_cols), cmap="gray")
    plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.axis("off")

plt.tight_layout()
plt.show()
