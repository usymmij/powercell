import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

input_shape = (80,80,1)

BATCHES=1
EPOCHS=10

labls = np.load('data/label.npy', allow_pickle=True) 
imgs = np.load('data/data.npy',allow_pickle=True)
print('imported data')

imgs = imgs[0:int(float(len(labls)) / 4)]
labls = labls[0:int(float(len(labls)) / 4)]

im = []
for img in imgs:
    im.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
im = np.asarray(im)
print("data length:" + str(len(labls)))
im = im.reshape((-1, 80,80,1))
lb = labls
lb = lb.reshape(-1,3)
print(labls.shape)
print(im.shape)

model = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=(10,10), strides=1,
                        padding= 'same', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides= (1,1),
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, kernel_size=(5,5), strides= 2,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides= (1,1),
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, kernel_size=(2,2), strides= 4,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.Conv2D(64, kernel_size=(2,2), strides= 4,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.Conv2D(32, kernel_size=(2,2), strides= 4,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides= (4,4),
                              padding= 'same', data_format= None),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(3, activation="relu"),
])

model.summary()

print('compiling model')

model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.01),
    loss="mae",
    metrics=["mae"])

model.fit(im, lb, BATCHES, EPOCHS, shuffle=True)
model.save("model.h5")