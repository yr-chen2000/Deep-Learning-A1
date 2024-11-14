#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
fashion_mnist = keras.datasets.fashion_mnist
cifar10=keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
#create the validation set
X_valid, X_train = X_train_full[:6000] / 255.0, X_train_full[6000:] / 255.0
y_valid, y_train = y_train_full[:6000], y_train_full[6000:]
#add label for class
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
]


# In[2]:


kinit='glorot_normal'
lr=0.001
dr=0.1
#a convolutional neural network described in Ch. 14, p. 447.
model = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, activation="relu",kernel_initializer=kinit, padding="same",
                            input_shape=[32, 32, 3]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(64, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(10, activation="softmax")
])
opt=keras.optimizers.Adam(learning_rate=lr)

model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# In[3]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()


# In[4]:


model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
print(np.array(class_names)[y_test[:3]])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.


# In[5]:


kinit='glorot_uniform'
lr=0.001
dr=0.1
#a convolutional neural network described in Ch. 14, p. 447.
model = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, activation="relu",kernel_initializer=kinit, padding="same",
                            input_shape=[32, 32, 3]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(64, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(10, activation="softmax")
])
opt=keras.optimizers.Adam(learning_rate=lr)

model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# In[6]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()


# In[7]:


model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
print(np.array(class_names)[y_test[:3]])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.


# In[8]:


kinit='he_uniform'
lr=0.001
dr=0.1
#a convolutional neural network described in Ch. 14, p. 447.
model = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, activation="relu",kernel_initializer=kinit, padding="same",
                            input_shape=[32, 32,3]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu",kernel_initializer=kinit, padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(64, activation="relu",kernel_initializer=kinit),
        keras.layers.Dropout(dr),
        keras.layers.Dense(10, activation="softmax")
])
opt=keras.optimizers.Adam(learning_rate=lr)

model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))


# In[9]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()


# In[10]:


model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
print(np.array(class_names)[y_test[:3]])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.

