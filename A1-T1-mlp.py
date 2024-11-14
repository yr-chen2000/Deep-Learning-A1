#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
cifar10=keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
#create the validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
#add label for class
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
]


# In[2]:


#A multi-layer perceptron described in detail in Ch. 10, pp. 299-307
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))# convert each input image into a 1D array
model.add(keras.layers.Dense(300, activation="relu",kernel_initializer='he_normal'))# a Dense hidden layer with 300 neurons.
model.add(keras.layers.Dense(100, activation="relu",kernel_initializer='he_normal'))# a Dense hidden layer with 100 neurons.
model.add(keras.layers.Dense(10, activation="softmax"))#a Dense output layer with 10 neurons (one per class)

opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
y_proba.round(2)


# In[3]:


y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(np.array(class_names)[y_new])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.


# In[4]:


#A multi-layer perceptron described in detail in Ch. 10, pp. 299-307
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))# convert each input image into a 1D array
model.add(keras.layers.Dense(300, activation="relu",kernel_initializer='glorot_normal'))# a Dense hidden layer with 300 neurons.
model.add(keras.layers.Dense(100, activation="relu",kernel_initializer='glorot_normal'))# a Dense hidden layer with 100 neurons.
model.add(keras.layers.Dense(10, activation="softmax"))#a Dense output layer with 10 neurons (one per class)

opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
y_proba.round(2)


# In[5]:


y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(np.array(class_names)[y_new])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.


# In[6]:


#A multi-layer perceptron described in detail in Ch. 10, pp. 299-307
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))# convert each input image into a 1D array
model.add(keras.layers.Dense(300, activation="relu",kernel_initializer='he_uniform'))# a Dense hidden layer with 300 neurons.
model.add(keras.layers.Dense(100, activation="relu",kernel_initializer='he_uniform'))# a Dense hidden layer with 100 neurons.
model.add(keras.layers.Dense(10, activation="softmax"))#a Dense output layer with 10 neurons (one per class)

opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)# get prediction probabilities
y_proba.round(2)


# In[7]:


y_pred = np.argmax(y_proba, axis=-1) # get predicted class index by finding the index of maximum probability along the last axis
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(np.array(class_names)[y_new])
#The predict_classes method was deprecated in TensorFlow 2.6 and has been removed in later versions.
#Instead, you should use model.predict followed by np.argmax to get the predicted class indices.


# In[11]:


plt.imshow(X_test[2])
print((np.array(class_names)[y_test[2]]))


# In[ ]:




