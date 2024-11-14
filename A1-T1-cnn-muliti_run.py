#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
#create the validation set
X_valid, X_train = X_train_full[:6000] / 255.0, X_train_full[6000:] / 255.0
y_valid, y_train = y_train_full[:6000], y_train_full[6000:]
#add label for class
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[4]:


kinits=['he_normal','he_uniform','glorot_normal','glorot_uniform']
#lrs=[0.05,0.01,0.005,0.001,0.0005,0.0001]
drs=[0.1,0.2,0.3,0.4,0.5]


# In[5]:


for kinit in kinits:
#for kinit in ['glorot_normal']:
    #for lr in lrs:
    for lr in [0.001]:
        #for dr in drs:
        for dr in [0.1]:
            #
            #a convolutional neural network described in Ch. 14, p. 447.
            model = keras.models.Sequential([
                    keras.layers.Conv2D(64, 7, activation="relu",kernel_initializer=kinit, padding="same",
                                        input_shape=[28, 28, 1]),
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
            df=pd.DataFrame(history.history)
            df.to_csv(f'cnn/{kinit}_relu_Adam_{lr}_{dr}.csv')
            df.plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)


# In[6]:


import os


files=os.listdir('cnn/')
saving=[]
for filename in files:
    if '.csv' in filename:
        data=pd.read_csv(f'cnn/{filename}')[-1:]
        kinit=filename.split('_')[0]+'_'+filename.split('_')[1]
        act=filename.split('_')[2]
        opt=filename.split('_')[3]
        reg='drop'
        drop=float(filename.split('_')[5][:-4])
        lr=float(filename.split('_')[4])
        acc=np.array(data['accuracy'])[0]
        loss=np.array(data['loss'])[0]
        val_acc=np.array(data['val_accuracy'])[0]
        val_loss=np.array(data['val_loss'])[0]
        read_out=[kinit,act,opt,reg,drop,lr,acc,loss,val_acc,val_loss]
        saving.append(read_out)
        


# In[7]:


df=pd.DataFrame(saving,
                columns=['kernal_init','activation','optimizer','regularization','regularization_value','learning_rate',
                         'accuracy','loss','val_accuracy','val_loss'])


# In[8]:


df.nlargest(10, 'accuracy')


# In[9]:


df.nlargest(10, 'val_accuracy')


# In[ ]:




