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


# In[2]:


print(tf.config.list_physical_devices())


# In[3]:


#variable
acts=["relu","linear"]
kinits=['he_normal','he_uniform','glorot_normal','glorot_uniform']
lrs=[0.5,0.1,1.0]


# In[4]:


#A multi-layer perceptron described in detail in Ch. 10, pp. 299-307
d1=300
d2=100
for act in acts:
    for kinit in kinits:
        for lr in lrs:
            print(act,kinit,lr)
            model = keras.models.Sequential()
            model.add(keras.layers.Flatten(input_shape=[28, 28]))# convert each input image into a 1D array
            model.add(keras.layers.Dense(d1, activation=act,kernel_initializer=kinit))# a Dense hidden layer with 300 neurons.
            #model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(d2, activation=act,kernel_initializer=kinit))# a Dense hidden layer with 100 neurons.
            #model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(10, activation="softmax"))#a Dense output layer with 10 neurons (one per class)

            opt=keras.optimizers.Adam(learning_rate=lr)
            model.compile(loss="sparse_categorical_crossentropy",
                            optimizer=opt,
                            metrics=["accuracy"])

            history = model.fit(X_train, y_train, epochs=30,
                                validation_data=(X_valid, y_valid))
            df=pd.DataFrame(history.history)
            df.to_csv(f'mlp/{d1}_{d2}_{kinit}_{act}_Adam_{lr}.csv')
            df.plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)


# In[8]:


import os


files=os.listdir('mlp/')
saving=[]
for filename in files:
    if '.csv' in filename:
        data=pd.read_csv(f'mlp/{filename}')[-1:]
        d1=float(filename.split('_')[0])
        d2=float(filename.split('_')[1])
        kinit=filename.split('_')[2]+'_'+filename.split('_')[3]
        act=filename.split('_')[4]
        opt=filename.split('_')[5]
        if 'd0.5' in opt:
            opt=opt[:-4]
            reg='d0.5'
        elif 'l10.01' in opt:
            opt=opt[:-6]
            reg='l10.01'
        elif 'l20.01' in opt:
            opt=opt[:-6]
            reg='l20.01'
        else:
            reg='None'
        lr=float(filename.split('_')[6][:-4])
        acc=np.array(data['accuracy'])[0]
        loss=np.array(data['loss'])[0]
        val_acc=np.array(data['val_accuracy'])[0]
        val_loss=np.array(data['val_loss'])[0]
        read_out=[d1,d2,kinit,act,opt,reg,lr,acc,loss,val_acc,val_loss]
        saving.append(read_out)
        


# In[9]:


df=pd.DataFrame(saving,
                columns=['d1','d2','kernal_init','activation','optimizer','regularization','learning_rate',
                         'accuracy','loss','val_accuracy','val_loss'])


# In[11]:


df.nlargest(10, 'accuracy')


# In[12]:


df.nlargest(10, 'val_accuracy')


# In[89]:


#learning_rate 
subset0=df[(df['kernal_init']=='glorot_uniform')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='SGD')]
plt.scatter(subset0['learning_rate'],subset0['accuracy'],label='accuracy')
plt.scatter(subset0['learning_rate'],subset0['val_accuracy'],label='val_accuracy')
plt.scatter(subset0['learning_rate'],subset0['loss'],label='accuracy')
plt.scatter(subset0['learning_rate'],subset0['val_loss'],label='val_accuracy')
plt.grid()
plt.legend()
plt.ylim(0,1)
plt.xscale('log')
plt.xlabel('learning rate')


# In[13]:


#optimizer influence
subset1=df[(df['kernal_init']=='glorot_uniform')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='Adam')]
plt.scatter(subset1['learning_rate'],subset1['accuracy'],label='accuracy')
plt.scatter(subset1['learning_rate'],subset1['val_accuracy'],label='val_accuracy')
plt.scatter(subset1['learning_rate'],subset1['loss'],label='accuracy')
plt.scatter(subset1['learning_rate'],subset1['val_loss'],label='val_accuracy')
plt.grid()
plt.legend()
plt.ylim(0,1)
plt.xscale('log')
plt.xlabel('learning rate')


# In[16]:


#learning_rate 
subset0=df[(df['kernal_init']=='glorot_uniform')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='SGD')]
subset2=df[(df['kernal_init']=='glorot_normal')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='SGD')]
subset3=df[(df['kernal_init']=='he_normal')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='SGD')]
subset4=df[(df['kernal_init']=='he_uniform')
           *(df['regularization']=='None')
           *(df['activation']=='relu')
           *(df['optimizer']=='SGD')]
plt.scatter(subset2['learning_rate'],subset2['accuracy'],label='glorot_normal')
plt.scatter(subset0['learning_rate'],subset0['accuracy'],label='glorot_uniform')
plt.scatter(subset3['learning_rate'],subset2['accuracy'],label='he_normal')
plt.scatter(subset4['learning_rate'],subset2['accuracy'],label='he_uniform')

plt.grid()
plt.legend()
plt.ylim(0,1)
plt.xscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')


# In[ ]:


cifar10=keras.datasets.cifar10.load_data()

