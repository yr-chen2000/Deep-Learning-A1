#!/usr/bin/env python
# coding: utf-8

# In[16]:


from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


# In[17]:


#Load Data
images = np.load('data/images.npy')
labels = np.load('data/labels.npy')

#Shuffle
indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# input image dimensions
img_rows, img_cols = images.shape[1], images.shape[2]

if K.image_data_format() == 'channels_first':
    images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images = images.reshape(images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

images = images.astype('float32')
images /= 255

#80/10/10% splits for training/validation and test sets
train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.2)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5)


# In[18]:


#“common sense” accuracy
def common_sense_accuracy(predicted, actual):

    pred_minutes = predicted[0]*60+predicted[1]
    actual_minutes = actual[0]*60+actual[1]

    abs_diff = abs(pred_minutes-actual_minutes)
    
    return min(abs_diff, 12*60-abs_diff)


# In[19]:


def re_category(labels):

    transformed_labels = labels[:][0] + labels[:][0] / 60
    return transformed_labels

train_labels2 = np.array([re_category(h) for h in train_labels])
val_labels2= np.array([re_category(h) for h in val_labels])
test_labels2 = np.array([re_category(h) for h in test_labels])


# In[ ]:


batch_size = 32
epochs = 30
# Build Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error',  
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['mae'])

# Fit Model
Learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
history = model.fit(train_images, train_labels2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_images, val_labels2),
          callbacks=[Learning_rate])

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
# Evaluate Model
score = model.evaluate(val_images, val_labels2, verbose=0)
print('Validation loss:', score[0])
print('Validation MAE:', score[1])
score_final = model.evaluate(test_images, test_labels2, verbose=0)
print('Test loss:', score_final[0])
print('Test MAE:', score_final[1])


# In[24]:


df.plot(figsize=(8, 5))
plt.grid(True)
plt.show()


# In[21]:


def calculate_time_from_predictions(predictions):
    
    hours = np.floor(predictions).astype(int)
    
    minutes = ((predictions - hours) * 60).astype(int)
    return hours, minutes


# In[22]:


predictions = model.predict(test_images)
predicted_hours, predicted_minutes = calculate_time_from_predictions(predictions)
predicted_hours = predicted_hours.flatten()
predicted_minutes = predicted_minutes.flatten()
common_sense_errors = [common_sense_accuracy([predicted_hours[i], predicted_minutes[i]], test_labels[i]) 
                       for i in range(len(test_labels))]
print('Average common sense error:', np.mean(common_sense_errors))


# In[23]:


# Plot a sample of clock images
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f'Predict: {predicted_hours[i]}:{predicted_minutes[i]:02d},Label: {test_labels[i][0]}:{test_labels[i][1]:02d}')
    plt.axis('off')
plt.show()

