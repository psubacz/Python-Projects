# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:41:14 2018

@author: Eastwind
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:57:30 2018

@author: Eastwind
"""
import numpy as np
#import keras.layers.convolutional
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from random import shuffle
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

##Set Attributes
run = 0
height = 200
width = 150

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## load data
train_data = np.load('training_data_200x150_5.npy')
#train_data = np.load('Data//Final_Training_Data_200x150.npy')
print("Loaded training data")

##Shuffle Data
#shuffle(train_data)

## Split the data into test and training sets
train = train_data[:-1000]
test = train_data[-1000:]
print(train.shape)
print(test.shape)

##Extract training images and testing images and the one-hot array. the image
## needs to be reshaped due to it being a greyscale image. 
X_train = np.array([i[0] for i in train]).reshape(-1,width,height,1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1,width,height,1)
y_test = np.array([i[1] for i in test])

##Define a model to be created
def VGG_16(height,width):
    model = Sequential()
    #Block 1
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation="relu", name='block1_conv1',input_shape=(height,width,1)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation="relu", name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block1_pool'))

#    Block 2
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation="relu", name='block2_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation="relu", name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block2_pool'))

#    Block 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation="relu", name='block3_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation="relu", name='block3_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation="relu", name='block3_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation="relu", name='block3_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block3_pool'))

#    Block 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block4_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block4_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block4_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block4_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block4_pool'))

#    Block 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block5_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block5_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block5_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation="relu", name='block5_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block5_pool'))
#    Fully Connected layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc3'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc4'))
    model.add(Dense(3, activation='softmax'))
    return model

#Build the model
model = VGG_16(height,width)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.summary()
# Fit the model (train the model) 
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=16, verbose=1)

modelname = 'model_'+str(run)
model.save('model/'+modelname)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figs/Plot_training_and_validation_accuracy _values_'+run+'.png')
#
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('figs/Plot_training_and_validation_loss_values_'+run+'.png')