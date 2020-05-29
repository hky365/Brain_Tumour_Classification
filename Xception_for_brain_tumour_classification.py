# -*- coding: utf-8 -*-
"""Xception_for_Brain_Tumour_Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10AZhSCIlXowarLuq3K2dgwOC3m5oIn8b
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

# Commented out IPython magic to ensure Python compatibility.
#importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_curve
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Activation, GlobalAveragePooling2D, ZeroPadding2D,MaxPooling2D, Conv2D 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import keras
import sys
import warnings
from os import listdir
import cv2
import glob
import os
import imutils
import tensorflow.keras
import tensorflow.keras.layers
# %matplotlib inline
#Surpressing all deprecation and future warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#Sets the global random tf seed
#tf.random.set_random_seed(42)

#mounting the drive to this notebook
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#DONT ---loading the dataset
#!unzip -uq "/content/drive/My Drive/data/for_data_aug" -d "/content/drive/My Drive/data/for_data_aug"

image_size = [299, 299]

def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '/' + filename)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y

data_path = '/../Out_crop_brain/'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = data_path + 'yes' 
augmented_no = data_path + 'no'

IMG_WIDTH, IMG_HEIGHT = (299, 299)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

## checking the shapes of the data
X.shape
y.shape

def split_data(X, y, test_size=0.2):
       
    """
    Splits data into training, development and test sets.
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    Returns:
        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
        y_train: A numpy array with shape = (#_train_examples, 1)
        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
        y_val: A numpy array with shape = (#_val_examples, 1)
        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
        y_test: A numpy array with shape = (#_test_examples, 1)
    """
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.2)

# Downloading Xception
xception = Xception(input_shape = image_size + [3], weights= 'imagenet', include_top= False)

train_yes_path = '/../train/train_yes/'
train_no_path = '/../train/train_no/'
test_yes_path = '/../test/test_yes/'
test_no_path ='/../test/test_no/'
val_yes_path = '/../val/val_yes/'
val_no_path = '/../val/val_no/'

#test accuracy ######

## labeling the data ##
y_test_files = '/../for_data_aug/test/' 
y_test_files_yes = y_test_files + 'test_yes'
y_test_files_no = y_test_files + 'test_no'
y_test_labels = load_data([y_test_files_yes, y_test_files_no])

#computing f1 score
changes = []

for i in y_test_prob_1:
  if i[0] > 0.5:
    changes.append([0])
  else:
    changes.append([1])

#Helper Function
def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score

#### new model #####

xception_1 = Xception(input_shape = image_size + [3], weights= 'imagenet', include_top= False)

#The best model configuration
x = xception_1.output
x = GlobalAveragePooling2D()(x)
#x = Dense(1024)(x)
x = Dense(512)(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Dropout(0.5)(x)
x = Dense(128, kernel_regularizer= tf.keras.regularizers.l2(l=0.01))(x)
x = Dropout(0.5)(x)
preds = Dense(1, activation = 'sigmoid', name = 'fc')(x)
model_1 = Model(inputs = xception_1.input, outputs = preds)

# I trained the whole model instead of freezing the layers.

for layer in xception_1.layers:
    layer.trainable = False

for layer in xception_1.layers[:20]:
    layer.trainable = True

for layer in model_1.layers:
    if 'bn' in layer.name:
        layer.trainable = True

# compile the new model using a adam optimizer
model_1.compile(optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

#defining checkpoint and the early stopping to prevent overfitting
checkpoint = ModelCheckpoint(filepath='bestmodel_1.hdf5', monitor='val_loss', 
                             save_weights_only=True,  mode='min', save_best_only=True, verbose=1)
Early_Stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

model_1.fit(x= X_train, y= y_train, batch_size= 32, epochs = 40, callbacks= [Early_Stopping, checkpoint], validation_data=[X_val, y_val] )

history = model_1.history.history

plot_metrics(history)

#Load the best weights
model_1.load_weights('bestmodel_1.hdf5')

best_model_1 = model_1

best_model_1.metrics_names

loss, acc = best_model_1.evaluate(x = X_test, y= y_test)

print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")

y_test_prob_2 = best_model_1.predict(X_test)

compute_f1_score(y_test, y_test_prob_2)

#Grad_Cam

best_model_1.summary()

#pre-processing the image
image_path = train_yes_path + "sample.jpg"

def prep_data(image_path):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    #y = []
    image_width, image_height = image_size
    
    #for directory in dir_list:
     #   for filename in listdir(directory):
            # load the image
    image = cv2.imread(image_path)
    # resize image
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    # convert image to numpy array and append it to X
    X.append(image)
                
    X = np.array(X)
    #X = np.expand_dims(X, axis=0)
    #y = np.array(y)
    
    # Shuffle the data
    #X, y = shuffle(X, y)
    
    #print(f'Number of examples is: {len(X)}')
    #print(f'X shape is: {X.shape}')
    #print(f'y shape is: {y.shape}')
    
    return X

image_5 = prep_data(image_path)

predict_image = best_model_1.predict(image_5)
target_class = np.argmax(predict_image[0])
#print("Target Class = %d"%target_class)

### GRAD-CAM ###
last_conv = best_model_1.get_layer('block14_sepconv2_act') #block14_sepconv2_act
grads = K.gradients(best_model_1.output[:,0],last_conv.output)[0]# 1 because the target class is 1

pooled_grads = K.mean(grads,axis=(0,1,2))
iterate = K.function([best_model_1.input],[pooled_grads,last_conv.output[0]])
pooled_grads_value,conv_layer_output = iterate([image_5])

for i in range(512):
    conv_layer_output[:,:,i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output,axis=-1)

for x in range(heatmap.shape[0]):
    for y in range(heatmap.shape[1]):
        heatmap[x,y] = np.max(heatmap[x,y],0)

heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
plt.imshow(heatmap)

## resizing ###

image_resize = cv2.imread(image_path)
image_resize = cv2.resize(image_resize, (299, 299))

upsample = cv2.resize(heatmap, (299,299))
#plt.imshow(cv2.imread(image_path))
plt.imshow(image_resize)
plt.imshow(upsample,alpha=0.5)
plt.show()

###### ------ the end ------ ####