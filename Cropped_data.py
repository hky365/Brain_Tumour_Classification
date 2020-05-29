#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation

# **About the data:** <br>
# The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous. You can find [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

# Since this is a small dataset, I used data augmentation in order to create more images.

# Also, we could solve the data imbalance issue (since 61% of the data belongs to the tumorous class) using data augmentation.

# ## Import Necessary Modules

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import imutils
import matplotlib.pyplot as plt
import os
from os import listdir
import glob
import PIL.Image
from PIL import Image
import time    

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pip install pillow


# In[4]:



yes_path = '../yes/'
no_path = '../no/'


# In[5]:


def augment_data(file_dir, n_generated_samples, save_to_dir):
    """
    Arguments:
        file_dir: A string representing the directory where images that we want to augment are found.
        n_generated_samples: A string representing the number of generated samples using the given image.
        save_to_dir: A string representing the directory in which the generated images will be saved.
    """
    
    #from keras.preprocessing.image import ImageDataGenerator
    #from os import listdir
    
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )

    
    for filename in listdir(file_dir):
        if filename == ".DS_Store":
            continue
        else:
            # load the image
            image = cv2.imread(file_dir + '/' + filename)
            # reshape the image
            image = image.reshape((1,)+image.shape)
            # prefix of the names for the generated sampels.
            save_prefix = 'aug_' + filename[:-4]
            # generate 'n_generated_samples' sample images
            i=0
            for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                        save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i > n_generated_samples:
                    break


# In[11]:


#Creating Paths
final_yes_path = r'/../yes' 
if not os.path.exists(final_yes_path):
    os.makedirs(final_yes_path)
final_no_path = r'/../no' 
if not os.path.exists(final_no_path):
    os.makedirs(final_no_path)


# In[118]:


#resizing the no_images and yes images 
JPG = "*.JPG"
images_JPG = glob.glob(no_path+JPG)
#aug_yes_path = '/Users/arturmalantowicz/Hamza/thesis/testing/aug_yes' #made above
#for JPG
folderLen = len(no_path)
for img in images_JPG:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_no_path+ "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)
    
#for jpg
corr = JPG.lower()
images_jpg = glob.glob(no_path+corr)
for img in images_jpg:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_no_path + "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)

#for png
png = "*.png"
images_png = glob.glob(no_path+png)
for img in images_png:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_no_path+ "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)

#for jpeg
jpeg = "*.jpeg"
images_png = glob.glob(no_path+jpeg)
for img in images_png:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_no_path + "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)    

## for yes images repeat the same procedure


# In[123]:


#resizing the yes_images
JPG = "*.JPG"
images_JPG = glob.glob(yes_path+JPG)
#aug_yes_path = #made above
#for JPG
folderLen = len(yes_path)
for img in images_JPG:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_yes_path+ "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)
    
#for jpg
corr = JPG.lower()
images_jpg = glob.glob(yes_path+corr)
for img in images_jpg:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_yes_path + "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)

#for png
png = "*.png"
images_png = glob.glob(yes_path+png)
for img in images_png:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_yes_path+ "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)

#for jpeg
jpeg = "*.jpeg"
images_png = glob.glob(yes_path+jpeg)
for img in images_png:
    image= cv2.imread(img)
    imgresized = cv2.resize(image, (299,299))
    cv2.imwrite(final_yes_path + "/"+ img[folderLen:], imgresized)
    cv2.imshow("image", imgresized)    


# In[162]:


augmented_data_path = '/Users/arturmalantowicz/Hamza/thesis/Thesis_data/aug_data/'
yes_path = '/Users/arturmalantowicz/Hamza/thesis/Thesis_data/yes/'
no_path = '/Users/arturmalantowicz/Hamza/thesis/Thesis_data/no/'

# augment data for the examples with label equal to 'yes' representing tumurous examples
augment_data(file_dir=final_yes_path, n_generated_samples=6, save_to_dir=augmented_data_path+'yes')
# augment data for the examples with label equal to 'no' representing non-tumurous examples
augment_data(file_dir=final_no_path, n_generated_samples=9, save_to_dir=augmented_data_path+'no')


# In[163]:


def data_summary(main_path):
    
    yes_path = main_path+'yes'
    no_path = main_path+'no'
        
    # number of files (images) that are in the the folder named 'yes' that represent tumorous (positive) examples
    m_pos = len(listdir(yes_path))
    # number of files (images) that are in the the folder named 'no' that represent non-tumorous (negative) examples
    m_neg = len(listdir(no_path))
    # number of all examples
    m = (m_pos+m_neg)
    
    pos_prec = (m_pos* 100.0)/ m
    neg_prec = (m_neg* 100.0)/ m
    
    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {m_pos}") 
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {m_neg}") 


# In[165]:


#checking the shit
aug_path = r'../aug_data/' 
data_summary(aug_path)


# In[ ]:


# since all the data images have the same size now (299, 299, 3), we proceed to the next step.


# In[6]:


def crop_brain_contour(image, plot=False):
    
    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image


# In[ ]:


## cropping all the images to brain ##


# In[32]:


## Yes images cropping##
yes_path = r'../aug_data/yes/'
c_yes_path = r'../Out_crop_brain/yes'
folderLen = len(yes_path)
for img in glob.glob(yes_path+"*.jpg"):
    #print(img)
    image= cv2.imread(img)
    #print(image.shape)
    imgcrop = crop_brain_contour(image)
    cv2.imwrite(c_yes_path+ "/"+ img[folderLen:], imgcrop)
    cv2.imshow("image", imgcrop)


# In[33]:


## No images cropping##
no_path = r'../aug_data/no/'
c_no_path = r'../Out_crop_brain/no'
folderLen = len(no_path)
for img in glob.glob(no_path+"*.jpg"):
    #print(img)
    image= cv2.imread(img)
    #print(image.shape)
    imgcrop = crop_brain_contour(image)
    cv2.imwrite(c_no_path+ "/"+ img[folderLen:], imgcrop)
    cv2.imshow("image", imgcrop)


# In[ ]:


## the end ## the rest is now on google colab

