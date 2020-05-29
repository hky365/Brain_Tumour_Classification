# Brain_Tumour_Classification
A Convolutional Neural Network, with the help of Tensorflow and Keras, was used for the classification of brain MRI images data available on Kaggle.

About the data:
The dataset consists of 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous.

Pre-Processing Steps Taken:
1) Data Augmentation and class balancing because the dataset was small and imbalanced (final outcome = 1085 yes and 980 no images)
2) Image Cropping and resizing to 299 x 299. The reason for doing that is to remove background noise and making the images similar in size.
3) Normalization of the images.
4) Data Split: 80% Train, 10% Validation, 10% Test

Transfer Learning and Model performance:
Transfer learning was used to improve the accuracy of the model. The pre-trained network used was a modified depth wise seperable Xception.

Steps Taken:
1) Downloaded Xception from Keras.
2) Trained the first 20 layers of Xception
3) Freezed the rest except the BatchNormalization layers
4) Trained the model with the best weights stored

Grad-CAM:
Grad-CAM was applied for model transparency.

Results:
Accuracy 98.02%, F1-score 0.98
