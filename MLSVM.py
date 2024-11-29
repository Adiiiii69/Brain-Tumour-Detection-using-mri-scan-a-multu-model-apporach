#!/usr/bin/env python
# coding: utf-8

# In[2]:

import subprocess 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}
import cv2
X = []
Y = []
for cls in classes:
    pth = 'brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)
xtrain = xtrain/255
xtest = xtest/255
from sklearn.decomposition import PCA
pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest

from sklearn.svm import SVC
sv = SVC()
sv.fit(xtrain, ytrain)
pred = sv.predict(xtest)
dec = {0:'Not Tumor', 1:'Tumor Detected'}

############################################################

#############################################################

# MLSVM.py
import cv2
import matplotlib.pyplot as plt
import sys

# Define function to predict and display image
def predict_and_display_image(image_path, model, class_labels):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to read the image.")
        return
    
    # Resize the image
    img_resized = cv2.resize(img, (200, 200))
    
    # Flatten and normalize the image
    img_flattened = img_resized.reshape(1, -1) / 255.0
    
    # Predict the class label
    predicted_label = model.predict(img_flattened)[0]
    
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(class_labels[predicted_label])
    aa = class_labels[predicted_label]
    print(aa)
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Access the image path argument
    image_path = sys.argv[2]  # The second argument, as the first one is the script name
    predict_and_display_image(image_path, sv, {0: 'No Tumor', 1: 'Positive Tumor'})




