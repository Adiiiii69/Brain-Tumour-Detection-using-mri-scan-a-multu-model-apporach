

import streamlit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder 
import tempfile
import shutil
import sys
from subprocess import call
file_path = sys.argv[2]
encoder = OneHotEncoder()
encoder.fit([[0], [1]]) 
data = []
paths = []
result = []

for r, d, f in os.walk(r'C:/Users/LENOVO/OneDrive/Desktop/Brain_T/tumour_dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
        
paths = []
for r, d, f in os.walk(r"C:/Users/LENOVO/OneDrive/Desktop/Brain_T/tumour_dataset/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())
        
data = np.array(data)
data.shape
result = np.array(result)
result = result.reshape(139,2)
x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)
model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')

# Save the original stdout and stderr
original_stdout = sys.stdout


# Redirect stdout and stderr to os.devnull
sys.stdout = open(os.devnull, 'w')


# Call model's summary() method to hide the summary output
print(model.summary())

# Restore the original stdout and stderr
sys.stdout = original_stdout


history = model.fit(x_train, y_train, epochs = 50, batch_size = 40, verbose = 0,validation_data = (x_test, y_test))
from PIL import Image
import numpy as np
import cv2  # Import OpenCV for image processing
import sys

import numpy as np
import matplotlib.pyplot as plt

def predict(image_path, model, class_labels):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to read the image.")
        return
    
    # Convert grayscale image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize the image
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Convert the image to a numpy array
    x = np.array(img_resized)
    
    # Reshape the array
    x = x.reshape(1, 128, 128, 3)
    
    # Predict the class label
    res = model.predict_on_batch(x)
    
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(class_labels[res.argmax()])
    aa = class_labels[res.argmax()]
    print(aa)
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Access the image path argument
    image_path = sys.argv[2]  # The second argument, as the first one is the script name
    predict(file_path, model, {1: 'No Tumor', 0: 'Positive Tumor'})






