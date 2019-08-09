'''
This file processes all of the images in the Pictures directory under the specific folder name given. Each folder in
the given picture directory should be the class name of the data feature. As the program iterates over each image, it
will save the corresponding label of the image as the specific folder's name. As each image is loaded, it is converted
to greyscale, resized to given dimensions through antialiasing, and flattened into a 1-D array. The y and X matrices
are then exported to a CSV file; this allows the data to essentially be compressed and easily loaded for further
processing.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import glob
import os
import numpy as np
import pickle
import pandas as pd

# The labels array is created and each folder name in the specific folder in Pictures is saved
folder_name = 'chest_xray_set'
labels = []

for root, dirs, files in os.walk("C:/Users/Lae/PycharmProjects/MLPneumoniaClassifier/Pictures/" + folder_name + "/",
    topdown=False):
        for name in dirs:
            labels.append(os.path.join(name))

print('Labels: ' + str(labels))

# The target (y) and feature (X) matrices are created along with the width and height dimensions
y = []
X = []
width = 250
height = 250

for filename in glob.glob("C:/Users/Lae/PycharmProjects/MLPneumoniaClassifier/Pictures/chest_xray_set/**/*.jpeg"):
    temp_target = ''
    for label in labels:
        if label in filename:
            temp_target = label
            y.append(temp_target)
            break

    im = Image.open(filename).convert('L') # imports and converts picture to greyscale
    im2 = im.resize((width, height), Image.ANTIALIAS) # changes dimensions to given width and height via antialiasing
    im = np.asarray(im2) # converts picture into a width x height matrix
    im = im.flatten() # flattens the matrix row by row into a 1-D array
    X.append(im) # appends processed image to feature matrix X

print('{} pictures imported with scaling of {} x {}'.format(len(X), width, height))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

X_train_matrix = 'X_train_matrix.pkl'
y_train_matrix = 'y_train_matrix.pkl'
X_test_matrix = 'X_test_matrix.pkl'
y_test_matrix = 'y_test_matrix.pkl'

pickle.dump(X_train, open(X_train_matrix, 'wb'))
pickle.dump(y_train, open(y_train_matrix, 'wb'))
pickle.dump(X_test, open(X_test_matrix, 'wb'))
pickle.dump(y_test, open(y_test_matrix, 'wb'))






