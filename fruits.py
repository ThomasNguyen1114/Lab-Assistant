import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import imageio

train_directory = "train/train"
test_directory = "test"

image_size = (64,64)

X = []
Y = []

print("Loading training images...")

for fruit_type in os.listdir(train_directory):
    fruit_path = os.path.join(train_directory, fruit_type)

    if not os.path.isdir(fruit_path):
        continue

    print("Found Folder: ", fruit_path)

    for image_file in os.listdir(fruit_path):
        image_path = os.path.join(fruit_path, image_file)
        image = imageio.imread(image_path)
        image_resized = resize(image, image_size, anti_aliasing = True)
        X.append(image_resized.flatten())
        Y.append(fruit_type)

    print("Done")

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 42, stratify = Y)

fruit_classifier = svm.SVC(kernel = 'linear')
fruit_classifier.fit(X_train, Y_train)

fruit_predictions = fruit_classifier.predict(X_test)

print("Fruits Support Vector Machine Accuracy: ", accuracy_score(Y_test, fruit_predictions) * 100)


from sklearn.model_selection import cross_val_score
scores = cross_val_score(fruit_classifier, X_train, Y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean() * 100)


