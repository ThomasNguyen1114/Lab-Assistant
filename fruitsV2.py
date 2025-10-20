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
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import imageio

train_directory = "train/train"
test_directory = "test"
image_size = (64, 64)
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
        image_resized = resize(image, image_size, anti_aliasing=True)
        X.append(image_resized.flatten())
        Y.append(fruit_type)
print("Done")

X = np.array(X)
Y = np.array(Y)

# Use stratified split for balanced classes
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"\nDataset info:")
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature dimensions: {X.shape[1]}")
print(f"\nClass distribution in training:")
print(pd.Series(Y_train).value_counts())

# Create pipeline with PCA for dimensionality reduction
print("\nTraining model with PCA + SVM...")
pipeline = Pipeline([
    ('pca', PCA(n_components=100)),  # Reduce to 100 components
    ('svm', svm.SVC(kernel='linear'))
])

# Train the model
pipeline.fit(X_train, Y_train)

# Test set predictions
fruit_predictions = pipeline.predict(X_test)
test_accuracy = accuracy_score(Y_test, fruit_predictions) * 100
print(f"\nTest Set Accuracy: {test_accuracy:.2f}%")

# Cross-validation (this will be much faster now)
print("\nPerforming 5-fold cross-validation...")
scores = cross_val_score(pipeline, X_train, Y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(Y_test, fruit_predictions))
