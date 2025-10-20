import numpy as np                       # Library for numerical operations and arrays
import matplotlib.pyplot as plt          # Library for plotting and visualization
import pandas as pd                      # Library for data handling and analysis

df = pd.read_csv("iris.csv")             # Load the Iris dataset from a CSV file into a pandas DataFrame

X = df.iloc[:, :4].values                # Select all rows and the first 4 columns (features)
Y = df['species'].values                 # Select the 'species' column as the target labels

from sklearn.model_selection import train_test_split  # Import function to split data into training and test sets
# Split data: 98% for training, 2% for testing (randomly)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .02, random_state = 42)  

from sklearn.preprocessing import StandardScaler      # Import tool for scaling/normalizing feature values - makes bigger numbers smaller
ss = StandardScaler()                                 # Create a StandardScaler object
X_train = ss.fit_transform(X_train)                   # Fits scaler to training data (learns mean & std) and scales X_train
X_test = ss.transform(X_test)                         # Scale test data using the same scaling parameters

from sklearn.naive_bayes import GaussianNB            # Import the Gaussian Naive Bayes model
classifier = GaussianNB()                             # Create a GaussianNB classifier object
classifier.fit(X_train, Y_train)                      # Predicts class labels for new/unseen data using the trained model

Y_prediction = classifier.predict(X_test)             # Predict the class labels for the test set using the trained model

from sklearn.metrics import confusion_matrix          # Compares true vs predicted labels, returns a confusion matrix
cm = confusion_matrix(Y_test, Y_prediction)           # Compute confusion matrix to evaluate performance

from sklearn.metrics import accuracy_score            # Import accuracy function
print("Accuracy: ", accuracy_score(Y_test, Y_prediction))  # P# Calculates the fraction of correct predictions (accuracy)
print(cm)                                             # Print the confusion matrix

# Create a DataFrame comparing actual vs predicted values
df = pd.DataFrame({'Real Values': Y_test, 'Predicted Values': Y_prediction})  
print(df)                                             # Display the comparison table

