import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 42)

svm_classifier = SVC(kernel = 'linear')
svm_classifier.fit(X_train, Y_train)

svm_predictions = svm_classifier.predict(X_test)

print("Support Vector Machine Accuracy: ", accuracy_score(Y_test, svm_predictions))
print("Support Vector Machine Classification: ", classification_report(Y_test, svm_predictions))

df = pd.DataFrame({'Real Values': Y_test, 'Predicted Values': svm_predictions})  
print(df) 