import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state = 42)

lr_classifier = LogisticRegression(max_iter = 200)
lr_classifier.fit(X_train, Y_train)

lr_predictions = lr_classifier.predict(X_test)

print("Logistic Regression Accuracy: ", accuracy_score(Y_test, lr_predictions))
print("Logistic Regression Classification: ", classification_report(Y_test, lr_predictions))

df = pd.DataFrame({'Real Values': Y_test, 'Predicted Values': lr_predictions})  
print(df) 