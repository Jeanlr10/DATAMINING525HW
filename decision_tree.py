#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris # To download the dataset
from sklearn.metrics import accuracy_score

# Load the dataset
X, y = load_iris(return_X_y=True)

print(y)

# Create a scatter plot for class 0 (setosa)
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Setosa', c='r')
# Create a scatter plot for class 1 (versicolor)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Versicolor', c='g')
# Create a scatter plot for class 2 (virginica)
plt.scatter(X[y == 2, 0], X[y == 2, 1], label='Virginica', c='b')
# Add labels and a legend
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

# Show the plot
plt.show()

#We split the data between the train and the test
# 10% of samples are used as a test sample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

dt = tree.DecisionTreeClassifier() #Create the decision tree model

dt.fit(X_train, y_train)

print('the features from the test set', X_test) #print out the features form test set

print('the true labels of the test sample', y_test) #print out the true labels of test samples
print('the predicted labels of X_test', dt.predict(X_test)) #predict out the predicted labels

print('The test score is', dt.score(X_test, y_test)) #predict out the test score

print("Prediction Accuracy:", accuracy_score(y_test, dt.predict(X_test))) #print out the prediction accuracy
