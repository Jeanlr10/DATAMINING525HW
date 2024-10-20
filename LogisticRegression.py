from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = np.array([[2.22, 2.4], [2.99, 2.3],
              [3.2, 3.2], [2.9, 3.2], [2.3, 12.32],[4.3, 2.4],[2.3, 5.4],
              [12.4, 2.3], [5.1, 5.4], [5.3, 5.9]])

y = np.array([2, 2, 1, 1, 1, 1, 1, 0, 0, 0]) #define the label/class

# Plot the samples. Class/label: black, blue, and red.
plt.scatter(x[0:2,0], x[0:2,1], color='black')
plt.scatter(x[2:7,0], x[2:7,1], color='blue')
plt.scatter(x[7:10,0], x[7:10,1], color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Splitting the data into training and testing data, 4:1, 0.2=20%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create the logistic regression model.clf is a handle
clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=200)

# train the logistic regression model ith training data (8 samples if test_size=0.2).
clf.fit(X_train, y_train)

print('the samples in testing set:\n', X_test)
print('\nthe true classes/labels of samples in testing set:\n', y_test)
print('\nthe predicted classes/labels of samples in testing set:\n', clf.predict(X_test))
print('\nthe probabilities of testing samples belong to class 0/1/2:\n',clf.predict_proba(X_test))
print('\nThe testing score: ', clf.score(X_test, y_test))




