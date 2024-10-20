from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#define training samples
#x = np.array([[6,99],[7,86],[8,70],[7,88],[2,111],[17,86],[2,103],[9,87],[4,94],[11,78]])
#target = np.array([1,100,2,1,2,1,2,100,2,2])

#define testing samples
#define training samples
x=np.array([[1], [3], [4], [5], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17]]) #number of quarters after jan 2008
target=([[50],  [41], [45],[41], [26], [22], [26], [18], [16], [15], [12],  [8],  [4], [12], [6]]) #number of banks failed





#define testing samples
x_test = np.array([[2], [6]])
y_test = np.array([[45],[30]])



#splitting the data into training and testing data
lg = LinearRegression()

#train the model
lg.fit(x,target)

#print the testing score
#the best score is 1.0 and can be negative
print(lg.score(x_test,y_test))
print(lg.score(x,target))

y_pred = lg.predict(x_test)
print("The predicted labels: ", y_pred)
print("True labels: ",y_test)

x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
y_pred = lg.predict(x_range)

plt.scatter(x_test,y_test,color='b',label='Test Data Points')
plt.scatter(x,target,color='black',label='Data Points')
plt.plot(x_range, y_pred, color='blue', linewidth=2, label='Regression Line')
plt.xlabel('Number of Quarters after January 2008')
plt.ylabel('Number of Bank Failures')
plt.legend()

plt.show()


# Design training dataset & test dataset