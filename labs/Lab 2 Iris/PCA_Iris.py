import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

pca=PCA(n_components=2)
pca.fit(iris.data)
Iris_reduced=PCA(n_components=2).fit_transform(iris.data)
print(pca.explained_variance_ratio_)
print(pca.fit_transform(iris.data))
class_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
colors = ['r', 'y', 'b']
for i, class_name in enumerate(class_names):
    #plt.scatter(Iris_reduced[iris.target==i, 0], Iris_reduced[iris.target==i, 1], color=colors[i], label=class_name, edgecolor='k')
    plt.scatter(iris.data[iris.target==i, 0], iris.data[iris.target==i, 1], color=colors[i], label=class_name, edgecolor='k')
plt.legend()
plt.show()
