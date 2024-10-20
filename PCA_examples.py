import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.array([[1, 2, 3], [2, 1, 1], [3, 8, 5], [0, 2, 0]])
# X = np.array([[1, 2, 3], [2, 1, 1], [3, 8, 5], [0, 2, 0]])

# plt.plot(X[:,0], X[:,1], 'r*')
# plt.show()

# Create the Principal component analysis (PCA)
# and assign the number of components to 2.
pca = PCA(n_components=3)

# Fit the model with X.
pca.fit(X)

# Percentage of variance explained by
# each of the selected components.
print(pca.explained_variance_ratio_)


# Fit the model with X and apply the dimensionality reduction on X.
X_reduced = PCA(n_components=2).fit_transform(X)
print(X_reduced)