import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances

iris = load_iris()
data = iris.data[:6]

def proximity_matrix_custom(data):
    n = data.shape[0]
    proximity_matrix = np.zeros((n, n))
    for i in range (n):
        for j in range (i + 1, n):
            proximity_matrix[i][j] = np.linalg.norm(data[i] - data[j])
            proximity_matrix[j][i] = proximity_matrix[i][j]
            return proximity_matrix

def proximity_matrix_sklearn(data):
    proximity_matrix = euclidean_distances(data)
    return proximity_matrix

prox_matrix_custom = proximity_matrix_custom(data)
prox_matrix_sklearn = proximity_matrix_sklearn(data)

print("Proximity Matrix using sklearn: ")
print(prox_matrix_sklearn)

print("Proximity Matrix without using sklearn: ")
print(prox_matrix_custom)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
dendrogram(linkage(prox_matrix_sklearn[np.triu_indices(prox_matrix_sklearn.shape[0], k = 1)], method='single'))
plt.title('Dendrogram-Single Linkage')
plt.xlabel('Data points')
plt.ylabel('distance')

plt.subplot(1, 3, 2)
dendrogram(linkage(prox_matrix_sklearn[np.triu_indices(prox_matrix_sklearn.shape[0], k = 1)], method='complete'))
plt.title('Dendrogram-complete Linkage')
plt.xlabel('Data points')
plt.ylabel('distance')

plt.tight_layout()
plt.show()