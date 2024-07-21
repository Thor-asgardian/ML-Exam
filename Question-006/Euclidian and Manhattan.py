import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k, distance_fn):
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = [self.distance_fn(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load the dataset
df = pd.read_csv('glass.csv')
X = df.drop('Type', axis=1).values
y = df['Type'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Using Euclidean distance
clf_euclidian = KNN(k=3, distance_fn=euclidian_distance)
clf_euclidian.fit(X_train, y_train)
predictions_euclidian = clf_euclidian.predict(X_test)
accuracy_euclidian = np.sum(predictions_euclidian == y_test) / len(y_test)
print("Accuracy with Euclidean distance:", accuracy_euclidian)

# Using Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

clf_manhattan = KNN(k=3, distance_fn=manhattan_distance)
clf_manhattan.fit(X_train, y_train)
predictions_manhattan = clf_manhattan.predict(X_test)
accuracy_manhattan = np.sum(predictions_manhattan == y_test) / len(y_test)
print("Accuracy with Manhattan distance:", accuracy_manhattan)