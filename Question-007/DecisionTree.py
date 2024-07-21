import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def calculate_gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - sum(p ** 2 for p in probabilities)

def calculate_gain(X, y, feature, threshold, criterion):
    left_indices = X[:, feature] < threshold
    y_left, y_right = y[left_indices], y[~left_indices]

    if criterion == "entropy":
        parent_metric = calculate_entropy(y)
        left_metric = calculate_entropy(y_left)
        right_metric = calculate_entropy(y_right)
    else:  # Gini impurity
        parent_metric = calculate_gini(y)
        left_metric = calculate_gini(y_left)
        right_metric = calculate_gini(y_right)
    
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    
    gain = parent_metric - (weight_left * left_metric + weight_right * right_metric)
    return gain

def build_tree(X, y, max_depth, criterion):
    if len(set(y)) == 1 or max_depth == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    n_features = X.shape[1]
    best_feature = best_threshold = None
    best_gain = -1
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = calculate_gain(X, y, feature, threshold, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    if best_gain == 0:
        value = max(set(y), key=list(y).count)
        return Node(value=value)
    
    left_indices = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[~left_indices], y[~left_indices]
    
    left_subtree = build_tree(X_left, y_left, max_depth - 1, criterion)
    right_subtree = build_tree(X_right, y_right, max_depth - 1, criterion)
    
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] < tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)

def evaluate_tree(tree, X_test, y_test):
    y_pred = [predict_tree(tree, x) for x in X_test]
    accuracy = sum(y_pred == y_test) / len(y_test)
    return accuracy

df = pd.read_csv("weather_forecast.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Play_Yes', axis=1).values
y = df['Play_Yes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_id3 = build_tree(X_train, y_train, max_depth=5, criterion="entropy")
accuracy_id3 = evaluate_tree(tree_id3, X_test, y_test)
print("ID3 Algorithm Results:")
print(f"Accuracy: {accuracy_id3}")

tree_cart = build_tree(X_train, y_train, max_depth=5, criterion="gini")
accuracy_cart = evaluate_tree(tree_cart, X_test, y_test)
print("CART Algorithm Results:")
print(f"Accuracy: {accuracy_cart}")