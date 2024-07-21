import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                z = self.W.T.dot(x)
                a = self.activation_fn(z)
                e = d[i] - a
                self.W = self.W + self.lr * e * x

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d_or = np.array([0, 1, 1, 1])

and_perceptron = Perceptron(input_size=2)
or_perceptron = Perceptron(input_size=2)

and_perceptron.fit(X_and, d_and)
or_perceptron.fit(X_or, d_or)

print("AND Function:")
for x in X_and:
    print(f"{x} -> {and_perceptron.predict(x)}")

print("\nOR Function:")
for x in X_or:
    print(f"{x} -> {or_perceptron.predict(x)}")