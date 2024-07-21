import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def create_model(layers):
    model = {'weights': [], 'biases': []}
    for i in range(len(layers) - 1):
        model['weights'].append(np.random.rand(layers[i], layers[i + 1]))
        model['biases'].append(np.random.rand(1, layers[i + 1]))
    return model

def forward_propagation(model, X):
    activations = [X]
    for i in range(len(model['weights'])):
        net_input = np.dot(activations[-1], model['weights'][i]) + model['biases'][i]
        activations.append(sigmoid(net_input))
    return activations

def train_model(model, X, y, epochs=2000, learning_rate=0.1):
    for _ in range(epochs):
        activations = forward_propagation(model, X)
        error = y - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]
        
        for i in range(len(model['weights']) - 2, -1, -1):
            deltas.append(deltas[-1].dot(model['weights'][i + 1].T) * sigmoid_derivative(activations[i + 1]))
        deltas.reverse()

        for i in range(len(model['weights'])):
            model['weights'][i] += learning_rate * activations[i].T.dot(deltas[i])
            model['biases'][i] += learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

def evaluate_model(model, X, y):
    predictions = np.round(forward_propagation(model, X)[-1])
    accuracy = np.mean(predictions == y) * 100
    return accuracy, predictions

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_andnot = np.array([[0], [0], [1], [0]])
y_xor = np.array([[0], [1], [1], [0]])

andnot_model = create_model([2, 1])
xor_model = create_model([2, 2, 1])

train_model(andnot_model, X, y_andnot)
train_model(xor_model, X, y_xor)

andnot_acc, andnot_predictions = evaluate_model(andnot_model, X, y_andnot)
xor_acc, xor_predictions = evaluate_model(xor_model, X, y_xor)

print(f"AND-NOT Model Accuracy: {andnot_acc:.2f}%")
print("AND-NOT Predictions:")
print(andnot_predictions)

print(f"XOR Model Accuracy: {xor_acc:.2f}%")
print("XOR Predictions:")
print(xor_predictions)