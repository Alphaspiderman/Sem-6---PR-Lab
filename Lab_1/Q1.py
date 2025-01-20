import numpy as np

# Define the activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron learning algorithm
def perceptron_learning(X, y, learning_rate=0.1, epochs=10):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            # Calculate the linear combination
            linear_output = np.dot(X[i], weights) + bias
            # Apply the activation function
            prediction = step_function(linear_output)
            # Update weights and bias
            error = y[i] - prediction
            weights += learning_rate * error * X[i]
            bias += learning_rate * error

    return weights, bias

# Input data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Train the perceptron
weights, bias = perceptron_learning(X, y)

# Test the perceptron
def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return step_function(linear_output)

# Print the results
for x in X:
    print(f"Input: {x}")
    print(f"Predicted Output: {predict(x, weights, bias)}")
