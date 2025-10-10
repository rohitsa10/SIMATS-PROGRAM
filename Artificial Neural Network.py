import numpy as np

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sample input
X = np.array([0.5, 0.1, 0.4])  # 3 input features

# Network architecture
input_neurons = 3
hidden_neurons = 4
output_neurons = 2

# Randomly initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(hidden_neurons, input_neurons)   # Weights: input -> hidden
b1 = np.random.rand(hidden_neurons)                 # Biases for hidden layer
W2 = np.random.rand(output_neurons, hidden_neurons) # Weights: hidden -> output
b2 = np.random.rand(output_neurons)                 # Biases for output layer

# --------------------------
# Forward propagation
# --------------------------

# Hidden layer
z1 = np.dot(W1, X) + b1
a1 = sigmoid(z1)

# Output layer
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

print("Hidden layer output:", a1)
print("Output layer result:", a2)
