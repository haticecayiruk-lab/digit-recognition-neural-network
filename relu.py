
import numpy as np


softmaxOutputs= np.array([ [0, 0, 0.25],      # I came up with a different set of values 
                           [1, 0.8, 0.56],    # beause softmaxOutputs can only be postive 
                           [1-1e-7, 0.4, 0.43] ])

classTargets = [2,2,2]
confidence = softmaxOutputs[range(len(softmaxOutputs)), classTargets]
clippedConfidence = np.clip(confidence , 1e-7, 1-1e-7 )
losses = -np.log(clippedConfidence)
meanLoss = np.mean(losses)

'''''
print("clipped\n", clippedConfidence)
print("losses\n", losses)
print("meanloss:\n", meanLoss  )
'''

test = 3

#np.save("x.npy", x)

np.save("test.npy", test)


'''''

inputs = np.array([[2.1,0.004,0,1,-3.3,5000],
                   [1,1,-8.567,-4,3,0],
                   [-6,-7,87,4,0,0.1]])




for row in inputs:
    max = 0
    for i in range(len(row)):
        if row[i]> max:
            max = row[i] 
    
    for i in range(len(row)):
        row[i] = row[i]-max 


for row in inputs:
    for i in range(len(row)):
        row[i] = np.exp(row[i])

 
for row in inputs:
    normBase = 0
    for i in range(len(row)):
        normBase = normBase + row[i]
    print("normase:\n", normBase)

    for i in range(len(row)):
        row[i] = row[i]/normBase




import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilized softmax
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Generate synthetic dataset
np.random.seed(42)
X = np.random.uniform(0, 6, 1000).reshape(-1, 1)  # Random numbers between 0 and 6
y = np.array([[1, 0] if x <= 3.2 else [0, 1] for x in X])  # Class labels

# Network parameters
input_size = 1
hidden_size = 5
output_size = 2

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# ReLU activation and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Forward pass
def forward(X):
    Z1 = np.dot(X, W1) + b1  # Input to hidden layer
    A1 = relu(Z1)            # Activation in hidden layer
    Z2 = np.dot(A1, W2) + b2  # Input to output layer
    A2 = softmax(Z2)         # Softmax activation for probabilities
    return A1, A2

# Backpropagation
def backprop(X, y, A1, A2, learning_rate=0.01):
    m = X.shape[0]  # Number of samples

    # Compute the loss (cross-entropy)
    loss = -np.sum(y * np.log(A2 + 1e-9)) / m  # Add epsilon to avoid log(0)

    # Gradients for output layer
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Gradients for hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights and biases
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return loss

# Training the network
epochs = 1000
losses = []
for epoch in range(epochs):
    A1, A2 = forward(X)  # Forward pass
    loss = backprop(X, y, A1, A2)  # Backpropagation
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Visualize loss over time
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Testing the network
test_X = np.array([[2.0], [3.5], [4.8], [1.2], [3.2]])
_, test_A2 = forward(test_X)
predictions = np.argmax(test_A2, axis=1)
print("Test Inputs:", test_X.flatten())
print("Predicted Classes:", predictions)





'''
  






