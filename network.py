import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)  # Set a global seed

class layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 
        return self.output
    def GetWeights(self):
        return self.weights  
    def GetBiases(self):
        return self.biases 
    



class Relu: 
    def forwardRelu(self, inputs):
        self.output = np.where(inputs <= 0, 0, inputs)
        return self.output
    

class softmax:
    def forwardSoftmax(self, inputs):

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
   

            for i in range(len(row)):
                row[i] = row[i]/normBase
    
        self.output = inputs
        return self.output 



def calculateLoss( outputs, targets):
    confidence = outputs[range(len(outputs)), targets]
    clippedConfidence = np.clip(confidence , 1e-7, 1-1e-7 )
    losses = -np.log(clippedConfidence)
    meanLoss = np.mean(losses)
    return meanLoss
    
def reluDerivative(x):
    return (x > 0).astype(float)


reluInstance = Relu()
softmaxInstance = softmax()



global w1, b1, w2, b2

layer1= layer(1,5)  # make an instance of the layer class 
layer2= layer(5,2)

w1 = layer1.GetWeights()
b1 = layer1.GetBiases()
w2 = layer2.GetWeights()
b2 = layer2.GetBiases()


def backProp(A1, A2, x, y, learningRate):

    global w1, b1, w2, b2

    m = x.shape[0]  # Number of samples

    #gradient of output layer 
    dz2 = softmaxOutput - y  
    dw2= np.dot(reluOutput.T, dz2)/m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  

    # gradient for hidden layer
    dA1 = np.dot(dz2, w2.T)
    dz1 = dA1 * reluDerivative(reluOutput)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m


 # Update weights and biases
    w1 -= learningRate * dw1
    b1 -= learningRate * db1
    w2 -= learningRate * dw2
    b2 -= learningRate * db2





# Generate synthetic dataset
np.random.seed(42)
x = np.random.uniform(-5, 5, 1000).reshape(-1, 1)  # Random numbers between 0 and 6
y = np.array([[1, 0] if v <= 3.2 else [0, 1] for v in x])  # Class labels



# traning the network 
epochs = 1000
losses = []


for epoch in range(epochs):

    #the forward pass
    result = layer1.forward(x)
    reluOutput = reluInstance.forwardRelu(result)
    result2= layer2.forward(reluOutput)
    softmaxOutput = softmaxInstance.forwardSoftmax(result2)

    #backpropagation and loss
    loss = calculateLoss(softmaxOutput, np.argmax(y, axis=1))
    backProp(reluOutput, softmaxOutput, x, y, 0.01)

    # print losses to keep track of network's performance
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


print("w1:\n", w1)
print("b1:\n", b1)
print("w2:\n", w2)
print("b2:\n", b2)

# Visualize loss over time
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

'''''
x = np.load("x.npy")
print(x)

'''