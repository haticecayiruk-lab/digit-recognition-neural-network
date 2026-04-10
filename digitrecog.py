
import numpy as np
import torch 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define a transformation (e.g., normalize the data)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
])


# Download and load the training and test datasets
train_dataset = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root="mnist_data", train=False, transform=transform, download=False)

# Create data loaders for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



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


# instance of the layer class
layer1= layer(784,256)  
layer2= layer(256,128)
layer3= layer(128,64)
layer4= layer(64,10)


# assign varibales names to the weights and biases
global w1, w2, w3, w4, b1, b2, b3, b4
w1 = layer1.GetWeights()
b1 = layer1.GetBiases()
w2 = layer2.GetWeights()
b2 = layer2.GetBiases()
w3 = layer3.GetWeights()
b3 = layer3.GetBiases()
w4 = layer4.GetWeights()
b4 = layer4.GetBiases()


def backProp(A1, A2, A3, A4, x, y, learningRate):

    global w1, w2, w3, w4, b1, b2, b3, b4
    m = x.shape[0]  # Number of samples

    yOneHot= F.one_hot(y, num_classes=10)
    yOneHot = yOneHot.detach().numpy()  # Convert Tensor to numpy array
    
    #gradient of output layer 
    dz4 = softmaxOutput - yOneHot 
    dw4= np.dot(reluOutput3.T, dz4)/m
    db4 = np.sum(dz4, axis=0, keepdims=True) / m  

    #gradient of 4th hidden layer 
    dA3 = np.dot(dz4, w4.T)
    dz3 = dA3 * reluDerivative(reluOutput3)
    dw3 = np.dot(reluOutput2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m
    
    # gradient of 3rd hidden layer 
    dA2 = np.dot(dz3, w3.T)
    dz2 = dA2 * reluDerivative(reluOutput2)
    dw2 = np.dot(reluOutput1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    #gradient of 2nd hidden layer 
    dA1 = np.dot(dz2, w2.T)
    dz1 = dA1 * reluDerivative(reluOutput1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

 # Update weights and biases
    w1 -= learningRate * dw1
    b1 -= learningRate * db1
    w2 -= learningRate * dw2
    b2 -= learningRate * db2
    w3 -= learningRate * dw3
    b3 -= learningRate * db3
    w4 -= learningRate * dw4
    b4 -= learningRate * db4



# traning the network 
epochs = 10
losses = []
learningRate = 0.1


for epoch in range(epochs):
    epochLoss = 0  # Track total loss for the epoch

    # Accessing data and labels from train_loader
    for batch_idx, (images, labels) in enumerate(train_loader):

        flattened_images = images.view(images.size(0), -1)
        flattened_images = flattened_images.detach().numpy() #Convert Tensor to numpy array


        #the forward pass
        result = layer1.forward(flattened_images)
        reluOutput1 = reluInstance.forwardRelu(result)

        result2= layer2.forward(reluOutput1)
        reluOutput2 = reluInstance.forwardRelu(result2)

        result3= layer3.forward(reluOutput2)
        reluOutput3 = reluInstance.forwardRelu(result3)

        result4 = layer4.forward(reluOutput3)
        softmaxOutput = softmaxInstance.forwardSoftmax(result4)

        #backpropagation and loss
        loss = calculateLoss(softmaxOutput, labels)
        backProp(reluOutput1, reluOutput2, reluOutput3, softmaxOutput, flattened_images, labels, learningRate)

        epochLoss += loss
        
        
    #calculate and store average epoch loss 
    averageLoss = epochLoss / len(train_loader)
    losses.append(averageLoss)

# Visualize loss over time
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()    


np.save("b12.npy", b1)
print("b1SAVED")
np.save("b22.npy", b2)
np.save("b32.npy", b3)
np.save("b42.npy", b4)
np.save("w12.npy", w1)
np.save("w22.npy", w2) 
np.save("w32.npy", w3)
np.save("w42.npy", w4)





