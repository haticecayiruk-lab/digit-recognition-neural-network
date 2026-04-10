
import numpy as np
import torch 
from torchvision import datasets, transforms
import torch.nn.functional as F


class layerTrained:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases= biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 
        return self.output

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





reluInstance = Relu()
softmaxInstance = softmax()

w1 = np.load("w12.npy")
w2 = np.load("w22.npy")
w3 = np.load("w32.npy")
w4 = np.load("w42.npy")
b1 = np.load("b12.npy")
b2 = np.load("b22.npy")
b3 = np.load("b32.npy")
b4 = np.load("b42.npy")

layer1 = layerTrained(w1,b1)
layer2= layerTrained(w2,b2)
layer3 = layerTrained(w3,b3)
layer4 = layerTrained(w4,b4)



# Initialize the variables for testing
correctPredictions = 0
totalSamples = 0

# Iterate through the test loader
for batch_idx, (images, labels) in enumerate(test_loader):

    flattened_images = images.view(images.size(0), -1) # flatten the images
    flattened_images = flattened_images.detach().numpy() #Convert Tensor to numpy array

    result1 = layer1.forward(flattened_images)
    reluOutput1 = reluInstance.forwardRelu(result1)

    result2 = layer2.forward(reluOutput1)
    reluOutput2 = reluInstance.forwardRelu(result2)

    result3 = layer3.forward(reluOutput2)
    reluOutput3 = reluInstance.forwardRelu(result3)

    result4 = layer4.forward(reluOutput3)
    softmaxOutput = softmaxInstance.forwardSoftmax(result4)

    
    labelsNumpy = labels.detach().numpy()  # Convert Tensor to numpy array

    # Predictions: Get the index of the highest probability for each sample
    predictedLabels = np.argmax(softmaxOutput, axis=1)
    totalSamples += labels.size(0)

    correctPredictions += np.sum(predictedLabels == labelsNumpy)
                                                
accuracy = correctPredictions / totalSamples # Calculate and print accuracy
print(f"Test Accuracy: {accuracy * 100}%")














