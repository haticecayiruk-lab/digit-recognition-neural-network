
import numpy as np
import matplotlib.pyplot as plt

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






layer1= layer(1,5)
result = layer1.forward(x)
reluOutput = reluInstance.forwardRelu(result)
layer2 =layer(5,2)
result2= layer2.forward(reluOutput)
softmaxOutput = softmaxInstance.forwardSoftmax(result2)





print("result\n", result )
print("relu\n", reluOutput)
print("result2\n", result2)
print("softmax\n", softmaxOutput)


weights1 = layer1.GetWeights()
biases1 = layer1.GetBiases()
print("Weights1:\n", weights1)
print("Biases1:\n", biases1)

weights2 = layer2.GetWeights()
biases2 = layer2.GetBiases()
print("Weights2:\n", weights2)
print("Biases2:\n", biases2)



