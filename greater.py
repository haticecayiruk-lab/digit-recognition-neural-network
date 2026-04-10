
import numpy as np

class layerTrained:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases= biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 
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



class Relu: 
    def forwardRelu(self, inputs):
        self.output = np.where(inputs <= 0, 0, inputs)
        return self.output    
    
reluInstance = Relu()
softmaxInstance = softmax()



w1 = np.array([[0.47468516, -0.6262392, -0.0058295, 0.55658071, -0.05775182]])
b1 = np.array([[-0.15383985, 0.34791339, 0.02737126, -0.1651453, 0.01902208]])
w2 = [[-0.29528969,  0.42979728],
      [ 0.5250772 , -0.49528116],
      [ 0.03261702, -0.02470278],
      [-0.40895036,  0.38657362],
      [-0.14110351, -0.2227163 ]]
b2 = np.array([[1.02731593, -1.02731593]])


layer1 = layerTrained(w1,b1)
layer2= layerTrained(w2,b2)

x = float (input("enter a number"))

result = layer1.forward(x)
reluOutput = reluInstance.forwardRelu(result)
result2= layer2.forward(reluOutput)
softmaxOutput = softmaxInstance.forwardSoftmax(result2)

print(softmaxOutput)

if softmaxOutput[0,0] > softmaxOutput[0,1]:
    print("your value is smaller than or equal to 3.2" )
else: 
    print("your value is greater than 3.2")    

