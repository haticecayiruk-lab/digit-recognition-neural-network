

import numpy as np

''''
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.show()
'''

def calculateLoss( outputs, targets):
    confidence = outputs[range(len(outputs)), targets]
    clippedConfidence = np.clip(confidence , 1e-7, 1-1e-7 )
    losses = -np.log(clippedConfidence)
    meanLoss = np.mean(losses)
    return meanLoss



softmaxOutputs= np.array([ [0, 0, 0.25],     
                           [1, 0.8, 0.56],    
                           [1-1e-7, 0.4, 0.43] ])

classTargets = [0,0,0]

#loss = calculateLoss(softmaxOutputs, classTargets)
#print(loss)

x = np.load("test.npy")
print(x)