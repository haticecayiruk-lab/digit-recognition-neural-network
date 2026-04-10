import tkinter as tk
from tkinter import PhotoImage
from PIL import ImageTk, Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torchvision import datasets, transforms
from tkinter import filedialog
from tkinter import messagebox


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


app = tk.Tk()
app.geometry("500x600")
app.title('Digit Recognition by Hatice Cayir')


# Function to open file dialog and load the image
def loadAndProcessImage():
    # Open a file dialog to select an image
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if filepath:  # If the user selected a file
        # Open and display the selected image
        global image
        image = Image.open(filepath)
        image = image.resize((200, 200))  # Resize for display purposes
        photo = ImageTk.PhotoImage(image)

        #Update the label with the new image
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        
        #process the image

        if image is None:
            messagebox.showerror('Python Error', 'Error: please enter a valid image!')

        numpyImage = np.array(image)
        formattedImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2BGR)
        filteredImage = cv2.pyrMeanShiftFiltering(formattedImage, 2, 10)
        
        blurredImage = cv2.medianBlur(filteredImage, 9)
        imageSmall = cv2.resize(blurredImage, (28,28), interpolation= cv2.INTER_AREA)
        imageGrey = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY)

        #normalise the image
        normalizedImage = cv2.normalize(
        imageGrey, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) 

        # Display the processed image using OpenCV
        #cv2.imshow("Processed Image", imageGrey)  # Display the grayscale processed image
        #cv2.waitKey(0)  # Wait until a key is pressed
        #cv2.destroyAllWindows() 

        
        finalImage = normalizedImage.flatten()
        result = 1- finalImage

        result1 = layer1.forward(result)
        reluOutput1 = reluInstance.forwardRelu(result1)

        result2 = layer2.forward(reluOutput1)
        reluOutput2 = reluInstance.forwardRelu(result2)

        result3 = layer3.forward(reluOutput2)
        reluOutput3 = reluInstance.forwardRelu(result3)

        result4 = layer4.forward(reluOutput3)
        softmaxOutput = softmaxInstance.forwardSoftmax(result4)

        
        index = np.argmax(softmaxOutput, axis = 1)
        confidence = softmaxOutput[0,index]
        labelText1.config(text = "the predicted digit: "+ str(index))
        labelText2.config(text ="the confidence of the network: " + str(confidence*100) + "%")

        print("layer2\n", reluOutput1)
        print("layer2\n" ,reluOutput2)
        print("layer3\n" ,reluOutput3)
        

    
load_button = tk.Button(app, text="Load Image", command=loadAndProcessImage)
load_button.pack(pady=10)

labelText1 = tk.Label(app, text="the predicted digit: ")
labelText1.pack()

labelText2 = tk.Label(app, text = "the confidence of the network: ")
labelText2.pack()

labelText3 = tk.Label(app, text="image being processed:" )
labelText3.pack(pady=10)


# Add a label to display the image
image_label = tk.Label(app)
image_label.pack(pady=10)
app.mainloop()


'''''

        labelText1.config(text = "the predicted digit: "+ str(index))
        labelText2.config(text ="the confidence of the network: " + str(confidence))

def printInput(): 
    inp = inputtxt.get(1.0, "end-1c") 
    lbl.config(text = "Provided Input: "+inp) 
    

# TextBox Creation 
inputtxt = tk.Text(app, 
                   height = 5, 
                   width = 20) 
  
inputtxt.pack() 
  
# Button Creation 
printButton = tk.Button(app, 
                        text = "Print",  
                        command = printInput) 
printButton.pack() 
  
# Label Creation 
lbl = tk.Label(app, text = "") 
lbl.pack()

picture = Image.open('D:\\Projeler\\neural network\\four.png')
picture = ImageTk.PhotoImage(picture)
image_label = tk.Label(app, image=picture)
image_label.pack()

#picture = PhotoImage(file = 'D:\\Projeler\\neural network\\four.png')
#pictureLabel = tk.Label(app, image = picture)
#pictureLabel.pack()
'''



'''''
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
])

train_dataset = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root="mnist_data", train=False, transform=transform, download=False)

index = 13 # Index of the image you want to access
imagee, label = test_dataset[index]

image_numpy = imagee.squeeze().numpy()
'''


#cv2.imwrite('D:\Projeler\neural network\63661cf7a3df52dce5e2c8d0419c20e5_t.jpeg', image)

#button = tk.Button(app, text='Stop', width=25, command=app.destroy)
#button.pack()

#label = tk.Label(app,
#    text="Hello, Tkinter",
#   foreground="white",  
#  background="black"  )
