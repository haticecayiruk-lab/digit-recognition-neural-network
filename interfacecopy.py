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
app.title('digit recognition')


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

        # Update the label with the new image
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        #label_text.config(text=f"Loaded Image: {filepath.split('/')[-1]}")  # Show filename

        #process the image

        if image is None:
            messagebox.showerror('Python Error', 'Error: please enter a valid image!')

        numpyImage = np.array(image)
        formattedImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2BGR)
        filteredImage = cv2.pyrMeanShiftFiltering(formattedImage, 2, 10)
        
        blurredImage = cv2.medianBlur(filteredImage, 11)
        imageSmall = cv2.resize(blurredImage, (28,28), interpolation= cv2.INTER_AREA)
        imageGrey = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY)

        #normalise the image
        normalizedImage = cv2.normalize(
        imageGrey, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) 

        
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

        print(index)



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
