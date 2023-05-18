import matplotlib.pyplot as plt
import numpy as np
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import random
from init import oldweight
from init import connweight
from init import oldactivation
from init import presynapticnewron
import numpy as np
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
x1=0
y1=0


class NeuralNetworknew():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((5000, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            
            weightincrease=sigmoid(oldweight+(oldweight*self.synaptic_weights) - (oldweight/1000))
            
            weightchange=sigmoid(oldweight-(oldweight/1000))
            
            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        
        return output



class NeuralNetwork():
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors



while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Epochs"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    frame1=cv2.imread("1.jpg")
    #frame1 = imutils.resize(frame, width=500, height=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    input_vectors = np.array(firstframe)
            
    targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

    learning_rate = 0.1

    neural_network = NeuralNetwork()

    training_error = neural_network.train(input_vectors, targets, 5000)
    
    activationincrease=activationchange+(oldweight*neural_network/5)-(oldactivation/100)
    
    if(training_error<1):
        activationchange=oldactivation-(oldactivation/100) 
    else:
        threshhold=sigmoid(outputs-oldoutput*0.01) - (oldthreshold/10000)
        oldoutput=output
    
    target = 0
    percentage=output
    mse = np.square(prediction - target)


    while x1<500:
        while y1<100:
            if percentage < 99:
                if random.random()>1:
                    frame1=cv2.circle(frame1,(x1,y1),3,(0,0,255),3)
                else:
                    frame1=cv2.circle(frame1,(x1,y1),3,(0,255,0),3)
            y1=y1+1
        x1=x1+1
        y1=0

    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")