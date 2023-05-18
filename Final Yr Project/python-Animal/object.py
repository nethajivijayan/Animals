import tensorflow.python.keras
from PIL import Image, ImageOps
import numpy as np
import time
import cv2
from time import sleep
import requests


model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

while True:
        ret, image = cap.read()
        cv2.imwrite('image.jpg', image)
                

        np.set_printoptions(suppress=True)


        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = Image.open('image.jpg')

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)

        
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array


        newsize = (480, 480)
        image = image.resize(newsize)
        image.show()


        prediction = model.predict(data)
        ynew=np.argmax(prediction,axis=1)
        print(prediction)
        if ynew == [0]:
                print(" ")
        if ynew == [1]:
                print("Tiger")
        if ynew == [2]:
                print("Lion")
        if ynew == [3]:
                print("Cheetah")
        if ynew == [4]:
                print("Elephant")
        if ynew == [5]:
                print("Panda")
        print(ynew)