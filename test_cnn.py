# python -i catdogset/single_prediction/cat0.jpg
import cv2
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
import scipy.misc

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the image to being classified")

# Loading Model
model_architecture = "catdog_architecture.json"
model_weights = "catdog_architecture.h5"
classifier = model_from_json(open(model_architecture).read())
classifier.load_weights(model_weights)

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

test_image = image.load_img("catdogset/single_prediction/cat0.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print("Prediction {0}".format(prediction))