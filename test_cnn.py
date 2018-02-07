# python test_cnn.py -t catdogset/single_prediction

# Importing dependencies
import cv2
from imutils import paths
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
import scipy.misc
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True, help="folder containing the test iamges")
args = vars(ap.parse_args())

# Loading Model
model_architecture = "catdog_architecture.json"
model_weights = "catdog_architecture.h5"
classifier = model_from_json(open(model_architecture).read())
classifier.load_weights(model_weights)

# Compiling classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Grab the Training image paths
imageTrainPaths = list(paths.list_images(args['test']))

for imagePath in imageTrainPaths:
    test_image = image.load_img(imagePath, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("File: {0} \n Prediction {1}".format( imagePath.split('\\')[1], prediction))