# python test_cnn_hs.py -t hsset/test

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
model_architecture = "hs_architecture.json"
model_weights = "hs_architecture.h5"
classifier = model_from_json(open(model_architecture).read())
classifier.load_weights(model_weights)

# Compiling classifier
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Grab the Training image paths
imageTrainPaths = list(paths.list_images(args['test']))

for imagePath in imageTrainPaths:
    test_image = image.load_img(imagePath, target_size = (32,32))
    preview_image = np.array(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Street'
    else:
        prediction = 'Human'
    print("File: {0} \n Prediction {1}".format( imagePath.split('\\')[1], prediction))
    # Previewing thw Image
    proba = result[0][0]
    #proba = head if head > nohead else nohead
    label = "{0}".format(prediction)
    # draw the label on the image
    # Convert RGB to BGR 
    preview_image = preview_image[:, :, ::-1].copy()
    preview_image = cv2.resize(preview_image, (400, 400))
    cv2.putText(preview_image, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)    
    # show the output image
    cv2.imshow("Preview Image", preview_image)
    cv2.waitKey(0)
    