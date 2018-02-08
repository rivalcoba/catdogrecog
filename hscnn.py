# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Summarizing the model
classifier.summary()

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('hsset/train',
target_size = (32, 32),
batch_size = 32,
class_mode = 'binary')

print("Classes: {0}".format(training_set.class_indices))

test_set = test_datagen.flow_from_directory('hsset/validation',
target_size = (32, 32),
batch_size = 32,
class_mode = 'binary')

history = classifier.fit_generator(
training_set,
steps_per_epoch = 3510 + 399 // 32,
epochs = 25,
validation_data = test_set,
verbose = 1,
validation_steps = 2000
)

#save model
model_json = classifier.to_json()
open('hs_architecture.json', 'w').write(model_json)
#And the weights learned by our deep network on the training set
classifier.save_weights('hs_architecture.h5', overwrite=True)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('hsset/test/h0.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Street'
else:
    prediction = 'Human'
print("Prediction {0}".format(prediction))