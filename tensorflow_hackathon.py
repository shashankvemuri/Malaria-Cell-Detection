from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import imageio
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from resizeimage import resizeimage

i = 0
arr_infected =[]

for filename in os.listdir("/Users/shashank/Desktop/cell_images/Parasitized/"):
    if filename.endswith(".png") or filename.endswith(".py"): 
        print(str(i) + "picture Parasitized")
        i+=1
        arr_infected.append(cv2.imread("/Users/shashank/Desktop/cell_images/Parasitized/" + filename))
        if len(arr_infected) == 500:
            break
        continue
    else:
        continue
    
i = 0
#arr_uninfected = []
for filename in os.listdir("/Users/shashank/Desktop/cell_images/Uninfected/"):
    if filename.endswith(".png") or filename.endswith(".py"): 
        print(str(i) + "picture Uninfected")
        i+=1
        arr_infected.append(cv2.imread("/Users/shashank/Desktop/cell_images/Uninfected/" + filename))
        if len(arr_infected) == 1000:
            break
        continue
    else:
        continue
classNames = []
for i in range(500):
    classNames.append(0)
for i in range(500):
    classNames.append(1)
    
print(len(classNames))
print(classNames)


npa = np.asarray(arr_infected)
npm = np.asarray(classNames)

print(npm.shape)

train_images = npa[0:400]
train_labels = npm[0:400]
test_images = npa[401:500]
test_labels = npm[401:500]

train_images2 = npa[500:900]
train_labels2 = npm[500:900]
test_images2 = npa[901:1001]
test_labels2 = npm[901:1001]

#FULL ARRAYS
trainImages = np.concatenate((train_images,train_images2))
trainLabels = np.concatenate((train_labels,train_labels2))
testImages = np.concatenate((test_images,test_images2))
testLabels = np.concatenate((test_labels,test_labels2))

'''
def image_resize(image, width = 145, height = 145, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = (145, 145)
    (h, w) = image.shape[:2]


    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


for im in trainImages:
    im = image_resize(im, width =145, height = 145)

for im in testImages:
    im = image_resize(im, width =145, height = 145)     
'''


for im in arr_infected:
    new_width  = 145
    new_height = 145
    im = im.resize(new_width, new_height, refcheck=False)         


#TESTS
print(trainImages.shape)
print(trainLabels.shape)
print(testImages.shape)
print(testLabels.shape)

'''
trainImages = train_images / 255.0
testImages = test_images / 255.0
'''

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
    plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(145, 145)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainImages, trainLabels, epochs=10)

test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)

print('\nTest accuracy:', test_acc)
plt.imshow(arr_infected[10])
