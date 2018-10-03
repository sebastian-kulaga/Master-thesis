# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (120, 90))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
#(sparrow, dove, raven, lark, blackbird) = model.predict_classes(image)[0]
bird=model.predict_classes(image)
(sparrow,dove,raven,lark,blackbird)=model.predict(image)[0]
# build the label
if bird == 0:
	label = "Sparrow"
	proba = sparrow
if bird  == 1:
	label = "Dove"
	proba = dove
if bird == 2:
	label = "Raven"
	proba = raven
if bird  == 3:
	label = "Lark"
	proba = lark
if bird == 4:
	label = "Blackbird"
	proba = blackbird
label = "{}: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}%".format(label, sparrow*100,dove*100,raven*100,lark*100,blackbird*100)
# draw the label on the image
output = imutils.resize(orig, width=1200)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)