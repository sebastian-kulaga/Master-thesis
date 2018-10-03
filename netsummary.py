from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])
print(model.summary())