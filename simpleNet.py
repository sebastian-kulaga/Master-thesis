# import the necessary packages
import plaidml.keras
plaidml.keras.install_backend()
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K

class simpleNet:
	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		model = Sequential([
			Flatten(input_shape=inputShape),
			Dense(32, activation='relu'),
			Dense(64, activation='relu'),
			Dense(128, activation='relu'),
			Dense(classes, activation='softmax')
		])
		return model