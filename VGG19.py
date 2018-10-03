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

class VGG19:
	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		model = Sequential([
			Conv2D(32, (5, 5), input_shape=inputShape, padding='same', activation='relu'),
			Conv2D(32, (5, 5), activation='relu', padding='same'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(64, (5, 5), activation='relu', padding='same'),
			Conv2D(64, (5, 5), activation='relu', padding='same',),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(64, (5, 5), activation='relu', padding='same',),
			Conv2D(64, (5, 5), activation='relu', padding='same',),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(128, (5, 5), activation='relu', padding='same',),
			Conv2D(128, (5, 5), activation='relu', padding='same',),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Conv2D(256, (5, 5), activation='relu', padding='same',),
			Conv2D(256, (5, 5), activation='relu', padding='same',),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
			Flatten(),
			Dense(1024, activation='relu'),
			Dense(1024, activation='relu'),
			Dense(classes, activation='softmax')
		])
		return model