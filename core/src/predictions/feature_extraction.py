from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array


def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature