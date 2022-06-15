from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from keras.layers.merge import add
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import array
from pickle import load
import string
import os

def load_document(filename=""):
    
    if not os.path.exists(filename):
        return None

    current_file = open(filename,"r")
    text = current_file.read()
    current_file.close()
    return text


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def load_descriptions(file):
    ''' Extracting Description for an Image 

        keyword arguments:
        arguments -- .pkl file 
        Return: return dict of image and description
    
    '''
    
    mapping = dict()
    for line in file.split('\n'):
        tokens = line.split()
        
        # Empty Spaces 
        if len(line) < 2:
            continue
        
        # Break down tokens into image file name and description of image. 
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        
        # create a new key if it doesn't exist in the dictionary
        if image_id not in mapping:
            mapping[image_id] = []
        
        mapping[image_id].append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    ''' 
    	Prepare translation table for removing punctuation

        keyword arguments:
        arguments -- .pkl file 
        Return: return dict of image and description
    
    '''
    
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc) 


def to_vocabulary(descriptions):
    '''
        Convert the loaded descriptions into vocabulary of words
    
        keyword arguments:
        arguments -- descriptions  
        Return: return unique descriptions.

    '''
    
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]        
    return all_desc


def save_descriptions(descriptions,filename):
    lines = list()
    
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + " " + desc)
    data = "\n".join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()
    

def load_set(filename):
    doc = load_document(filename)
    datasets = []
    
    for line in doc.split("\n"):
        if len(line) < 1: continue
        identifier = line.split(".")[0]
        datasets.append(identifier)
    return set(datasets)

def load_clean_descriptions(filename, datasets):
    
    doc = load_document(filename)
    descriptions = dict()
    for line in doc.split("\n"):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in datasets:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = 'starseq ' + ' '.join(image_desc) + " endseq"
            descriptions[image_id].append(desc)
    return descriptions


def load_image_features(filename, datasets):
    all_features = load(open(filename, "rb"))
    filtered_features = {k: all_features[k] for k in datasets}
    
    return filtered_features





# filename = "datasets/Text_data/trainImages.txt"

def to_lines(description):
    
    all_desc = []
    for key in description.keys():
        [all_desc.append(d) for d in description[key]]
    return all_desc


def create_tokenizer(description):
    lines = to_lines(description)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo,vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)


# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	return model

def data_generator(descriptions, photos, tokenizer, max_length,vocab_size):
    	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo,vocab_size)
			yield [[in_img, in_seq], out_word]