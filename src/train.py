from features.build_features import *
from models.train_model import *
from models.test_model import *
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from pickle import dump
import os


if __name__ == "__main__":
    
    imagePath = "../data/raw/Images"
    filePath = "../data/external"
    savePath = "../data/processed"
    # Extracting Features
    # features = extracting_features(imagePath)
    # Save the file
    # dump(features,open(os.path.join(savePath,"featues.pkl"),"wb")) 
    
    doc = load_document(os.path.join(filePath,"captions.txt"))
    # Parse Descriptions
    descriptions = load_descriptions(doc)
    clean_descriptions(descriptions)
    vocabulary = to_vocabulary(descriptions)
    save_descriptions(descriptions,os.path.join(savePath,"Descriptions.txt"))
    
    train = load_set(os.path.join(filePath,"trainImages.txt"))
    
    train_descriptions = load_clean_descriptions(os.path.join(savePath,"descriptions.txt"),train)
    
    train_features = load_image_features("../data/processed/featues.pkl",train)
    
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index)+1
    
    max_length = max_length(train_descriptions)
    
    # Train model 
    
    model = define_model(vocab_size,max_length)
    
    epochs = 10
    steps = len(train_descriptions)
    
    for i in range(epochs):
        
        generator = data_generator(train_descriptions,train_features,tokenizer,max_length,vocab_size)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save(os.path.join(savePath,"model_"+str(i)+'.h5'))
    
    # load test set
    filename = 'Flickr_8k.testImages.txt'
    test = load_set(os.path.join(filePath,filename))
    # descriptions
    test_descriptions = load_clean_descriptions(os.path.join(savePath,'descriptions.txt'), test)
    # photo features
    test_features = load_image_features(os.path.join(savePath,'features.pkl'), test)
    filename = os.path.join(savePath,'model_18.h5')
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)