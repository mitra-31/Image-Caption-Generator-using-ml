from pickle import load
from keras.models import load_model
from predictions.feature_extraction import *
from predictions.generate_description import *



def run_model():
    tokenizer = load(open('predictions/tokenizer.pkl', 'rb'))
    max_length = 34
    model = load_model('predictions/model_18.h5')
    photo = extract_features('static/test.jpeg')
    description = generate_desc(model, tokenizer, photo, max_length)
    query = description
    stopwords = ['startseq','endseq']
    querywords = query.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result