import pandas as pd
import re
import json
import string
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


contraction_json = open('helpers/contraction.json',)
contraction_map = json.load(contraction_json)
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def rep_func(prep_texts): 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(prep_texts)
    return X, vectorizer

def bal_data(df, classes): #used when dataset is not balanced
    g = df.groupby(classes)
    g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min(), random_state=1234).reset_index(drop=True)))
    return g

def contraction(word):
    for contraction, replacement in contraction_map.items():
        word = word.replace(contraction, replacement)
    return word

def preprocess(text):
    text = text.lower()
    #removables-1
    text = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)",'',text) #links
    text = re.sub('@[^\s]+','', text) #usernames
    #emojis are already removed in this dataset
    text = re.sub('#[^\s]+','', text) #hastags
    #text = re.sub('^a-z0-9<>', '', text) #remove non-alphanumeric symbols
    text = re.sub('[0-9]+', '', text) #numbers
    #transformations-1
    text = " ".join([contraction(word) for word in text.split()])
    #removables-2
    text = ''.join([c for c in text if c not in string.punctuation])#punctuations
    #transformations-2
    text = " ".join([stemmer.stem(word) for word in text.split()])
    text = " ".join([word for word in text.split() if word not in stop_words]) #" ".join(text), text is modified using list comprehension
    
    return text

def load_objects():
    file = open("vectorizer.pkl",'rb')
    vectorizer = pickle.load(file)
    file.close()
    file = open("expA.pkl",'rb')
    classifier = pickle.load(file)
    file.close()
    return vectorizer, classifier