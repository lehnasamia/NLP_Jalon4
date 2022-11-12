import pickle
import numpy as np
import pandas as pd
import numpy as np
import os
import nltk
import contractions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from pickle import *
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

'''Partie 1 : Préprocessing '''

tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed

'''nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])'''

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()
    
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'a')) # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'v')) # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'n')) # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'r')) # Lemmatise adverbs
        else:
            lemmatized_text_list.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatisation
    
    return " ".join(lemmatized_text_list)

def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])

def contraction_text(text):
    return contractions.fix(text)

negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"

def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i+1 for i in range(len(tokens)-1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx]= negative_prefix + tokens[idx]
    
    tokens = [token for i,token in enumerate(tokens) if i+1 not in negative_idx]
    
    return " ".join(tokens)

def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]
    
    return " ".join([word for word in text.split() if word not in english_stopwords])

def preprocess_text(text):
    
    # Tokenize review
    text = tokenize_text(text)
    
    # Lemmatize review
    text = lemmatize_text(text)
    
    # Normalize review
    text = normalize_text(text)
    
    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text

'''Partie 2 : Prédiction d'un sujet d'insatisfaction'''

'''Importer les fichiers pickles 'model' et 'vectoriseur

model_pickle = open ('modelEntraineLehna','rb')    
modelEntraine = load(model_pickle)


vectoriseur_pickle = open ('vectoriseurLehna','rb')
vectorizer = load(vectoriseur_pickle)

model_pickle = open('modelEntraineLehna','rb')
vectoriseur_pickle = open('vectoriseurLehna','rb')

with (model_pickle) as f:
    modelEntraineLehna =pickle.load(f)

with (vectoriseur_pickle) as p:
    vectoriseurLehna =pickle.load(p)
'''

file_name1 = open("modelEntraineLehna",'rb')
vectorizer = load(file_name1)

file_name2 = open("vectoriseurLehna","rb")
model_pred = load(file_name2)




def predict_topics(model, vectorizer, n_topics, text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < 0:
        text = preprocess_text(text)
        text = [text]
        vectorized = vectorizer.transform(text)
        topics_correlations = model.transform(vectorized)
        unsorted_topics_correlations = topics_correlations[0].copy()
        topics_correlations[0].sort()
        sorted = topics_correlations[0][::-1]
        print(sorted)
        topics = []
        for i in range(n_topics):
            corr_value = sorted[i]
            result = np.where(unsorted_topics_correlations == corr_value)[0]
            topics.append(topics1.get(result[0]))
        print(topics)
    else:
        return polarity
