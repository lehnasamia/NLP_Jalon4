import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import contractions

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
'''
model_pickle = open('modelEntraineLehna','rb')
vectoriseur_pickle = open('vectoriseurLehna','rb')

with (model_pickle) as f:
    modelEntraineLehna =pickle.load(f)

with (vectoriseur_pickle) as p:
    vectoriseurLehna =pickle.load(p)

def prediction(model,vectorizer,n_topic,new_reviews):
  blob=TextBlob(new_reviews)
  sentimentBlob=blob.sentiment.polarity
  new_reviews = preprocess_text(new_reviews)
  new_reviews = [new_reviews]
  new_reviews_transformed=vectorizer.transform(new_reviews)

  prediction= model.transform(new_reviews_transformed)
 
  topics=[ 'Cadre du lieu',
           'Plats en sauce',
           'Menu pizza ',
           'Service livraison et commandes',
           'Qualité des plats ',
           'Qualité du service',
           'Menu burger',
           'Temps attente',
           'Menu chicken',
           'Service bar ',
           'Localisation du lieu',
           'Relation client',
           'Menu sandwich',
           'Menu sushis',
           'Clients revenus']

  if sentimentBlob<0 and sentimentBlob>-1:
    max = np.argsort(prediction)
    max_list=(list(max[0]))
    max_list.reverse()
    print(max_list)
    topic=[]
    for i in range(n_topic):
      topic.append(topics[max_list[i]])  
    return sentimentBlob,prediction,topic

  return sentimentBlob
