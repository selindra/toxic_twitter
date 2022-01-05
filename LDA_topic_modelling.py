#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 16:37:35 2021

@author: selin
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from pprint import pprint
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm_notebook as tqdm
import logging
import warnings
import seaborn as sns
from plotly import graph_objs as go

#enter you working directory here
path = ' '


# Any results you write to the current directory are saved as output.
df = pd.read_excel( path +"all_tweets.xlsx")
df = df.drop(columns=['username','id','permalink',])
df = df.dropna()
df['toxicity'] = df['toxicity'].astype(int)


df = df[(df['toxicity'] == 0)|(df['toxicity'] == 1)|(df['toxicity'] == 2)]

#clean data from tech symbols
def clean_message(msg: str):
  msg = re.sub(r'http\S+','',msg)
  return msg.replace("\n", " ").lower()
df['text'] = df['text'].apply(clean_message)

#tweets tokenization
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 200
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['toxicity']).values
print('Shape of label tensor:', Y.shape)

#downloading the stopwords list
nltk.download('stopwords')

# Enable logging for gensim - optional
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore",category=DeprecationWarning)

#here we take only one group of tweets (here 1=social group) to divide it for topics
df_toxic = df[df['toxicity']==1]
#creating pipeline
nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words.update(['toxic', 'Toxic','█', 
                                '\n','\n ','\n  ', '\n   ' '\n\n', "\n\n ", 
                                "  ", "|", '$',
                                'u', 'ur', 'people'
                                ])
# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True
    
def lemmatizer(doc):
    # This takes in a doc of tokens from the NER and lemmatizes them. 
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so it should be removed
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    doc = u' '.join(doc)
    return nlp.make_doc(doc)
    
def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc

# The add_pipe function appends our functions to the default pipeline.
nlp.add_pipe(lemmatizer,name='lemmatizer',after='ner')
nlp.add_pipe(remove_stopwords, name="stopwords", last=True)
doc_list = []
# Iterates through batches of articles article in the corpus.
for doc in tqdm(nlp.pipe(df_toxic.text),desc='tqdm'):
    doc_list.append(doc)
    
# Creates DICTIONARY which is a mapping of word IDs to words.
words = corpora.Dictionary(doc_list)

# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in doc_list]
print(corpus[:1])

#Finding optimal n of topics for LDA

def compute_gen_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):
        model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=words,
                                           workers=3,
                                           num_topics=num_topics, 
                                           random_state=2,
                                           passes=10,
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
    
# Can take a long time to run.
limit=30; start=8; step=3;
gen_model_list, gen_coherence_values = compute_gen_coherence_values(
                                                                    dictionary=words, 
                                                                    corpus=corpus, 
                                                                    texts=doc_list, 
                                                                    start=start, 
                                                                    limit=limit, 
                                                                    step=step
    )

# Show graph
x = range(start, limit, step)
plt.plot(x, gen_coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

#The recommended amount of topics will have the highest coherence value
for m, cv in zip(x, gen_coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
n_of_topics=14 #change according to desired amount of topics   
lda_multi_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=words,
                                           workers=3,
                                           num_topics=n_of_topics, 
                                           random_state=2,
                                           passes=10,
                                           per_word_topics=True)

# Compute Perplexity
print('\nPerplexity: ', lda_multi_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_multi_model, texts=doc_list, dictionary=words, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)    

# Select the model and print the topics
model_name = " " # CHANAGE THIS EVERY TIME
optimal_model = lda_multi_model
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

#Finding the dominant topic in each sentence
a = optimal_model[corpus[0]]    
topics_words = pd.DataFrame(optimal_model.show_topics(), columns=['Topic','Words'])

def format_topics_sentences(ldamodel, corpus=corpus, texts=doc_list):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution']
    return(sent_topics_df)

df_topics = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=doc_list)

#View topic distribution over the year
sns.set(rc={'figure.figsize':(16, 10)})

df_topics['date'] = df_toxic["date"].values
df_topics['Topic'] = df_topics['Dominant_Topic'].apply(lambda x: int(x))
df_topics = df_topics.drop('Dominant_Topic', axis=1)
df_topics = df_topics.drop('Perc_Contribution', axis=1)
    
#Amount of tweets in every group
sns.countplot(x='Topic', data=df_topics)

# Plot for one topic on the time scale
topic = 3 # Change to desired topic
df_cat1 = df_topics[df_topics['Topic']==topic].groupby('date').count()
fig = go.Figure(data=[go.Scatter(x=df_cat1.index, y=df_cat1['Topic'])])
fig.show()

# Plot all topics on time scale
fig = go.Figure()
for i in range(n_of_topics):
  df = df_cat1 = df_topics[df_topics['Topic']==i].groupby('date').count()
  fig.add_trace(go.Scatter(x=df.index, y=df['Topic'], name=f"Topic {i}"))
fig.layout.update(title='Time Series with Rangeslider')
fig.show()
