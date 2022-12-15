from pythainlp import word_tokenize
import wordcloud
import matplotlib.pyplot
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pythainlp.corpus.common import thai_stopwords
import pandas as pd

def text_process(text):
    stopwords = list(thai_stopwords())
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final, engine= "attacut")
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in stopwords)
    return final

def ev_text_recognizer(text):
    lr = pickle.load(open("KaTE/KaTE/NLP/TextClassification/evrec/model/evrecog.model", 'rb'))
    cvec = pickle.load(open("KaTE/KaTE/NLP/TextClassification/evrec/model/evrecog.vector", 'rb'))
    tokens = text_process(text)
    bow = cvec.transform(pd.Series([tokens]))
    predictions = lr.predict(bow)
    return predictions[0]