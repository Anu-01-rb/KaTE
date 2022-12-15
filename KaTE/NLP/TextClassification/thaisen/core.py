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
    final = word_tokenize(final, engine= "deepcut")
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in stopwords)
    return final

def get_sentiment(text):
    lr = pickle.load(open("KaTE/NLP/TextClassification/thaisen/model/thaisen.model", 'rb'))
    cvec = pickle.load(open("KaTE/NLP/TextClassification/thaisen/model/vector.vec", 'rb'))
    tokens = text_process(text)
    bow = cvec.transform(pd.Series([tokens]))
    predictions = lr.predict(bow)
    return predictions[0]