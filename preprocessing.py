import numpy as np
import pandas as pd 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

class text_preprocessing():
  def __init__(self, column):
    self.column = column

    nltk.download('stopwords')
    nltk.download('wordnet')
    
    self.wordnet_lemmatizer = WordNetLemmatizer()

    self.stopwords = nltk.corpus.stopwords.words('english')
  
  def remove_punctuation(self, text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
  
  def tokenization(self, text):
    tokens = text.split()
    return tokens
  
  def remove_stopwords(self, text):
    output= [i for i in text if i not in self.stopwords]
    return output

  def lemmatizer(self, text):
    lemm_text = [self.wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
  
  def remove_numbers(self, text):
        nonum_text = [re.sub(r'[0-9]', '', word) for word in text]
        return nonum_text

  def process_f(self,x):
    a = x[self.column].lower()
    b = self.remove_punctuation(a)
    c = self.tokenization(b)
    d = self.remove_stopwords(c)
    e = self.lemmatizer(d)
    f = self.remove_numbers(e)
    
    return f
