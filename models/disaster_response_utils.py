import pandas as pd
import numpy as np
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin




class DenseTfIdf(BaseEstimator, TransformerMixin):
    '''
    Custom transformer that appends a .toarray()
    '''
    def __init__(self, smooth_idf=True):
        self.smooth_idf = smooth_idf
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return TfidfTransformer(smooth_idf=self.smooth_idf).fit_transform(X).toarray()


def show_results(column, predictions, actual_label):
    '''
    Input
        -column = Column/category name to be assessed
        -predictions = Predicted labels from classifier (Multioutput)
        -actual_label = Actual labeled data
    Output 
        -None
        
    This function, given the column name and the multioutput predictions & label, 
    will print out the accuracy, precision, recall and f1 score for the particular column
    
    '''
    print('Results for ', column)
    accuracy = accuracy_score(actual_label, predictions)
    precision = precision_score(actual_label, predictions)
    recall = recall_score(actual_label, predictions)
    f1 = f1_score(actual_label, predictions)
    print('Accuracy Score: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1-Score: {:.2f}\n'.format(accuracy, precision, recall, f1))
    

def tokenize(text):
    '''
    Input:
        - text = Text data (responses from disasters)
        - stop_words = List of stop words from the text language (English in this exercise)
        - lemmatizer = Lemmatizer object from nltk library to reduce words to their stem forms
    Ouput: 
        - A list of word tokens with preprocessings. Preprocessings done: 
            1. Punctuations removed (only alphanumeric characters left)
            2. Forced lowercase
            3. Leading/trailing space removed
            4. Exclude english stopwords
            5. Lemmatized to its stem word
    '''
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    alphanumeric_pattern = re.compile('[^a-zA-Z0-9 ]')
    alphanumeric_text = re.sub(alphanumeric_pattern, ' ', text)
    alphanumeric_text_lower = alphanumeric_text.lower()
    
    tokens = word_tokenize(alphanumeric_text_lower)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens