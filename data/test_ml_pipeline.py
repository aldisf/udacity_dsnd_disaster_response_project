# import libraries
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

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


from sqlalchemy import create_engine



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
    

# load data from database
engine = create_engine('sqlite:///disasterpipeline.db')
df = pd.read_sql('SELECT * FROM messages', engine)
df = df.drop(['child_alone'], axis=1)
X = df['message']
y = df[df.columns[4:]]


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, max_features=500, ngram_range=(2,2)))
    ,('tf-idf',DenseTfIdf())
    ,('multi_output_clf', MultiOutputClassifier(estimator=LogisticRegression(), n_jobs=-1))  
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'vect__max_features':[200,500]
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted')

print('Performing 3-fold Cross Validation')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

model = grid_search.best_estimator_

y_preds = model.predict(X_test)

for index, col in enumerate(y.columns):

    predictions = y_preds[:, index]
    actual_label = y_test.values[:, index]

    show_results(col, predictions, actual_label)

joblib.dump(model, 'best_estimator')

