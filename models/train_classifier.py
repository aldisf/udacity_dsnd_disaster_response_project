import sys
import pandas as pd
import numpy as np
import re

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

from sklearn.metrics.classification import UndefinedMetricWarning

from sqlalchemy import create_engine

from disaster_response_utils import DenseTfIdf, show_results, tokenize

import warnings

def fxn():
    warnings.warn("future", FutureWarning)
    warnings.warn("undefinedMetric", UndefinedMetricWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def load_data(database_filepath):
    '''
    This function will read from a specified SQL database path with
    a table named 'message' inside. The table should be generated by the ETL
    part of this project.

    Input
        -db_path = Path to the SQL database file to be read by the script

    Output
        -X = Messages to be parsed as features in the ML pipeline
        -y = List of news labels to act as the classification target
        -category_names = Label names, corresponds to y.columns
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    query = 'SELECT * FROM messages'

    df = pd.read_sql(query, engine)
    X = df['message']
    y = df[df.columns[4:]]
    
    category_names = y.columns

    return X, y, category_names


def build_model(X_train, y_train):
    '''
    Given the training set, this function will build a preprocessing+classifier 
    pipeline. Hyperparameter tuning is done via GridSearchCV(3 folds). 

    The model performing best on the validation folds (based on weighted f-1 score) 
    will then be returned. 

    Input
        -X_train, y_train = Training data
    Output
        -model = Model with the best performance over the validation folds
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize))
        ,('tf-idf',DenseTfIdf())
        ,('multi_output_clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))  
    ])

    # Parameter grid for GridSearch to iterate 

    param_grid = {
        'vect__max_features': [200,300,500],
        'vect__ngram_range': [(1,3)],
        'multi_output_clf__estimator__n_estimators':[10,50,100]
        }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='recall_weighted')

    print('Performing 3-fold Cross Validation')
    print('Training model...')
    grid_search.fit(X_train, y_train)

    print('Best Parameters found:')
    print(grid_search.best_params_)

    model = grid_search.best_estimator_

    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function will perform evaluation on each category of news
    by iterating through the categories in category_names, and subsequently
    checking the accuracy, precision, recall and f1-score in the corresponding 
    category. This is done via show_results function

    Input
        -model = Model to be evaluated
        -X_test, y_test = Testing data
        -category_names = List of news categories from the raw data

    Output
        None 
    '''


    y_preds = model.predict(X_test)

    for index, col in enumerate(category_names):

        predictions = y_preds[:, index]
        actual_label = y_test.values[:, index]

        show_results(col, predictions, actual_label)

def save_model(model, model_filepath):
    '''
    To save the model at the specified filepath

    Input
        -model = model (Python object) to be saved
        -model_filepath = Path for saving the model

    Output 
        None
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, y_train)
    
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()