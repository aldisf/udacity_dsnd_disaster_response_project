import sys
import pandas as pd
import numpy as np
from sqlalchemy import *


def load_data(messages_filepath, categories_filepath):
    '''
    Reads two csv files for messages and categories, 
    merges them together and return the merged dataframe

    Input
        -messages_filepath, categories_filepath = Filepaths for csv files to be read and merged

    Output
        -df = Merged messages and categories dataframe

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    '''
    Perform label extraction and cleaning to the loaded 
    dataframe. 

    Originally, categories are in the form of: {category_name}-{1 or 0}
    They will then be split to 36 binary columns. 

    Input
        -df = Loaded data frame from load_data function

    Output
        -df = Message and categories dataframe with the categories extracted into 36 binary columns

    '''
    categories = df['categories'].str.split(';', expand=True)

    row = categories.loc[0]

    category_colnames = [i.split('-')[0] for i in row.values.tolist()]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]))
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # related column has category = 2
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df[df['related'] != 2].drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Function to save the cleaned and extracted data into
    a SQL database file.

    Input
        -df = Dataframe to be saved
    
    Output
        -database_filename = Path to the file where the dataframe will be saved as
            a SQL database file
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()