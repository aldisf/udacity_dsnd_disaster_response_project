# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('disaster_messages.csv')
categories = pd.read_csv('disaster_categories.csv')

df = messages.merge(categories, on=['id'])

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

engine = create_engine('sqlite:///disasterpipeline.db')
df.to_sql('messages', engine, index=False)

