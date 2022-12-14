# import libraries 
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
    messages_filepath - path to messages csv file
    categories_filepath - path to categories csv file
    
    OUTPUT:
    df - Merged data
    """
    
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    INPUT:
    df - Merged data
    
    OUTPUT:
    df - Cleaned data
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    # iterate through the category columns in df to keep only the
    # last character of the string
    # set each value to be the last character of the string
    # convert column from string to numeric
    # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    # drop duplicates
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)
    df = df.drop('child_alone', axis = 1)
    # replace 2s with 1s in related column
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
   
    
    return df
    
def save_data(df, database_filepath):
    """
        INPUT:
    df - cleaned data
    database_filename - database filename for sqlite database with (.db) file type
    
    OUTPUT:
    None - save cleaned data into sqlite database
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  


def main():
    """Loads data, cleans data, saves data to database"""
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
