import sys
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Loads data from the specified location

    Inputs: 
        messages_filepath: filepath to the csv file with messages
        categories_filepath: filepath to the csv file with categories
    Output:
        df: dataframe of merged messages and categories 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    ''' Cleans data 

    Inputs: 
        df: merged dataframe as returned from load_data()
    Output:
        df: cleaned dataframe  
    '''
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]) 
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1]) 
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], sort=False, axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    ''' Saves data 

    Inputs: 
        df: cleaned dataframe as returned from clean_data()
        database_filename: specify the filename of the database to be saved (including subfolders if necessary)
                             e.g. data/DisasterResponse.db
    Output: None
    '''
    #specify database_filename e.g. data/DisasterResponse.db
    engine = create_engine(f'sqlite:///{database_filename}')
    sql = "DROP TABLE IF EXISTS DisasterResponse"
    engine.execute(sql)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    ''' Executes the processing steps - loading, cleaning, saving

    Inputs: None
    Output: saved sql database 
    '''
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