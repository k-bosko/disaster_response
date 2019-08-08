import sys
import warnings
import pickle
import gzip
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier



warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(database_filepath):
    ''' Loads data 

        Inputs: 
            database_filepath: specify the filepath to the saved sql database (e.g. data/DisasterResponse.db)
        Output:
            X: messages 
            Y: 36 categories
            category_names: list of the categories' names
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    #dropping child_alone category that has no 1s. Otherwise won't be able to use SGDClassifier
    df_nochildalone = df.drop(['child_alone'], axis=1)
    X = df_nochildalone['message']
    Y = df_nochildalone.iloc[:, 4:]
    #X = df['message']
    #Y = df.iloc[:, 4:]
    #category_names = df.columns[4:]
    category_names = df_nochildalone.columns[4:]
    return X, Y, category_names


def tokenize(text):
    ''' Tokenizer for CountVectorizer() 

        Inputs: 
            text: message instance
        Output: 
            clean_tokens: list of lemmatized tokens based on words from the message
    ''' 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' Builds model as pipeline 

        Inputs: 
            None
        Output: 
            model: pipeline consisting of nlp steps and final estimator with multioutput wrapper
    '''
    #specifying the best parameters that were determined in the gridsearch steps (see ML Pipeline Preparation.ipynb)
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, ngram_range=(1, 2), stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(random_state=42)))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Predicts on houldout data and prints out classification report for each category  

        Inputs: 
            model: model as specified in build_model()
            X_test: holdout data with messages
            Y_test: holdout data with categories
            category_names: list of categories' names
        Output: None
            
    '''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(i, col)
        print(classification_report(Y_test.to_numpy()[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    ''' Saves model as gzip pickle object 

        Inputs: 
            model: model as specified in build_model()
            model_filepath: speficy model filepath (with subfolders if necessary), e.g. "models/model.p.gz"
        Output: None
            
    '''
    with gzip.open(model_filepath, 'wb') as gzipped_f:
    # Pickle the trained pipeline and save as gzip.
        pickled = pickle.dumps(model)
        gzipped_f.write(pickled)


def main():
    ''' Executes modeling steps - loading data from sql database, building model, training, eveluating and saving

        Inputs: None
        Output: None
            
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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