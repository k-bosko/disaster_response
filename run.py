import json
import plotly
import pandas as pd
import gzip

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
df_nochildalone = df.drop(['child_alone'], axis=1)

with gzip.open('../models/model.p.gz', 'rb') as f:
    # load model
    model = joblib.load(f)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    category_names = list(df.columns.values)[4:]
    category_counts = df.iloc[:,4:].sum().values
    
    rowSums = df.iloc[:,4:].sum(axis=1)
    multiCategory_counts = rowSums.value_counts()

    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Number of Messages in each Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=multiCategory_counts.index,
                    y=multiCategory_counts.values,
                    marker=dict(color='rgb(255, 140, 0)')
                )
            ],

            'layout': {
                'title': 'Distribution of Messages with Multiple Categories',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Multiple Categories",
                    'dtick': 1
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_nochildalone.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()