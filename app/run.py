import sys

# add parent dir to python path so models and data modules are accessible by run.py script
sys.path.append('../')

import json
import joblib
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from models.train_classifier import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")



@app.route('/')
@app.route('/index')
def index():
    """
    extracts data needed for the visualisations,
    displays data visuals and receives user input text for model
    """

    # extract genres and get the number of messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract categories and get the number of messages per categorie
    categories = df.columns[4:].tolist()
    messages_per_category = df[categories].sum()

    # get the number of messages being posted in english
    df['original_english'] = df['message'] == df['original']
    english_message = df['original_english'].value_counts()
    labels = df['original_english'].unique()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories,
                    y=messages_per_category
                )
            ],

            'layout': {
                'title': 'Distribution of Messages per Categorie',
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
                    x=labels,
                    y=english_message
                )
            ],

            'layout': {
                'title': 'Number of English Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Englisch Messages"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    handles user query and displays model results
    """

    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    runs the app
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()