# import packages
import sys
import pickle
import logging

# import libraries
import nltk
nltk.download(['words', 'punkt', 'wordnet'])

import pandas as pd

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# define logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def tokenize(text):
    """
    Turns the input text into a list of lowercase, lemmatized tokens

    :param text: text to be tokenized
    :return: list of lemmatized tokens
    """

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    # iterate through each token
    for token in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_token= lemmatizer.lemmatize(token.lower().strip())
        clean_tokens.append(clean_token)

    return clean_tokens


def load_data(database_fp, database_table):
    """
    Loads table from sqlite database into dataframe.

    :param database_fp: filepath of the sqlite database
    :param database_table: database table name
    :return: a tuple (X, Y) X: dataframe containing messages, Y: dataframe containing categories
    """
    logging.info('Loading data from {} , table: {} .format(database_fp, database_table)')
    engine = create_engine('sqlite:///{}'.format(database_fp))
    df = pd.read_sql("SELECT * FROM {}".format(database_table), engine)
    X = df.message

    # categories start from column 5 on
    categories = df.columns[4:].tolist()
    Y = df[categories]
    return X, Y


def build_model_pipeline():
    """
    Build model pipeline.

    :return: model pipeline
    """
    logging.info('Building model pipline')
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def create_grid_search(pipeline):
    """
    Create GridSearchCV object with predefined parameters.
    :param pipeline: the model pipeline to perform the search for
    :return: the grid search object
    """
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 30, 50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    return GridSearchCV(pipeline, param_grid=parameters)


def train_model(X_train, y_train, grid_search):
    """
    Trains a model using the supplied train data.

    :param X_train: the X train data
    :param y_train: the y train data
    :param grid_search: a GridSearchCV object configured with a model pipeline and search parameters
    :return:
    """
    logging.info('Model training started')
    model = grid_search.fit(X_train, y_train)
    logging.info('Model has been trained')
    return model


def evaluate_model(X_test, y_test, model, evaluation_report_fp):
    """
    Evaluates model predictions using test data prints the classification reports on each category
    and saves them into a file

    :param X_test: the X test data
    :param y_test: the y test data
    :param model: the model
    :param evaluation_report_fp: filepath for saving the evaluation report
    """
    logging.info('Evaluating model')
    # predictions on test data set
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)


    # output and save classification reports per category
    logging.info('Test data performance:')
    report_file = open(evaluation_report_fp, 'w')
    for column in y_test.columns:
        report = classification_report(y_test[column], y_pred[column], zero_division=0)
        logging.info('Category: {}\n{}'.format(column, report))
        report_file.write('Category: {}\n{}'.format(column, report))
    report_file.close()

def export_model(model, model_filepath):
    """
    Saves the model in a pickle file.

    :param model: trained model
    :param model_filepath: filepath where the model should be saved
    """
    logging.info('Saving model to {}'.format(model_filepath))
    pickle.dump(model, open(model_filepath, 'wb'))


def run_pipeline(database_fp, database_table, model_filepath):
    """
    runs a machine learning pipline: loads the data, builds a model pipline
    trains the model and saves it into a pickle file

    :param database_fp: filepath to the database where the data is saved
    :param database_table: The name of the table, where the data is saved
    :param model_filepath: Filepath where the trained model should be saved
    :return:
    """

    X, y = load_data(database_fp, database_table)              # load data from database
    model_pipeline = build_model_pipeline()                    # build model pipeline
    grid_search = create_grid_search(model_pipeline)           # create grid search to optimize parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # Split input data in to training and test set
    model = train_model(X_train, y_train, grid_search)         # train model pipeline

    evaluation_report_fp = "evaluation_report.txt"
    evaluate_model(X_test, y_test, model, evaluation_report_fp)   # print and save model evaluation reports
    export_model(model, model_filepath)                        # save model


if __name__ == '__main__':
    if len(sys.argv) == 4:
        database_fp = sys.argv[1]  # get filename of database
        database_table = sys.argv[2]  # get table name
        model_fp = sys.argv[3]  # get filepath for saving the trained model
        run_pipeline(database_fp, database_table, model_fp)  # run data pipeline
    else:
        logging.info('To run this program you need 3 additional arguments: \n '
                     'database filepath, table name and model filepath \n'
                     'Please add them to your command line.'
                     'Example: python train_classifier.py ../data/DisasterResponse.db disaster_messages classifier.pkl')
