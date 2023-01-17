# import packages
import sys

# import libraries
import nltk
nltk.download(['words','punkt', 'wordnet'])

import pandas as pd

import pickle
import logging
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
    Turns the input text into a list of lemmatized tokens.

    Resulting tokens are lowercase with leading/trailing whitespace removed.

    :param text: Text to be tokenized
    :return: list of lemmatized tokens
    """

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def load_data(database_fp, database_table):
    """
    Loads table from sqlite database
    :param database_fp: filepath for the database
    :param database_table: table from the database
    :return: X- dataframe including train/test data, y-dataframe including prediction data
    """
    logging.info('Loading data from {} , table: {} .format(database_fp, database_table)')
    engine = create_engine('sqlite:///{}'.format(database_fp))
    df = pd.read_sql("SELECT * FROM {}".format(database_table), engine)
    X = df.message
    categories = df.columns[4:].tolist()
    y = df[categories]
    return X, y


def build_model():
    """
    build model pipline with GridSearch

    :return: Model pipline
    """
    # text processing and model pipeline

    logging.info('Building model pipline')

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators' : [10, 20, 30, 50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create gridsearch object
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def train_and_evaluate(X, y, model):
    """
    splits data into train and test, trains the model pipeline and evaluates the model on the test data

    :param X: Dataframe with data for the model
    :param y: Dataframe including predictions
    :param model: model pipline
    :return:
    """
    # train test split
    logging.info('Model training started')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # fit model
    model.fit(X_train, y_train)

    logging.info('Model has been trained')

    logging.info('Evaluating model')
    # predictions on test data set
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)

    # output model test results
    logging.info('Test data performance:')
    for column in y_test.columns:
        report = classification_report(y_test[column], y_pred[column])
        logging.info('Category: {}\n{}'.format(column, report))

    return model


def export_model(model, model_filepath):
    """
    saves the model in a pickle file

    :param model: trained model
    :param model_filepath: filipath where the model should be saved
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

    X, y = load_data(database_fp,database_table )  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train_and_evaluate(X, y, model)  # train model pipeline
    export_model(model, model_filepath)  # save model


if __name__ == '__main__':
    if len(sys.argv) == 4:
        database_fp = sys.argv[1]  # get filename of dataset
        database_table = sys.argv[2]
        model_fp = sys.argv[3]
        run_pipeline(database_fp, database_table, model_fp)  # run data pipeline
    else:
        logging.info('To run this program you need 3 Arguments: \ n Database filepath, table name and model filepath \n'
                     'Please add them to your command line. \n')