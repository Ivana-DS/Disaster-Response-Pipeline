# import libraries
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine

# define logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_data(filepath_1, filepath_2, index = None):
    """
    loads data from 2 csv files into separate pandas data frames and
    merges the data frames into one dataframe on index

    :param filepath_1: first data source
    :param filepath_2: second data source
    :param index: label or list of labels to join on, those must be found in both data frames
    :return: pandas data frame which contains the data from both data sources
    """
    dataset_1 = pd.read_csv(filepath_1)
    dataset_2 = pd.read_csv(filepath_2)

    # merge datasets
    df = pd.merge(dataset_1, dataset_2, how='outer', on=index)
    return df


def clean_data(df):
    """
    Takes a data frame as input and cleans the data

    :param df: data frame, which needs to be cleaned
    :return: cleaned data frame
    """

    # Split categories into separate category columns
    # create individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # extract a list of new column names for the categories.
    categories_names = categories.head(1).values
    categories_names = [x[:-2] for x in categories_names.tolist()[0]]

    # rename the columns of categories
    categories.columns = categories_names

    # Convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_df_to_sql(df, db_url, table_name):
    """
    Takes a dataframe as input, creates a data base and safes
    the data frame in a table in the data base

    :param df: data frame
    :param url: url for the data base
    :param table_name: name of the table
    """
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, index=False)


def main(argv):

    messages_fp = argv[0] # './disaster_messages.csv'
    categories_fp = argv[1] # './disaster_categories.csv'
    db_url = 'sqlite:///' + argv[2]
    table_name = 'disaster_messages'

    # load data
    logging.info('Loading data from files: {} {}'.format(messages_fp, categories_fp))
    df = load_data(messages_fp,categories_fp)

    # clean data
    logging.info('Cleaning data')
    df = clean_data(df)

    # save data to sql
    logging.info('Saving data in table: {}, data base url: {}'.format(table_name,db_url ))
    save_df_to_sql(df, db_url, table_name)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1:])
    else:
        logging.info('Please add the needed arguments to start this script. \n'
                     'Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')