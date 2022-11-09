'''
Module to test the churn_library.py code.

Author: Moutasem Akkad
Date: 11/09/2022
'''

import os
import logging
import churn_library as cls

import pytest

from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w')

@pytest.fixture(scope="module")
def df_raw():
    """
    func: dataframe fixture 
    returns: dataframe
    """
    try:
        logging.info("TEST LOG: build df")
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: built df")
    except FileNotFoundError:
        logging.info("ERROR: cannot import file to build df")
        logging.error("ERROR: cannot import file to build df")
        raise FileNotFoundError
    return df
    
@pytest.fixture(scope="module")
def df_encoded(df_raw):
    """
    func: encoded dataframe fixture 
    returns: encoded dataframe
    """
    try:
        df_encoded = encoder_helper(df_raw,
                                    category_lst=["Gender",
                                                  "Education_Level",
                                                  "Marital_Status",
                                                  "Income_Category",
                                                  "Card_Category"], response="Churn")
        logging.info("SUCCESS: Encoded dataframe fixture creation")
    except KeyError as err:
        logging.info("ERROR: Encoded dataframe fixture creation: Not existent column to encode")
        logging.error(
            "ERROR: Encoded dataframe fixture creation: Not existent column to encode")
        raise err

    return df_encoded

@pytest.fixture(scope="module")
def encode_df(df_raw):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """
    lis= ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
    try:
        encoded_df = encoder_helper(df_raw, lis, 'Churn')
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: Not existent column to encode")
        raise err

    return encoded_df

def test_perform_feature_engineering(df_encoded):
    """
    func: test the perform_feature_engineering from churn_library.py
    """
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df_encoded)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info("SUCCESS: Testing perform_feature_engineering") 
    except BaseException:
        logging.error("ERROR: Testing perform_feature_engineering")
        logging.info("ERROR: Testing perform_feature_engineering")
        raise BaseException

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_raw):
    '''
    test perform eda function from churn_library.py
    '''
    perform_eda(df_raw)
    
    eda_images = ['barplot.png', 'distribution.png', 'heatmap.png', 'histogram.png', 'histplot.png']
    for image in eda_images:
        try:
            with open(f"images/eda/{image}", "r"):
                logging.info("SUCCESS:Testing perform_eda")
        except FileNotFoundError as err:
            logging.error("ERROR: Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper(encode_df):
    '''
    func: test encoder helper from churn_library.py
    returns: n/a
    '''
    df_encoded = encode_df
    
    try:
        for column in ["Gender_Churn",
                       "Education_Level_Churn",
                       "Marital_Status_Churn",
                       "Income_Category_Churn",
                       "Card_Category_Churn"]:
            assert isinstance(encode_df[column].iloc[0], float)
        logging.info("PASSED: test_encoder_helper_test")
    except AssertionError:
        logging.error(
            "FAILED: econder_helper_test")
        raise 


# def test_perform_feature_engineering(perform_feature_engineering):
#     '''
#     test perform_feature_engineering
#     '''


# def test_train_models(train_models):
#     '''
#     test train_models
#     '''


if __name__ == "__main__":
    pass








