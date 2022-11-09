'''
Module to read, train and show statistics about the churn data

Author: Moutasem Akkad
Date: 11/09/2022
'''

# import libraries
import logging
import os

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scikitplot.metrics import plot_roc


import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w")


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/eda/histogram.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/distribution.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/barplot.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/histplot.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # gender encoded column
    for category in category_lst:
        category_lst = list()
        category_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            category_lst.append(category_groups.loc[val])
        if response:
            df[category + "_" + response] = category_lst
        else:
            df[category] = category_lst
        df.drop(category, axis=1, inplace=True)
        category_lst.clear()
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[response]
    X = df.drop(response, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure("figure", figsize=(5, 5))
    cs_report = classification_report(y_train, y_train_preds_lr)
    plt.text(0.0121, 0.6, cs_report)
    plt.savefig("./images/results/train_lr.png")
    plt.close()

    plt.figure("figure", figsize=(5, 5))
    cs_report = classification_report(y_train, y_train_preds_rf)
    plt.text(0.0121, 0.6, cs_report)
    plt.savefig("./images/results/train_rf.png")
    plt.close()

    plt.figure("figure", figsize=(5, 5))
    cs_report = classification_report(y_test, y_test_preds_lr)
    plt.text(0.0121, 0.6, cs_report)
    plt.savefig("./images/results/test_lr.png")
    plt.close()

    plt.figure("figure", figsize=(5, 5))
    cs_report = classification_report(y_test, y_test_preds_rf)
    plt.text(0.0121, 0.6, cs_report)
    plt.savefig("./images/results/test_rf.png")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(f"images/{output_pth}/Feature_Importance.jpg")
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, X_test, "results")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
        # store ROC curves plot
    lrc_plot = plot_roc(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc(cv_rfc.best_estimator_,
                       X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig("images/results/Roc_Curves.jpg")
    plt.close()

    # store classification report image
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # store feature importance plot
    feature_importance_plot(cv_rfc, X_test, "results")


if __name__ == '__main__':
    df = import_data(r"./data/bank_data.csv")
    logging.info("SUCCESS: read dataframe.")

    perform_eda(df)
    logging.info("SUCCESS: saved EDA plots to the images folder.")

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Attrition_Flag'
    ]

    encoded_df = encoder_helper(df, cat_columns, 'Churn')
    logging.info(
        "SUCCESS: Encoded categorial columns and dropped the string columns.")

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_df, 'Churn')
    logging.info("SUCCESS: splitted data intro train and test")

    train_models(X_train, X_test, y_train, y_test)

    logging.info(
        "SUCCESS: Training and predeiction done.")


#     # grid search
#     rfc = RandomForestClassifier(random_state=42)
#     # Use a different solver if the default 'lbfgs' fails to converge
#     # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#     lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

#     param_grid = {
#         'n_estimators': [200, 500],
#         'max_features': ['auto', 'sqrt'],
#         'max_depth' : [4,5,100],
#         'criterion' :['gini', 'entropy']
#     }

#     cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
#     cv_rfc.fit(X_train, y_train)

#     lrc.fit(X_train, y_train)

#     y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
#     y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

#     y_train_preds_lr = lrc.predict(X_train)
#     y_test_preds_lr = lrc.predict(X_test)

#     logging.info(
#         "SUCCESS: Training and predeiction done.")

#     classification_report_image(y_train,
#                                 y_test,
#                                 y_train_preds_lr,
#                                 y_train_preds_rf,
#                                 y_test_preds_lr,
#                                 y_test_preds_rf)
