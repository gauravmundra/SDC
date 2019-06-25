#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import os
import glob
import multiprocessing
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

n_jobs = multiprocessing.cpu_count() - 1
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('../pickle/ics/stop_words.pickle', 'rb') as f:
    stop_words = pickle.load(f)


def load_data(directory):
    """
    Load data from dataframes
    :param directory: dataframes directory
    :return: Dataframe
    """
    # load all the dataframes in directory
    df_list = glob.glob(directory + "/*.pkl")
    keys = pd.read_pickle('../pickle/ics/keys.pkl')
    df = pd.DataFrame()
    for pickl in df_list:
        pickl = pd.read_pickle(pickl)
        df = pd.concat([df, pickl], axis=0, sort=False, ignore_index=True)
    df = pd.concat([df, keys], axis=0, sort=False, ignore_index=True)
    return df


def factorize(df):
    df['category_id'] = df['category'].factorize()[0]
    category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_id_df.to_pickle("../pickle/ics/factorize/category_id_df.pkl")
    df_sub = df[df['case_no'] != 'key']
    # category_id_df for main classification and subcategory_id_df_{{i}} for subcategories
    for i in category_id_df['category_id']:
        df_i = df_sub.loc[df['category_id'] == i]
        df_i['subcategory_id'] = df_i['subcategory'].factorize()[0]
        subcategory_id_df = df_i[['subcategory', 'subcategory_id']].drop_duplicates().sort_values('subcategory_id')
        subcategory_id_df.to_pickle("../pickle/ics/factorize/subcategory_id_df_" + str(i) + ".pkl")
        df_i.to_pickle("../pickle/ics/categ_data/df_" + str(i) + ".pkl")
    df.to_pickle("../pickle/ics/data_final.pkl")
    return df


def build_model(df):
    """
    Build the model
    :return: The model accuracy
    """

    # vectorize
    print('Feature Engineering...')
    labels = np_utils.to_categorical(df.category_id)
    tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words=stop_words)
    features = tfidf.fit_transform(df.content).toarray()
    tfidf_s = TfidfVectorizer(analyzer='word', sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                              stop_words=stop_words)
    features_s = tfidf_s.fit_transform(df.subject).toarray()
    features_new = np.hstack((features, features_s))

    # Train Test Split
    print('Train Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(features_new, labels, stratify=labels, test_size=0.10)

    # build_model
    print('Building Model...')
    input_dim = X_train.shape[1]
    categories_n = labels.shape[1]

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.76))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(categories_n, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    print('Model Summary')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # fit_train
    model.fit(X_train, y_train, epochs=180, verbose=True, validation_data=(X_test, y_test), batch_size=120)

    # evaluate
    print('Evaluation...')
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.2f}".format(accuracy * 100))

    # save model
    pickle.dump(model, open('../pickle/ics/models/model.pkl', 'wb'))

    return accuracy


def build_submodel():
    """
    Build the submodels
    :return: The overall submodel accuracy
    """
    accuracy_o = 0
    j = 1
    category_id_df = pd.read_pickle('../pickle/ics/factorize/category_id_df.pkl')
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    for i in category_id_df['category_id']:
        try:
            df = pd.read_pickle("../pickle/ics/categ_data/df_" + str(i) + ".pkl")

            print('\n\n Next Sub Category')
            print(i, ": ", id_to_category[int(i)])

            # vectorize
            print('Feature Engineering...')
            labels = df.subcategory_id
            cv = CountVectorizer(stop_words=stop_words)
            features = cv.fit_transform(df.content).toarray()
            cv_s = CountVectorizer(stop_words=stop_words)
            features_s = cv_s.fit_transform(df.subject).toarray()
            # multiplier to lower effect of subject on results
            features_new = np.hstack((features, features_s))

            # Train Test Split
            print('Train Test Split...')
            X_train, X_test, y_train, y_test = train_test_split(features_new, labels, stratify=labels, test_size=0.10)

            # Oversampling
            # adas = ADASYN(n_neighbors=5, n_jobs=n_jobs)
            # sm = SMOTE(n_jobs=n_jobs)
            # X_train, y_train = sm.fit_sample(X_train, y_train)

            # build_model
            print('Building Model...')

            model = xgboost.XGBClassifier(n_jobs=n_jobs, multi="softmax")

            # fit_train
            model.fit(X_train, y_train)

            # evaluate
            print('Evaluation...')
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Testing Accuracy", i, ":  {:.2f}".format(round(accuracy * 100, 2)))
            accuracy_o = accuracy_o + accuracy

            # save model
            pickle.dump(model, open('../pickle/ics/submodels/' + str(i) + '.pkl', 'wb'))
            j = j + 1
        except Exception as e:
            print(e)
    accuracy_o = accuracy_o / j
    return accuracy_o


def train_(directory=r'..\pickle\ics\dataframes'):
    print('Loading dataframes...')
    df = load_data(directory)

    print('Factorizing...')
    df = factorize(df)

    print('Building and Evaluating model...')
    main_accuracy = build_model(df)

    print('Building and Evaluating submodel...')
    sub_accuracy = build_submodel()

    print('Main Accuracy: ', main_accuracy, ', Sub-Cat Accuracy: ', sub_accuracy)
    print('Trained model saved!')

    # update training database table
    training_database_index = pd.read_pickle('../pickle/ics/training_database_index.pkl')
    training_database_acc = pd.read_pickle('../pickle/ics/training_database_acc.pkl')
    try:
        max_id_ = int(training_database_acc['id'].max())
    except:
        max_id_ = 0
    training_database_acc = training_database_acc.append(
        {'id': max_id_ + 1, 'date': datetime.now(),
         'files': str(', '.join(str(v) for v in training_database_index.file_name.tolist())),
         'main_accuracy': main_accuracy,
         'sub_cat_accuracy': sub_accuracy}, ignore_index=True)
    training_database_acc.to_pickle("../pickle/ics/training_database_acc.pkl")

    return main_accuracy, sub_accuracy


def main():
    print('Manual Training Start...')
    train_()


if __name__ == '__main__':
    main()
