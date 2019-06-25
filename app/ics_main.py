#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import json
import pickle

import keras
import numpy as np
import pandas as pd
import plotly
from nltk.corpus import wordnet
from flask import flash, request
from ics_data_clean import clean_data, clean_text_round1, process_data
from ics_train_classifier import train_
from plotly.graph_objs import Bar, Scatter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# pickle load important files
category_id_df = pd.read_pickle('../pickle/ics/factorize/category_id_df.pkl')
id_to_category = dict(category_id_df[['category_id', 'category']].values)

with open('../pickle/ics/stop_words.pickle', 'rb') as f:
    stop_words = pickle.load(f)
# training_database_table = pd.read_pickle('../pickle/ics/training_database_index.pkl')

# process list
process_list = pd.read_excel('../data/ics/processes.xlsx')

# load training data
df = pd.read_pickle('../pickle/ics/data_final.pkl')

# define vectorizers
tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words=stop_words)
tfidf.fit(df.content)
cv = CountVectorizer(stop_words=stop_words)

tfidf_s = TfidfVectorizer(analyzer='word', sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                          stop_words=stop_words)
tfidf_s.fit(df.subject)
cv_s = CountVectorizer(stop_words=stop_words)

subject_list = df[['subject']].drop_duplicates().sort_values('subject').values
subject_list = [item for sublist in subject_list for item in sublist]

ALLOWED_EXTENSIONS = set(['xlsx', 'csv'])


# graph function
def ics_graphs():
    # extract data needed for visuals
    training_database_acc = pd.read_pickle('../pickle/ics/training_database_acc.pkl')
    tr_dt = [x.date().strftime("%x") for x in training_database_acc.tail(8).date.tolist()]
    sub_cat_y = [i * 100 for i in training_database_acc.tail(8).sub_cat_accuracy.tolist()]
    main_y = [i * 100 for i in training_database_acc.tail(8).main_accuracy.tolist()]
    cat_names = category_id_df['category'].sort_values().tolist()
    pos_ratios = df.groupby('category').case_no.count().tolist()
    graphs = [
        {
            'data': [Scatter(x=tr_dt, y=sub_cat_y, name='Sub Cat', line=dict(color='rgb(205, 12, 24)', width=4)),
                     Scatter(x=tr_dt, y=main_y, name='Main', line=dict(color='rgb(22, 96, 167)', width=4))],

            'layout': {'title': 'Accuracy History', 'yaxis': {'title': "Accuracy"}, 'xaxis': {'title': "Train Date"},
                       'ylim': {'bottom': 0, 'top': 1}}
        },
        {
            'data': [Bar(x=cat_names, y=pos_ratios)],

            'layout': {'title': 'Distribution of data', 'yaxis': {'title': "Number of Data rows"},
                       'xaxis': {'title': ""}}
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return subject_list, ids, graphJSON


def get_cosine_sim(dataframe):
    vectors_tfidf = tfidf.transform(dataframe).toarray()
    vectors = [t for t in vectors_tfidf]
    return cosine_similarity(vectors)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def wordnet_syn(opi):
    def synonym(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                synonyms.append(lm.name())
        return list(dict.fromkeys(synonyms))

    sentence = []
    for i in opi.split():
        sentence = sentence + synonym(i)[:5]
    return str(' '.join(sentence))


def ics_go():
    keras.backend.clear_session()
    # load model
    model = pickle.load(open("../pickle/ics/models/model.pkl", "rb"))

    # save user input in query
    client_query = request.args.get('query', '')
    query = client_query
    query_subject = request.args.get('query_subject', '')

    ## testing
    # query = clean_data(query)
    # query = wordnet_syn(query)

    query = clean_data(query)
    query_subject = clean_text_round1(query_subject)

    feature_transform = tfidf.transform([query]).toarray()
    # if no subject is selected
    if query_subject == "none":
        subject_transform = tfidf_s.transform(['']).toarray()
    else:
        subject_transform = tfidf_s.transform([query_subject]).toarray()

    query_tfidf = np.hstack((feature_transform, subject_transform))
    # use model to predict classification for query
    classification_index = int(model.predict_classes(query_tfidf)[0].tolist())
    classification_label = id_to_category[classification_index]

    # use submodel to predict subclassification for query
    try:
        submodel = pickle.load(open("../pickle/ics/submodels/" + str(classification_index) + ".pkl", "rb"))
    except Exception as e:
        print(e)
        flash('Model for sub-classification not trained. Ask administrator to retrain the model.')
        subclassification_label = -1

        processes = process_list.loc[process_list['categ'] == classification_label]['link']
        if processes.empty and subclassification_label == -1:
            processes = pd.Series(['https://checkvist.com/checklists/682361-ics-troublehshooting-charter'])
    else:
        df_cv = df[df['category_id'].isin([classification_index])]
        cv.fit(df_cv[df_cv['case_no'] != 'key'].content)
        cv_s.fit(df_cv[df_cv['case_no'] != 'key'].subject)
        feature_transform_cv = cv.transform([query]).toarray()
        subject_transform_cv = cv_s.transform([query_subject]).toarray()
        query_cv = np.hstack((feature_transform_cv, subject_transform_cv))

        # use model to predict sub-classification for query
        try:
            subclassification_index = int(submodel.predict(query_cv))
        except Exception as e:
            print(e)
            flash('Inconsistent features for sub-classification. Contact administrator')
            subclassification_label = -1
        else:
            subcategory_id_df = pd.read_pickle(
                "../pickle/ics/factorize/subcategory_id_df_" + str(classification_index) + ".pkl")
            id_to_subcategory = dict(subcategory_id_df[['subcategory_id', 'subcategory']].values)
            subclassification_label = id_to_subcategory[subclassification_index]

        processes = process_list.loc[process_list['categ'] == subclassification_label]['link']
        if processes.empty:
            processes = process_list.loc[process_list['categ'] == classification_label]['link']
            if processes.empty:  # and subclassification_label == -1):
                processes = pd.Series(['https://checkvist.com/checklists/682361-ics-troublehshooting-charter#'])

    if subclassification_label != -1:
        # related queries
        df_rel = pd.DataFrame(np.array([[query, 0]]), columns=['content', 'case_no']).append(
            df.loc[df['subcategory'] == subclassification_label], sort=False, ignore_index=True)
    else:
        # related queries
        df_rel = pd.DataFrame(np.array([[query, 0]]), columns=['content', 'case_no']).append(
            df.loc[df['category'] == classification_label], sort=False, ignore_index=True)

    a_temp = get_cosine_sim(df_rel[df_rel['case_no'] != 'key'].content.tolist())
    related_queries = []
    for sd in (np.argsort(a_temp[0].tolist())[-4:-1]):
        related_queries = related_queries + [df['case_no'].iloc[sd]]
    related_queries.reverse()
    related_queries_link = df.loc[df['case_no'].isin(related_queries)]['case_activity']
    related_queries = dict(zip(related_queries, related_queries_link))
    print(related_queries)
    return client_query, classification_label, subclassification_label, related_queries, processes, subject_list


def save_data(file):
    # clean and save dataframe
    try:
        # save data
        process_data(file)
        # train
        train_()

    except Exception as e:
        print(e)
        return -1
