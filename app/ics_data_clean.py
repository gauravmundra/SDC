#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import re
import os
import string
from datetime import datetime
import glob
import nltk
import shutil
import pandas as pd
from nltk.stem import WordNetLemmatizer

ALLOWED_EXTENSIONS = set(['xlsx', 'csv'])


def load_data(file):
    print('LOADING DATA...')
    try:
        try:
            df = pd.read_excel(file)
        except:
            df = pd.read_csv(file)
        df = df.sample(frac=1)
        # prepare excel file for data pre-processing
        df.rename(columns={'Root Cause': 'category'}, inplace=True)
        df.rename(columns={'Description': 'content'}, inplace=True)
        df.rename(columns={'Case Number': 'case_no'}, inplace=True)
        df.rename(columns={'Subject': 'subject'}, inplace=True)
        df.rename(columns={'Sub Category': 'subcategory'}, inplace=True)
        df.rename(columns={'Case Activity': 'case_activity'}, inplace=True)
        df = df[['subject', 'category', 'content', 'case_no', 'subcategory', 'case_activity']]
    except Exception as e:
        print(e)
        return -1

    df['subcategory'].replace('', 'temp_subcat', inplace=True)
    df['subcategory'].fillna('temp_subcat', inplace=True)
    df['case_activity'].replace('', 'https://ideas-sas.lightning.force.com/lightning', inplace=True)

    '''
        REMOVE, ONLY FOR TESTING, OPTIMIZE THE DATA TO REMOVE THIS CODE
    '''
    ##
    df['category'] = df['category'].replace({'Data Discrepancy': 'Data Discrepancy/Data Population Issue'})
    df['category'] = df['category'].replace({'Data Population Issue': 'Data Discrepancy/Data Population Issue'})
    df['category'] = df['category'].replace({'Client Configuration Issue': 'Client Training'})
    df['category'] = df['category'].replace({'Further education': 'Client Training'})
    df['category'] = df['category'].replace({'Concept/ Client Understanding': 'Client Training'})
    df['category'] = df['category'].replace({'User interface concern/ configuration': 'Configuration Issue'})
    df['category'] = df['category'].replace({'System Configuration Issue': 'Configuration Issue'})

    df = df[df['category'].isin(
        ['Configuration Issue', 'Client Training', 'Data Discrepancy/Data Population Issue', 'Decision Related Issue',
         'Defect', 'Performance Issue', 'Scheduled Report Issue', 'System Not Upto date', 'Webrate Issue'])]
    ##

    ##
    # Add code to remove category and sub-category having very low data rows ( min 100 in classification and min 5 in sub-classification)
    ##

    return df


def clean_text_round1(text):
    # Make text lowercase, remove leading spaces
    text = text.strip().lower()
    # remove commas
    text = re.sub(',', ' ', text)
    # remove emails
    text = re.sub('\w*@\w*.com\w*', ' ', text)

    # remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # remove text in square brackets
    text = re.sub('\[.*?\]', ' ', text)

    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # remove
    text = re.sub('\dst', ' ', text)
    text = re.sub('\dth', ' ', text)
    text = re.sub('\drd', ' ', text)
    # remove numbers
    text = re.sub('\d', ' ', text)

    # Get rid of some additional punctuation
    text = re.sub('[‘’“”…]', ' ', text)
    # remove next line '\n'
    text = re.sub('\n', ' ', text)

    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\/", " ", text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)

    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"up *to *date", "uptodate", text)

    # remove text after thank you and best regards
    text = re.sub(' thank\w* *you\w* *( \w*)*', ' ', text)
    text = re.sub(' best\w* *regard\w* *( \w*)*', ' ', text)

    # remove duplicate white spaces
    text = re.sub(' +', ' ', text)

    return text


lem = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


def lemmatize_text(text):
    wd = ""
    for w in w_tokenizer.tokenize(text):
        wd = wd + " " + lem.lemmatize(w, "v")
    return wd


def clean_data(data):
    # just for cleaning queries
    print('CLEANING DATA...')
    data_clean = clean_text_round1(data)
    print('APPLYING LEMMATIZATION...')
    data_clean = lemmatize_text(data_clean)
    return data_clean


def process_data(file):
    # takes a file as input
    # save preprocessed file as pickle to dataframe folder
    try:
        df = load_data(file)
    except Exception as e:
        print(e)
        return -1
    print('CLEANING DATA...')
    df_temp = pd.DataFrame(df.content.apply(clean_text_round1))
    df.rename(columns={'content': 'content_old'}, inplace=True)
    df = pd.concat([df, df_temp], axis=1)
    df = df[df['content'].map(len) > 13]

    def apply_lemm(df_lemm):
        print('APPLYING LEMMATIZATION...')
        df_lemm['lemm_content'] = df.content.apply(lemmatize_text)
        df_lemm.drop(columns=['content'], inplace=True)
        df_lemm.rename(columns={'lemm_content': 'content'}, inplace=True)
        return df_lemm

    df = apply_lemm(df)
    print('SAVING DATA...')
    if type(file) == str:
        #  process_data(string) when called by app
        dataframe = file[20:-5]  # 20 to remove directory name and -5 to remove '.xlsx'
    else:
        #  process_data(file) when called by app
        dataframe = file.filename
    df.to_pickle("../pickle/ics/dataframes/" + dataframe + ".pkl")
    print('Clean Data Saved!')

    # update training database table
    training_database_index = pd.read_pickle('../pickle/ics/training_database_index.pkl')
    try:
        max_id = int(training_database_index['id'].max())
    except:
        max_id = 0
    training_database_index = training_database_index.append(
        {'id': max_id + 1, 'file_name': file.filename, 'date': datetime.now(), 'data_rows': df_temp.shape[0]},
        ignore_index=True)
    training_database_index.to_pickle("../pickle/ics/training_database_index.pkl")

    return df


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# take all files from ../data/ics/uploads, clean and save them to ../pickle/ics/dataframes
def main(directory=r'..\data\ics\uploads'):
    data_list = glob.glob(directory + "/*.xlsx") + glob.glob(directory + "/*.csv")
    # empty pickle/dataframe folder
    shutil.rmtree('../pickle/ics/dataframes')
    os.mkdir('../pickle/ics/dataframes')
    # empty training database table

    if len(data_list) != 0:
        for file in data_list:
            df_temp = process_data(file)
        try:
            df_temp = pd.read_pickle('../pickle/ics/data_final.pkl')
        except:
            df_temp.to_pickle("../pickle/ics/data_final.pkl")
    else:
        print(data_list)
        print(r'No training data present in ..\data\ics\uploads')


if __name__ == '__main__':
    main()
