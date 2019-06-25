#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import os

import plotly
import pandas as pd
from flask import Flask, flash, render_template, request
from ics_main import *
# from flask.ext.socketio import SocketIO, emit
from werkzeug.utils import secure_filename

app = Flask('SDC')

UPLOAD_FOLDER_ICS = '../data/ics/uploads'
ALLOWED_EXTENSIONS = set(['xlsx'])
app.config['UPLOAD_FOLDER_ICS'] = UPLOAD_FOLDER_ICS
app.secret_key = "secret!"
# socketio = SocketIO(app)

ics_training_database_table = pd.read_pickle('../pickle/ics/training_database_index.pkl')
# load training data
df = pd.read_pickle('../pickle/ics/data_final.pkl')


# main webpage
@app.route('/')
def start():
    return render_template('SDC.html')


# ics_master webpage displays training data and model statistics and receives user input text for model
@app.route('/ics')
def ics_master():
    subject_list, ids, graphJSON = ics_graphs()

    # render web page with plotly graphs
    return render_template('ics_master.html', subject_list=subject_list, ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/ics/go')
def go():
    query, classification_label, subclassification_label, related_queries, processes, subject_list = ics_go()
    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_label
        , subclassification_result=subclassification_label
        , related_queries=related_queries
        , processes=processes
        , subject_list=subject_list
    )


# upload training data
@app.route('/ics/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'train_data_file' not in request.files:
            flash('No File Selected')
            return render_template('ics_train.html', training_database=ics_training_database_table)
        file = request.files['train_data_file']
        # if user does not select file, browser can also submit an empty part without filename
        if file.filename == '':
            flash('No File Selected')
            return render_template('ics_train.html', training_database=ics_training_database_table)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                file.save(os.path.join(app.config['UPLOAD_FOLDER_ICS'], filename))
                save_data(file)
            except Exception as e:
                print(e)
                flash('Unable to process your request. Column names are wrong')
                return render_template('ics_train.html', training_database=ics_training_database_table)
            else:
                return render_template('ics_train_post.html',
                                       training_database=pd.read_pickle('../pickle/ics/training_database_index.pkl'))

    return render_template('ics_train.html', training_database=ics_training_database_table)


# ics help
@app.route('/ics_help')
def ics_help():
    return render_template('ics_help.html')


# portal for ROA data classification
@app.route('/roa')
def roa_master():
    return render_template('roa_master.html')


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()
