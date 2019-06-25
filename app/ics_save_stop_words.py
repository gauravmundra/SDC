#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import pickle

from sklearn.feature_extraction import text


def stopwords():
    add_stop_words = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                      'aug', 'sept', 'oct', 'nov', 'dec', 'january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december', 'please', 'see', 'attached',
                      'attach', 'gm', 'good', 'morning', 'morn', 'eve', 'evening', 'afternoon', 'noon', 'dear',
                      'regard', 'regards', 'team', 'thank', 'you', 'thanks', 'folks', 'folk', 'ideas', 'hello', 'hi',
                      'today', 'yesterday', 'afternoon', 'evening', 'night', 'like', 'im', 'asap', 'happy', 'hayley',
                      'munchen', 'gds', 'flemings', 'compititorsyou', 'hq']
    senseless_words = ['screensjot', 'aaa', 'ap', 'aarp', 'aaaaarp', 'abdelwahed', 'abdo', 'abdullah', 'abhishek',
                       'zrhzq', 'bslzh']
    add_stop_words = add_stop_words + senseless_words

    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    with open('../pickle/ics/stop_words.pickle', 'wb') as f:
        pickle.dump(stop_words, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    stopwords()
