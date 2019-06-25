#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:47:11 2019
@author: Gaurav Mundra
"""

import pandas as pd
from ics_data_clean import clean_text_round1


# reads 'keys.xlsx' file from data/ics/ directory for keywords, phrases and common questions and preprocess and save it to pickle/ics directory
def save_keys():
    # add weight during vectorizing
    print('Loading keys from ../data/ics/keys.xlsx')
    # key_weight = 5
    keys = pd.read_excel('../data/ics/keys.xlsx')
    keys.insert(0, "case_no", "key")
    keys.insert(1, "subject", "key")
    keys.insert(1, "subcategory", "")
    keys.insert(1, "case_activity", "key")
    keys.rename(columns={'key': 'content'}, inplace=True)

    keys_clean = pd.DataFrame(keys.content.apply(clean_text_round1))
    keys.rename(columns={'content': 'content_old'}, inplace=True)
    keys = pd.concat([keys, keys_clean], axis=1)
    print('Saving Keys...')
    keys.to_pickle("../pickle/ics/keys.pkl")


if __name__ == '__main__':
    save_keys()
