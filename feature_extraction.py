"""
Extract Features
"""

import numpy as np
from collections import Counter
from constants import *
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def count_common_words_in_subject(df, labels):
    """
    :param df: raw training dataset
    :param labels:
    :return: common words in email subject for phishing emails and ham emails
    """
    phish_idxs = labels[labels['Label'] == 1].index
    ham_idxs = labels[labels['Label'] == 0].index
    df = df.replace(np.nan, '', regex=True) # replace all missing values with empty string
    df['Subject_lower'] = df['Subject'].apply(lambda x: x.lower())

    phish_words_subject = Counter(" ".join(df.loc[phish_idxs,'Subject_lower']).split()).most_common(20)
    most_common_words_phising_subject = [i[0] for i in phish_words_subject if not i[0] in stop_words]
    ham_words_subject = Counter(" ".join(df.loc[ham_idxs, 'Subject_lower']).split()).most_common(20)
    most_common_words_ham_subject = [i[0] for i in ham_words_subject if not i[0] in stop_words]

    most_common_words_phising_subject_unique = [e for e in most_common_words_phising_subject if
                                                e not in most_common_words_ham_subject]
    most_common_words_ham_subject_unique = [e for e in most_common_words_ham_subject if
                                            e not in most_common_words_phising_subject]

    return most_common_words_phising_subject_unique, most_common_words_ham_subject_unique



# -*- coding: utf-8 -*-
def is_ascii(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_links_text(links_list):
    text_in_links = []
    for t in links_list:
        text_in_links.append(t.text)
    return text_in_links


def create_features(df, text_cols, common_words_subject_phishing, common_words_subject_ham):
    # TODO: optimize parameters transfer of common subject words
    df_extended = df.copy()
    # define null values features
    cols_with_missing_values = df.columns[df.isnull().any()].tolist()
    for miss_col in cols_with_missing_values:
        df[miss_col+'_missing'] = df[miss_col].isnull()
    # replace all missing values with empty string
    df = df.replace(np.nan, '', regex=True)
    # find if dtype is html
    df['html_dtype'] = df['Content-Type'].str.contains('html')
    # handle text data
    for text_col in text_cols:
        # make lower. if missing write 'missing'
        df[text_col+'_lower'] = df[text_col].apply(lambda x: x.lower())
        # count uppercase and lowercase
        df[text_col+'_Uppercase'] = df[text_col].str.findall(r'[A-Z]').str.len()
        df[text_col+'_Lowercase'] = df[text_col].str.findall(r'[a-z]').str.len()
        # text length
        df[text_col+'_length'] = df[text_col].str.len()
        # is english
        df[text_col+'_is_ascii'] = df[text_col+'_lower'].apply(is_ascii)
        # find links
        df[text_col+'_all_links'] = df[text_col+'_lower'].apply(lambda x: BeautifulSoup(x, 'html.parser').find_all('a'))
        df[text_col+'_n_links'] = df[text_col+'_all_links'].apply(lambda x: len(x))
        df[text_col+'_links_presence'] = df[text_col+'_n_links'] > 0
        df[text_col+'_suspicious_words_in_links_text'] = df[text_col+'_all_links'].apply(
            lambda x: any(('click' or 'link' or 'here' or 'login' or 'update')
                          in l for l in get_links_text(x)))
    # common words for phishing in subject
    df['Subject_phish_words'] = df['Subject_lower'].str.contains('|'.join(common_words_subject_phishing))
    df['Subject_ham_words'] = df['Subject_lower'].str.contains('|'.join(common_words_subject_ham))
    # common words in phising emails content from https://www.hindawi.com/journals/jam/2014/425731/
    df['Content_phish_words_1'] = df['Content_lower'].str.contains('|'.join(Content_common_phish_words_1))
    df['Content_phish_words_2'] = df['Content_lower'].str.contains('|'.join(Content_common_phish_words_2))
    df['Content_phish_words_3'] = df['Content_lower'].str.contains('|'.join(Content_common_phish_words_3))
    df['Content_phish_words_4'] = df['Content_lower'].str.contains('|'.join(Content_common_phish_words_4))
    df['Content_phish_words_5'] = df['Content_lower'].str.contains('|'.join(Content_common_phish_words_5))

    df.drop(cols_to_drop, axis=1)

    return df_extended
