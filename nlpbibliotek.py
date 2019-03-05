#!/usr/bin/env python

from __future__ import division

import gensim
from nltk.tokenize import word_tokenize
import numpy as np


"""
Zeichenen = Tokenize
Vonfeld = From_field
Anfeld = To_field
Bauen worter modell = build words model
Anhangen wortcodes = Add word codes
"""


def zeichenen(daten_pd, vonfeld, anfeld):

    # daten_pd[anfeld] = daten_pd.apply(lambda row: word_tokenize(row[vonfeld]), axis=1)
    daten_pd[anfeld] = daten_pd.apply(lambda row: row[vonfeld]
                                      .replace('"', '').replace("'", "")
                                      .replace(',', ' ').replace(':', ' ').replace('.', ' ')
                                      .replace('-', ' ').replace('|', ' ').replace('/', ' ')
                                      .replace('#', ' ').replace('&', ' ').replace('+', ' ')
                                      .replace('«', ' ').replace('–', ' ')
                                      .replace(')', ' ').replace('(', ' ').lower()
                                      .split(), axis=1)
    # daten_pd[anfeld] = daten_pd[vonfeld].split(' ')
    return daten_pd


def bauen_wortermodell(sentences, vec_size, vec_wind):
    model = gensim.models.Word2Vec(sentences, size=vec_size, window=vec_wind, min_count=1)
    return model


def anhangen_wortcodes(daten_pd, wortermodell, vonfeld, anfeld):

    wort_seq_list = daten_pd[vonfeld]
    wort_cseq_list = []
    # print 'wort_seq_list', wort_seq_list

    for i, wort_seq in enumerate(wort_seq_list):

        wort_cseq = []
        # codes_avg = None
        for j, wort in enumerate(wort_seq):

            wort_code = np.array(wortermodell[wort])
            wort_cseq.append(wort_code)
            # print i, j, 'wort_code', wort, wort_code

            # if (codes_avg is None):
            #     codes_avg = wort_code
            # else:
            #     codes_avg = codes_avg + wort_code
        # codes_avg = codes_avg / len(wort_seq)

        wort_cseq_list.append(wort_cseq)

    daten_pd[anfeld] = wort_cseq_list
    return daten_pd


def bauen_ausbildungen(daten_pd, wortermodell, vonfeld, neuefeld):


    wort_seq_list = daten_pd[vonfeld]
    wort_cseq_list = []
    # print 'wort_seq_list', wort_seq_list


    for i, wort_seq in enumerate(wort_seq_list):

        wort_cseq = []
        # codes_avg = None
        for j, wort in enumerate(wort_seq):

            wort_code = np.array(wortermodell[wort])
            wort_cseq.append(wort_code)
            # print i, j, 'wort_code', wort, wort_code

            # if (codes_avg is None):
            #     codes_avg = wort_code
            # else:
            #     codes_avg = codes_avg + wort_code
        # codes_avg = codes_avg / len(wort_seq)

        wort_cseq_list.append(wort_cseq)

    daten_pd[anfeld] = wort_cseq_list
    return daten_pd
