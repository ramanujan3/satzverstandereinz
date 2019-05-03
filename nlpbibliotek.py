#!/usr/bin/env python
from __future__ import division

import gensim
from nltk.tokenize import word_tokenize
import numpy as np


"""
VOKABULAR
Zeichenen = Tokenize
Vonfeld = From_field
Anfeld = To_field
Bauen worter modell = build words model
Anhangen wortcodes = Add word codes
"""


def zeichenen(daten_pd, vonfeld, anfeld):

    # daten_pd[anfeld] = daten_pd.apply(lambda row: word_tokenize(row[vonfeld]), axis=1)
    daten_pd[anfeld] = daten_pd.apply(lambda row: ([w for w in str(row[vonfeld])
                                      .replace('"', ' ').replace("'", "")
                                      .replace(',', ' ').replace(':', ' ').replace('.', ' ')
                                      .replace('-', ' ').replace('|', ' ').replace('/', ' ')
                                      .replace('#', ' ').replace('&', ' ').replace('+', ' ')
                                      .replace(')', ' ').replace('(', ' ').replace("*", " ")
                                      .replace('1', ' ').replace('2', ' ').replace('3', ' ')
                                      .replace('4', ' ').replace('5', ' ').replace('6', ' ')
                                      .replace('7', ' ').replace('8', ' ').replace('9', ' ').replace('0', ' ')
                                      .lower().split() if len(w) > 1]), axis=1)
    # daten_pd[anfeld] = daten_pd[vonfeld].split(' ')
    return daten_pd


def bauen_wortermodell(sentences, vec_size, vec_wind):
    model = gensim.models.Word2Vec(sentences, size=vec_size, window=vec_wind, min_count=1)
    return model


def anhangen_wortcodes(daten_pd, wortermodell, vekt_len, max_x, vonfeld, anfeld, anfeld2):

    wort_seq_list = daten_pd[vonfeld]
    wort_cseq_list = []
    wort_cseq_x_list = []
    # print 'wort_seq_list', wort_seq_list

    for i, wort_seq in enumerate(wort_seq_list):

        wort_cseq = []
        wort_cseq_x = []
        wort_seq_l = len(wort_seq)

        if (wort_seq_l >= max_x):
            ob_fill = True
            offset = 0
        else:
            ob_fill = False
            offset = max_x - wort_seq_l
            wort_cseq_x = [0] * vekt_len * offset

        # codes_avg = None
        for j, wort in enumerate(wort_seq):

            wort_code = np.array(wortermodell[wort])
            wort_cseq.append(wort_code)

            if (ob_fill):
                if (j < max_x):
                    wort_cseq_x.extend(wort_code)
            else:
                wort_cseq_x.extend(wort_code)
                   
            # print i, j,' wort_code', wort, wort_code

            # if (codes_avg is None):
            #     codes_avg = wort_code
            # else:
            #     codes_avg = codes_avg + wort_code
        # codes_avg = codes_avg / len(wort_seq)

        wort_cseq_list.append(wort_cseq)
        wort_cseq_x_list.append(wort_cseq_x)

    daten_pd[anfeld] = wort_cseq_list
    daten_pd[anfeld2] = wort_cseq_x_list
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
