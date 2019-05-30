#!/usr/bin/env python

from __future__ import division

import argparse
import time
import pandas as pd
import numpy as np

import datenlader as dl
import nlpbibliotek as nl
import vorhersager as vs

from sklearn.model_selection import train_test_split


"""
VOKABULAR
Worter (Words)
Satz = setence
Zeichen = Tokens
Verzeichnis (vzchn) = directory
Verstander = Understander

Laden = load
Die Unterlagen = the docs
"""


def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vz', '--vzchn', nargs='+', type=str, dest='vzchn',
                        default=['~/matching_model/data/cl3MM.csv',
                                 # '~/matching_model/data/cl1_100k.csv'],
                                 '~/matching_model/data/cl2f.csv'],
                                 # '~/matching_model/data/cl1.csv'],
                                 # '~/matching_model/data/cl2.csv'],
                                 # '~/matching_model/data/cl3.csv'],
                        help='verzeichnis zu laden')

    parser.add_argument('-pt', '--protexts', nargs='+', type=str, dest='protexts',
                        default=[['Procurement taxonomy', 'GL Name'],
                        		 # ['Vendor', 'Item Text']],
                        		 ['AP Sub Category', 'Company Name', 'AP 3rd Level']],
                        		 # ['Procurement Name', 'Procurement Name', 'GL Name']],
                        help='procurement namen zu laden')

    parser.add_argument('-pc', '--proclass', nargs='+', type=str, dest='proclass',
                        default=['Implan536Index',
                      			 'Implan536Index'],
                                 # 'Implan536Index',
                                 # 'Implan536Index'],
                        help='procurement namen zu laden')

    parser.add_argument('-tsf', '--traintestfinal', nargs='+', type=str, dest='tsf',
                        default=['T', 'F'])

    parser.add_argument('-vg', '--vekgrosse', type=int, dest='vek_grosse',
                        default=16,
                        help='Grosse des Vektors')

    parser.add_argument('-vf', '--vekfenster', type=int, dest='vek_fenster',
                        default=5,
                        help='Fenster des Vektors')

    return parser.parse_args()


def main():

    args = parse_args()

    SATZ_MAXLEN = 8
    CLS_MIN = 0
    CLS_MAX = 536

    start_time_1 = time.time()

    daten_ls = []
    daten_fl = None
    final_name = "No_name"
    for i, vzchn in enumerate(args.vzchn):

        print('File', i, vzchn, args.tsf[i])

        cols_zu_laden = args.protexts[i] + [args.proclass[i]]
        # print(args.protexts[i])
        # print(args.proclass[i])
        # print(cols_zu_laden)
        daten_pdi = dl.laden_unterlagen(vzchn, cols_zu_laden)

        # print daten_pd['Procurement Name']

        daten_pdi['procure_str'] = daten_pdi[args.protexts[i]].apply(lambda x: ' '.join(map(str, x)), axis=1)
        daten_pdi['procure_cls'] = daten_pdi[args.proclass[i]]

        daten_pdi['File ID'] = i
        print(daten_pdi['procure_str'][:4])
        daten_pdi = nl.zeichenen(daten_pdi, 'procure_str', 'procure_zeichen')
        daten_pdi['zeichen_len'] = daten_pdi['procure_zeichen'].str.len()
        daten_pdi['tsf'] = args.tsf[i]
        if (args.tsf[i] == 'F'):
            # daten_fl = daten_pdi
            final_name = vzchn
        daten_ls.append(daten_pdi)
    daten_pd = pd.concat(daten_ls)

    # nl.zeichenen(daten_pd, 'Procurement Name', 'Procurement Zeichen')
    print('  --- data load time: ', elapsed(time.time() - start_time_1))

    # -------------------------------------------

    start_time = time.time()

    print(daten_pd.head(10))
    print(daten_pd['procure_zeichen'][:10])
    wortermodell = nl.bauen_wortermodell(daten_pd['procure_zeichen'], args.vek_grosse, args.vek_fenster)
    worter = list(wortermodell.wv.vocab)
    print('          analyzed ', len(worter), ' words')

    print('  --- build words model time: ', elapsed(time.time() - start_time), '\n')
    worter.sort()
    print(worter[:37])

    # -------------------------------------------

    start_time = time.time()
    # No Matches can't be used for training and testing
    # lbld_daten_pd = daten_pd[daten_pd.procure_cls != "No Match"]
    # But let's try making them zeros
    lbld_daten_pd = daten_pd  # .copy()
    # lbld_daten_pd[lbld_daten_pd['procure_cls'] is "No Match"]['procure_cls'] = '0'

    lbld_daten_pd['procure_cls'].replace("No Match", "0", inplace=True)

    nl.anhangen_wortcodes(lbld_daten_pd, worter, wortermodell, args.vek_grosse, SATZ_MAXLEN,
                          'procure_zeichen', 'procure_coden', 'procure_x')

    # print('\n', daten_pd.iloc[234])
    # print('\n', daten_pd.iloc[2])

    print("\n Min/avg/max string lengths:", lbld_daten_pd['zeichen_len'].min(),
                                            lbld_daten_pd['zeichen_len'].mean(),
                                            lbld_daten_pd['zeichen_len'].max())
    lbld_daten_pd['procure_clsy'] = pd.to_numeric(lbld_daten_pd['procure_cls'])
    print(" Min/avg/max procurement class:", lbld_daten_pd['procure_clsy'].min(),
                                             lbld_daten_pd['procure_clsy'].mean(),
                                             lbld_daten_pd['procure_clsy'].max())
    print(' Num procure lines:',
          len(daten_pd))
    print(' Num labeled procure lines:',
          len(lbld_daten_pd))

    # print(daten_pd.iloc[234]['procure_coden'])
    # print(daten_pd.iloc[234]['procure_x'])

    lbld_daten_pd = lbld_daten_pd.sample(frac=1)

    x = lbld_daten_pd['procure_x']
    y_trn = []
    y_tst = []
    y_fnl = []
    # for clsyz, ttf in lbld_daten_pd['procure_clsy'], lbld_daten_pd['ttf']:
    #for clsyz, ttf in zip(lbld_daten_pd['procure_clsy'], lbld_daten_pd['ttf']):
    for clsyz, tsf in zip(lbld_daten_pd['procure_clsy'], lbld_daten_pd['tsf']):
        # print(tsf, clsyz)
        y_i = [0] * (CLS_MAX - CLS_MIN)
        y_i[clsyz - CLS_MIN] = 1
        if (tsf == 'T'):
            y_trn.append(y_i)
        elif (tsf == 'S'):
            y_tst.append(y_i)
        elif (tsf == 'F'):
            y_fnl.append(y_i)

    x_trn = x[lbld_daten_pd['tsf'] == 'T']  # .sample(frac=1)
    x_tst = x[lbld_daten_pd['tsf'] == 'S']  # .sample(frac=1)
    x_fnl = x[lbld_daten_pd['tsf'] == 'F']  # .sample(frac=1)
    # y_trn = y[lbld_daten_pd['ttf'] == 1].sample(frac=1)
    # y_tst = y[lbld_daten_pd['ttf'] == 2].sample(frac=1)
    # y_fnl = y[lbld_daten_pd['ttf'] == 3].sample(frac=1)

    x_trn = np.array([np.array(xi) for xi in x_trn])
    x_tst = np.array([np.array(xi) for xi in x_tst])
    x_fnl = np.array([np.array(xi) for xi in x_fnl])
    y_trn = np.array([np.array(yi) for yi in y_trn])
    y_tst = np.array([np.array(yi) for yi in y_tst])
    y_fnl = np.array([np.array(yi) for yi in y_fnl])

    x_trn = np.expand_dims(x_trn, axis=2)
    x_tst = np.expand_dims(x_tst, axis=2)
    x_fnl = np.expand_dims(x_fnl, axis=2)
    # x = x.reshape(23462, 77, 1)

    # x = np.array(x)

    # print('\n\n x:', x[234])
    # print('y: ', y[234])

    # print('\n', "aramark")
    # print(wortermodell.similar_by_vector('aramark', 5))
    # print("wholesale")
    # print(wortermodell.similar_by_vector('wholesale', 5))
    # print("recruitment")
    # print(wortermodell.similar_by_vector('recruitment', 5))
    # print("zurich")
    # print(wortermodell.similar_by_vector('zurich', 5))

    # ----
    # x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=13)
    # x = np.array([np.array(xi) for xi in x])
    # y = np.array([np.array(yi) for yi in y])
    # x = np.expand_dims(x, axis=2)

    print('\n', '  --- prep data and stats time: ', elapsed(time.time() - start_time), '\n')

    # -------------------------------------------

    start_time = time.time()

    m_hidden = 512
    epochs = 9
    lamba_lrate = 0.002
    batch_size = 128

    y_pred, y_conf = vs.run_lstm_KR(x_trn, x_tst, x_fnl, y_trn, y_tst, y_fnl,
                                    m_hidden, epochs, lamba_lrate, batch_size)

    lbld_daten_final = lbld_daten_pd[lbld_daten_pd['tsf'] == 'F']
    print("Saving data")

    # for clsyz, tsf in zip(lbld_daten_pd['procure_clsy'], lbld_daten_pd['tsf']):
    #     find

    print(len(y_pred))
    print(len(y_conf))
    print(lbld_daten_final.shape)
    lbld_daten_final['Pred_Implan536Index'] = y_pred
    lbld_daten_final['Pred_Conf'] = y_conf
    # daten_fl['Pred_Implan536Index'] = y_pred
    # daten_fl['Pred_Implan536Index'] = y_pred

    lbld_daten_final.to_csv(final_name + '_pred.csv')

    print('W2V: WVec Len \t| SentenceMax Len ')
    print('             ', args.vek_grosse, ' \t |', SATZ_MAXLEN)
    print('LSTM: Hidden Size \t| Epochs   \t| Learn R   \t| batch_size')
    print('              ', m_hidden, '\t ', epochs, '\t ', lamba_lrate, '\t ', batch_size)

    print("   --- machine learning time: ", elapsed(time.time() - start_time))
    print(' --- total time: ', elapsed(time.time() - start_time_1))
    print('\n--ERLEDIGT--')

if __name__ == "__main__":
    main()
