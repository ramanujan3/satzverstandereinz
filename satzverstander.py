#!/usr/bin/env python

from __future__ import division

import argparse
import pandas as pd

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vz', '--vzchn', nargs='+', type=str, dest='vzchn',
                        default=['~/matching_model/data/cl3.csv'],
                                 # '~/matching_model/data/cl1.csv',
                                 # '~/matching_model/data/cl2.csv',
                                 # '~/matching_model/data/cl3.csv'],
                        help='verzeichnis zu laden')

    parser.add_argument('-pt', '--protexts', nargs='+', type=str, dest='protexts',
                        default=[['Procurement Name', 'GL Name']],
                        		 # 'Item Text', 
                        		 # 'AP 3rd Level',
                        		 # 'Procurement Name'],
                        help='procurement namen zu laden')

    parser.add_argument('-pc', '--proclass', nargs='+', type=str, dest='proclass',
                        default=['Implan536Index'],
                      			 # 'Implan536Index',
                                 # 'Implan536Index',
                                 # 'Implan536Index'],
                        help='procurement namen zu laden')

    parser.add_argument('-vg', '--vekgrosse', type=int, dest='vek_grosse',
                        default=4,
                        help='Grosse des Vektors')

    parser.add_argument('-vf', '--vekfenster', type=int, dest='vek_fenster',
                        default=4,
                        help='Fenster des Vektors')

    return parser.parse_args()

def main():

    args = parse_args()

    SATZ_MAXLEN = 4
    CLS_MIN = 60
    CLS_MAX = 510

    for i, vzchn in enumerate(args.vzchn):

        cols_zu_laden = args.protexts[i] + [args.proclass[i]]
        print(args.protexts[i])
        print(args.proclass[i])
        print(cols_zu_laden)
        daten_pd = dl.laden_unterlagen(vzchn, cols_zu_laden)

        # print daten_pd['Procurement Name']

        daten_pd['procure_str'] = daten_pd[args.protexts[i]].apply(lambda x: ' '.join(x), axis=1)
        daten_pd['procure_cls'] =  daten_pd[args.proclass[i]]

        daten_pd['File ID'] = i
        nl.zeichenen(daten_pd, 'procure_str', 'procure_zeichen')
        daten_pd['zeichen_len'] = daten_pd['procure_zeichen'].str.len()


    # nl.zeichenen(daten_pd, 'Procurement Name', 'Procurement Zeichen')
    wortermodell = nl.bauen_wortermodell(daten_pd['procure_zeichen'], args.vek_grosse, args.vek_fenster)

 
    worter = list(wortermodell.wv.vocab)
    # worter.sort()
    # print(worter)

    # No Matches can't be used for training and testing
    lbld_daten_pd = daten_pd[daten_pd.procure_cls != "No Match"]

    print('\nNum words:',
		len(worter))

    nl.anhangen_wortcodes(lbld_daten_pd, wortermodell, args.vek_grosse, SATZ_MAXLEN,
                          'procure_zeichen', 'procure_coden', 'procure_x')

    print('\n', daten_pd.iloc[234])
    print('\n', daten_pd.iloc[2])

    print("\n Min/avg/max string lengths:", lbld_daten_pd['zeichen_len'].min(),
                                         lbld_daten_pd['zeichen_len'].mean(),
                                         lbld_daten_pd['zeichen_len'].max())
    lbld_daten_pd['procure_clsy'] = pd.to_numeric(lbld_daten_pd['procure_cls'])
    print(" Min/avg/max procurement class:", lbld_daten_pd['procure_clsy'].min(),
                                         lbld_daten_pd['procure_clsy'].mean(),
                                         lbld_daten_pd['procure_clsy'].max())
    print('\n Num procure lines:',
          len(daten_pd))
    print(' Num labeled procure lines:',
          len(lbld_daten_pd))



    # print(daten_pd.iloc[234]['procure_coden'])
    # print(daten_pd.iloc[234]['procure_x'])

    x = lbld_daten_pd['procure_x']
    y = []
    for clsyz in lbld_daten_pd['procure_clsy']:
        y_i = [0] * (CLS_MAX - CLS_MIN)
        y_i[clsyz - CLS_MIN] = 1
        y.append(y_i)

    print('\n\n x:', x[234])
    print('y', y[234])
    
    # print(wortermodell['aramark'])

    print('\n', "aramark")
    print(wortermodell.similar_by_vector('aramark', 5))
    print("wholesale")
    print(wortermodell.similar_by_vector('wholesale', 5))
    print("recruitment")
    print(wortermodell.similar_by_vector('recruitment', 5))
    # print("zurich")
    # print(wortermodell.similar_by_vector('zurich', 5))

    print('\n\n --- ERLEDIGT --- ')

    x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=13)
    vs.run_lstm(x_trn, x_tst, y_trn, y_tst)

if __name__ == "__main__":
    main()
