#!/usr/bin/env python

from __future__ import division

import argparse
# import pandas as pd
import datenlader as dl

###########

# Check in test

import nlpbibliotek as nl


"""
Worter (Words)
Satz = sentence
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
                        default=30,
                        help='Grosse des Vektors')

    parser.add_argument('-vf', '--vekfenster', type=int, dest='vek_fenster',
                        default=3,
                        help='Fenster des Vektors')

    return parser.parse_args()


def main():

    args = parse_args()

    for i, vzchn in enumerate(args.vzchn):

        cols_zu_laden = args.protexts[i] + [args.proclass[i]]
        print(args.protexts[i])
        print(args.proclass[i])
        print(cols_zu_laden)
        daten_pd = dl.laden_unterlagen(vzchn, cols_zu_laden)

        # print daten_pd['Procurement Name']

        daten_pd['Procurement String'] = daten_pd[args.protexts[i]].apply(lambda x: ' '.join(x), axis=1)
        nl.zeichenen(daten_pd, 'Procurement String', 'Procurement Zeichen')

    # nl.zeichenen(daten_pd, 'Procurement Name', 'Procurement Zeichen')
    wortermodell = nl.bauen_wortermodell(daten_pd['Procurement Zeichen'], args.vek_grosse, args.vek_fenster)

 
    worter = list(wortermodell.wv.vocab)
    worter.sort()
    print(worter)
    print('Number of Words:',
          len(worter))

    nl.anhangen_wortcodes(daten_pd, wortermodell, 'Procurement Zeichen', 'Procurement Coden')

    print(daten_pd.iloc[234])

    print('Number of Procurement lines:',
          len(daten_pd['Procurement Coden']))
    
    # print(wortermodell['aramark'])

    print("aramark")
    print(wortermodell.similar_by_vector('aramark', 5))
    print("wholesale")
    print(wortermodell.similar_by_vector('wholesale', 5))
    print("recruitment")
    print(wortermodell.similar_by_vector('recruitment', 5))
    print("zurich")
    print(wortermodell.similar_by_vector('zurich', 5))

    print('Done!')


if __name__ == "__main__":
    main()
