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
                        default=['~/matching_model/data/cl1.csv',
                                 '~/matching_model/data/cl2.csv',
                                 '~/matching_model/data/cl3.csv'],
                        help='verzeichnis zu laden')

    parser.add_argument('-pt', '--protext', nargs='+', type=str, dest='protext',
                        default=['Item Text', 'AP 3rd Level', 'Procurement Name'],
                        help='procurement namen zu laden')

    parser.add_argument('-pc', '--proclass', nargs='+', type=str, dest='proclass',
                        default=['Implan536Index', 'Implan536Index', 'Implan536Index'],
                        help='procurement namen zu laden')

    return parser.parse_args()


def main():

    args = parse_args()

    for i, vzchn in enumerate(args.vzchn):
        daten_pd = dl.laden_unterlagen(vzchn, [args.protext[i], 'Implan536Index'])

        # print daten_pd['Procurement Name']
        nl.zeichenen(daten_pd, args.protext[i], 'Procurement Zeichen')

    # nl.zeichenen(daten_pd, 'Procurement Name', 'Procurement Zeichen')
    wortermodell = nl.bauen_wortermodell(daten_pd['Procurement Zeichen'])

 
    worter = list(wortermodell.wv.vocab)
    worter.sort()
    # print(worter)
    print('Number of Words:',
          len(worter))

    nl.anhangen_wortcodes(daten_pd, wortermodell, 'Procurement Zeichen', 'Procurement Coden')

    print('Number of Procurement lines:',
          len(daten_pd['Procurement Coden']))
    # print(wortermodell['aramark'])

    print('Done!')


if __name__ == "__main__":
    main()
