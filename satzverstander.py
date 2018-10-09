#!/usr/bin/env python

from __future__ import division

import argparse
# import pandas as pd
import datenlader as dl
import nlpbibliotek as nl


"""
Worter (Words)
Satz = sentence
Verzeichnis (vzchn) = directory
Verstander = Understander

Laden = load
Die Unterlagen = the docs
Zeichen = Tokens
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vz', '--vzchn', type=str, dest='vzchn', default='./',
                        help='verzeichnis zu laden')

    return parser.parse_args()


def main():

    args = parse_args()

    daten_pd = dl.laden_unterlagen(args.vzchn, ['Procurement Name', 'Implan536Index'])

    # print daten_pd['Procurement Name']

    nl.zeichenen(daten_pd, 'Procurement Name', 'Procurement Zeichen')
    wortermodell = nl.bauen_wortermodell(daten_pd['Procurement Zeichen'])

    worter = list(wortermodell.wv.vocab)
    worter.sort()
    print worter
    print len(worter)

    nl.anhangen_wortcodes(daten_pd, wortermodell, 'Procurement Zeichen', 'Procurement Coden')

    print daten_pd['Procurement Coden']
    # print(wortermodell['aramark'])

    print('Done!')


if __name__ == "__main__":
    main()
