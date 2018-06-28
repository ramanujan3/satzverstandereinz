#!/usr/bin/env python

from __future__ import division

import argparse
# import pandas as pd
import datenlader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vz', '--vzchn', type=str, dest='verzeichnis', default='./', help='verzeichnis zu laden')

    return parser.parse_args()


def main():

    args = parse_args()

    daten_pd = datenlader.laden_unterlagen(args.verzeichnis, ['Procurement Name', 'Implan536Index'])
    print('Done!')


if __name__ == "__main__":
    main()
