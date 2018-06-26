#!/usr/bin/env python

from __future__ import division

import os, sys
import argparse, logging
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def main():

    args = parse_args()
    logging.basicConfig()

    laden_docs(args)
    print('Done!')


if __name__ == "__main__":
    main()
