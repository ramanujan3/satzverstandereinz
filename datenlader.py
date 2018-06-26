#!/usr/bin/env python

from __future__ import division

import pandas as pd


def laden_docs(dir):
    daten_pd = pandas.read_csv(dir)
    return daten_pd
