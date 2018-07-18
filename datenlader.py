#!/usr/bin/env python

from __future__ import division

import pandas as pd


"""
Worter (Words)
Verzeichnis = directory
Laden = load
Die Unterlagen = the docs
Die Konfiguration = the config
Die Spalten = columns
Benutzt = used
"""


def laden_konfiguration(verzeichnis, spalten_benutzt):
    """
    Eine Funktion zum laden der Konfiguration
    """
    konfigs_pd = pd.read_csv(verzeichnis, usecols=spalten_benutzt)
    return konfigs_pd


def laden_unterlagen(verzeichnis, spalten_benutzt):
    """
    Eine Funktion zum laden Unterlagen
    """

    print verzeichnis
    daten_pd = pd.read_csv(verzeichnis, usecols=spalten_benutzt)
    return daten_pd
