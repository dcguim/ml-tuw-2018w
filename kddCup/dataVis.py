#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script will predict if a direct mailing campaign's recipient will donate or not,
 the data are from the KDD Cup 98 (A small subset)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pydoc import help
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Reads project's classes
from lib.importer import Importer
from lib.preprocessor import Preprocessor
from lib.analyser import Analyser
from lib.utils import Performance


def main():
    # load th configuration
    cfg = Importer.get_cfg()

    # load the raw data
    raw_dat = Importer.get_raw_dat(cfg)

    print(raw_dat.shape)
    print(raw_dat.count()) # checks how many missing values are in the dataset
    print(raw_dat.head())
    print(raw_dat.columns)

    # Distribution of the target variables
    plt.plot(raw_dat.TARGET_B)
    plt.show()

    # Correlation between TARGET_B and the predictors
    target_b_corr = raw_dat.corr()["TARGET_B"].copy()
    #    TARGET_B_corr.sort(False)
    target_b_corr



if __name__ == '__main__':
    main()
