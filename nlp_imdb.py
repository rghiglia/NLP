# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:54:04 2016

@author: rghiglia
"""

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd

dnm = r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\NLP'
fnm = r'labeledTrainData.tsv'
fnmL = dnm + '\\' + fnm
train = pd.read_csv(fnmL, header=0, delimiter="\t", quoting=3)
