"""Contains all methods to do preprocessing in this progect"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class Preprocessor:
    @staticmethod
    def fill_nans(dat):
        """Fills in NaNs with either the mean or the most common value"""
        return DataFrameImputer.fit_transform(dat)


class DataFrameImputer(TransformerMixin):
    '''
    This class came from http://stackoverflow.com/questions/25239958/
    impute-categorical-missing-values-in-scikit-learn
    '''

    def __init__(self):
        '''
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value in col.
        Columns of other types are imputed with mean of column.
        '''

    def fit(self, x, y=None):

        self.fill = pd.Series([x[c].value_counts().index[0]
            if x[c].dtype == np.dtype('O') else x[c].mean() for c in x],
            index = x.columns)

        return self

    def transform(self, x, y=None):
        return x.fillna(self.fill)