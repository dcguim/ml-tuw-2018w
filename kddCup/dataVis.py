#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script will predict if a direct mailing campaign's recipient will donate or not,
 the data are from the KDD Cup 98 (A small subset)"""

import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin

from pydoc import help
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold

# Reads project's classes
from lib.importer import Importer
from lib.preprocessor import Preprocessor
from lib.analyser import Analyser
from lib.utils import Performance

def get_corr_vars(dat : pd.DataFrame ,  corr_val):
    """Outputs a list of redundant vars that are correlated with others"""

    # Computes correlation
    dat_cor = dat.corr()

    # Cherry picks the lower triangular without the diagonal
    dat_cor.loc[:, :] = np.tril(dat_cor, k=-1)

    # Stack the data
    dat_cor = dat_cor.stack()

    # get list of the correlated vars
    corr_pairs = dat_cor[dat_cor > corr_val].to_dict().keys()
    chosen_vars = [i[0] for i in corr_pairs]
    chosen_vars.extend([i[1] for i in corr_pairs if i[1] not in chosen_vars])

    redundent_vars = [var for var in [x for t in corr_pairs for x in t] if var not in chosen_vars]

    return redundent_vars

def get_redundant_vars(target, dat):
    """This method outputs a set of redundant variables."""

    redundant_vars = ['CONTROLN', 'ZIP']

    # Some vars that don't seem of good value
    redundant_vars = ['CONTROLN', 'ZIP']

    # Identifies numerical variables with variance zero < 0.1%
    #sel = feature_selection.VarianceThreshold(threshold = 0.001)
    #sel.fit_transform(dat)
    dat_var = dat.var()
    redundant_vars.extend(dat_var.index[dat_var < 0.001])

    # Identifies variables that are too sparse (less than 1%)
    idxs = dat.count() < int(dat.shape[0] * .01)
    redundant_vars.extend(dat.columns[idxs])

    # Identifies variables that are strongly correlated with others
    #redundant_vars.extend(Analyser.get_corr_vars(dat, corr_val = 0.9))

    return redundant_vars



def get_important_vars(target, dat):
        '''
        This method does Feature Selection.
        '''

        # Balances the dataset
        idxs_pos = dat[target] == 1
        pos = dat[idxs_pos]
        neg = dat[dat[target] == 0][1:sum(idxs_pos)]

        # Concatenates pos and neg, it's already shuffled
        sub_dat = pos.append(neg, ignore_index = True)

        # Imputes the data and fills in the missing values
        sub_dat = Preprocessor.fill_nans(sub_dat)

        # Changes categorical vars to a numerical form
        X = pd.get_dummies(sub_dat)

        #### Correlation-based Feature Selection ####

        # Computes correlation between cfg['target'] and the predictors
        target_corr = X.corr()[target].copy()
        target_corr = target_corr.sort_values(ascending = False)

        # Sorts and picks the first x features
        # TODO: get optimal x value automatically
        tmp = target_corr.abs()
        tmp = tmp.sort_values(ascending = False)
        important_vars = [tmp.index[0]]
        important_vars.extend(list(tmp.index[1:100]))

        #### Variance-based Feature Selection ####

        #sel = VarianceThreshold(threshold = 0.005)
        #X_new = sel.fit_transform(X)

        #### Univariate Feature Selection ####

        #y = X.TARGET_B
        #X = X.drop("TARGET_B", axis = 1)

        #X_new = SelectKBest(chi2, k = 10).fit_transform(X.values, y.values)

        #### Tree-based Feature Selection ####

        #clf = ExtraTreesClassifier()
        #X_new = clf.fit(X.values, y.values).transform(X.values)

        #aux = dict(zip(X.columns, clf.feature_importances_))
        #important_vars = [i[0] for i in sorted(
        #    aux.items(), key = operator.itemgetter(0))]

        return important_vars

#### Exploratory Analysis ####
def expl(data):
    return print(data.shape), print(data.count()), print(data.head()),print(data.columns)



class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)



def main():
    # load th configuration
    cfg = Importer.get_cfg()

    target = cfg['target']

    # load the raw data and test
    raw_dat = Importer.get_raw_dat(cfg)
    test_raw_dat = Importer.get_test_dat(cfg)

    #expl(raw_dat)

    # Distribution of the target variables
    plt.plot(raw_dat.TARGET_B)
    #plt.show()

    # Correlation between TARGET_B and the predictors
    target_b_corr = raw_dat.corr()[target].copy()
    target_b_corr = target_b_corr.sort_values(ascending=False)
    print(target_b_corr)

    # Some statistics about raw data variables
    raw_dat.describe()
    print(raw_dat.describe())

    # Variable distribution

    # [1:28] # demographics
    # [29:42] # response to other types of mail orders
    # [43:55] # overlay data
    # [56:74] # donor interests
    # [75] # PEP star RFA status
    # [76:361] # characteristics of donor neighborhood
    # [362:407] # promotion history
    # [408:412] # summary variables of promotion history
    # [413:456] # giving history
    # [457:469] # summary variables of giving history
    # [470:472] ## ID & TARGETS
    # [473:478] # RFA (recency-frequency-donation amount)
    # [479:480] # cluster & geocode

    # % of donors
    print('Percentage of donors: %s' % (
            100.0 * sum(raw_dat.TARGET_B) / raw_dat.shape[0]))

    # This data is quite noisy, high dimensional, with lots of missing values
    # and just with 5% of positive cases. Feature selection and preprocessing
    # will be vital for good modelling.

    #### Preprocessing ####

    # Gets some redundant variables based on variance, sparsity & common sense
    redundant_vars = get_redundant_vars(target, raw_dat)

    print(redundant_vars)
    print(redundant_vars[1:])

    #drop redundant variables
    dat = raw_dat.drop(redundant_vars, axis=1)
    test_data = test_raw_dat.drop(redundant_vars[1:], axis=1)

    dat = DataFrameImputer().fit_transform(dat)
    test_data = DataFrameImputer().fit_transform(test_data)

    # Shuffles observations
    dat.apply(np.random.permutation)

    # Gets important variables
    important_vars = get_important_vars('TARGET_B', dat)

    # Changes categorical vars to a numerical form
    feats = pd.get_dummies(dat)
    feats_test = pd.get_dummies(test_data)

    # Drops the non-important variables
    feats = feats[important_vars]
    control= test_data.CONTROLN
    control_feats = feats_test.CONTROLN
    feats_test = feats_test[important_vars[1:]]

    # Does train/test datasets, 70% and 30% respectively
    cut = int(feats.shape[0] * .7)

    train = feats[1:cut].drop(['TARGET_B'], axis=1)
    y_train = feats.TARGET_B[1:cut]

    test = feats[(cut + 1):-1].drop(['TARGET_B'], axis=1)
    y_test = feats.TARGET_B[(cut + 1):-1]

    # Creates a balanced trainset
    # In classification, some methods perform better with bal datasets,
    # particularly tree-based methods like decision trees and random forests.
    pos = train[y_train == 1]
    neg = train[y_train == 0][1:pos.shape[0]]
    y_train_bal = [1] * pos.shape[0]
    y_train_bal.extend([0] * neg.shape[0])
    train_bal = pos.append(neg, ignore_index=True)

    #### Training ####

    #### Model 1 | Decision Tree Model ####

    print("Model 1 executing...")

    # Training
    clf = DecisionTreeClassifier(max_depth=20)  # TODO: should let the tree fully grow
    # and then prune it automatically according to an optimal depth
    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_all_models = y_test_pred.copy()
    y_real_test = clf.predict(feats_test.values)

    y_result = pd.DataFrame({'TARGET_B': y_real_test})
    cn = pd.DataFrame(control)
    final_result = pd.concat([cn, y_result], axis=1)
    final_result.to_csv('result1.csv', sep='\t', encoding='utf-8')

    # Confusion Matrix
    print(pd.crosstab(
        y_test, y_test_pred, rownames=['actual'], colnames=['preds']))

    # Gets performance
    perf_model1 = Performance.get_perf(y_test.values, y_test_pred)
    print(perf_model1);

    #### Model 2 | Random Forest Model ####

    print("Model 2 executing...")

    # Training
    clf = ExtraTreesClassifier(n_estimators=500, verbose=1,
                               bootstrap=True, max_depth=30, oob_score=True, n_jobs=-1)

    # clf = RandomForestClassifier(
    #    n_estimators = 500, max_depth = 10, verbose = 1, n_jobs = -1)

    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_real_test = clf.predict(feats_test.values)
    y_all_models += y_test_pred

    y_result = pd.DataFrame({'TARGET_B': y_real_test})
    cn = pd.DataFrame(control)
    final_result = pd.concat([cn, y_result], axis=1)
    final_result.to_csv('result2.csv', sep='\t', encoding='utf-8')

    # Confusion Matrix
    print(pd.crosstab(
        y_test, y_test_pred, rownames=['actual'], colnames=['preds']))

    # Gets performance
    perf_model2 = Performance.get_perf(y_test, y_test_pred)
    print(perf_model2)

    #### Model 3 | Logistic Regression Model ####

    print("Model 3 executing...")

    # Training
    clf = LogisticRegression(max_iter=200, verbose=1)
    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_all_models += y_test_pred

    y_real_test = clf.predict(feats_test.values)

    y_result = pd.DataFrame({'TARGET_B': y_real_test})
    cn = pd.DataFrame(control)
    final_result = pd.concat([cn, y_result], axis=1)
    final_result.to_csv('result3.csv', sep='\t', encoding='utf-8')

    # Confusion Matrix
    print(pd.crosstab(
        y_test, y_test_pred, rownames=['actual'], colnames=['preds']))

    # Gets performance
    perf_model3 = Performance.get_perf(y_test, y_test_pred)

    #### Model 4 | Ensemble Model (majority vote for model 1, 2 and 3) ####

    print("Model 4 executing...")

    # Gets performance for an ensemble of all 3 models
    y_test_pred = np.array([0] * len(y_all_models))
    y_test_pred[y_all_models > 1] = 1
    perf_model_ensemble = Performance.get_perf(y_test, y_test_pred)

    y_real_test = clf.predict(feats_test.values)

    y_result = pd.DataFrame({'TARGET_B': y_real_test})
    cn = pd.DataFrame(control)
    final_result = pd.concat([cn, y_result], axis=1)
    final_result.to_csv('result4.csv', sep='\t', encoding='utf-8')

    # Confusion Matrix
    print(pd.crosstab(
        y_test, y_test_pred, rownames=['actual'], colnames=['preds']))

    #### Model comparison ####

    all_models = {'Decision Trees Model': perf_model1,
                  'Random Forest Model': perf_model2,
                  'Logistic Regression Model': perf_model3,
                  'Ensemble Model': perf_model_ensemble}

    perf_all_models = pd.DataFrame([[col1, col2, col3 * 100] for col1, d in
                                    all_models.items() for col2, col3 in d.items()], index=None,
                                   columns=['Model Name', 'Performance Metric', 'Value'])

    print(perf_all_models)





if __name__ == '__main__':
    main()
