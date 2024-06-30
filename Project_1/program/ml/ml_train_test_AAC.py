#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from encodingAA_23 import AAC

# dataset reading
def pad_input_csv(filename, seqwin, index_col=None):
    df1 = pd.read_csv(filename, delimiter=',', index_col=index_col)
    seq = df1.loc[:, 'seq'].tolist()
    for i in range(len(seq)):
        if len(seq[i]) > seqwin:
            seq[i] = seq[i][0:seqwin]
        seq[i] = seq[i].ljust(seqwin, 'X')
    for i in range(len(seq)):
        df1.loc[i, 'seq'] = seq[i]
    return df1

def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res

#############################################################################################
if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intrain', help='Path')
    parser.add_argument('--intest', help='Path')
    parser.add_argument('--outpath', help='Path')
    parser.add_argument('--machine', help='Path')
    parser.add_argument('--encode', help='Path')
    parser.add_argument('--kfold', type=int, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')
    
    args = parser.parse_args()
    path = args.intrain
    test_file = args.intest
    out_path_0 = args.outpath
    machine_method = args.machine
    encode_method = args.encode
    kfold = args.kfold
    seqwin = args.seqwin

    os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
    out_path = out_path_0 + '/' + machine_method + '/' + encode_method

    for i in range(1, kfold + 1):
        os.makedirs(out_path + "/" + str(i) + "/data_model", exist_ok=True)
        modelname = "machine_model.sav"

        train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col=None)
        val_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col=None)
        test_dataset = pad_input_csv(test_file, seqwin, index_col=None)

        train_seq = train_dataset['seq'].tolist()
        val_seq = val_dataset['seq'].tolist()
        test_seq = test_dataset['seq'].tolist()

        myOrder = 'ARNDCQEGHILKMFPSTWYV'
        kw = {'order': myOrder, 'type': 'Protein'}

        if encode_method == 'AAC':
            train_X = np.array(AAC(train_seq, **kw), dtype=float)
            valid_X = np.array(AAC(val_seq, **kw), dtype=float)
            test_X = np.array(AAC(test_seq, **kw), dtype=float)
        else:
            print('No encode method')
            exit()

        train_y = train_dataset['label'].to_numpy()
        valid_y = val_dataset['label'].to_numpy()
        test_y = test_dataset['label'].to_numpy()

        train_result = np.zeros((len(train_y), 2))
        train_result[:, 1] = train_y
        cv_result = np.zeros((len(valid_y), 2))
        cv_result[:, 1] = valid_y
        test_result = np.zeros((len(test_y), 2))
        test_result[:, 1] = test_y

        if machine_method == 'RF':
            model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
            clf = model.fit(train_X, train_y)
        elif machine_method == 'NB':
            model = GaussianNB()
            clf = model.fit(train_X, train_y)
        elif machine_method == 'KN':
            model = KNeighborsClassifier()
            clf = model.fit(train_X, train_y)
        elif machine_method == 'LR':
            model = LogisticRegression(random_state=0)
            clf = model.fit(train_X, train_y)
        elif machine_method == 'SVM':
            model = svm.SVC(probability=True)
            clf = model.fit(train_X, train_y)
        elif machine_method == 'XGB':
            xgb_train = xgb.DMatrix(train_X, train_y)
            xgb_eval = xgb.DMatrix(valid_X, valid_y)
            params = {
                "learning_rate": 0.01,
                "max_depth": 3
            }
            clf = xgb.train(params,
                            xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")],
                            num_boost_round=100, early_stopping_rounds=20)
        elif machine_method == 'LGBM':
            lgb_train = lgb.Dataset(train_X, train_y)
            lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'random_state': 123,
            }
            clf = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
                            )
        else:
            print('No learning method')
            exit()

        pickle.dump(clf, open(out_path + "/" + str(i) + "/data_model/machine_model", 'wb'))

        # CV
        if machine_method == 'LGBM':
            score = clf.predict(train_X, num_iteration=clf.best_iteration)
            train_result[:, 0] = score
            score = clf.predict(valid_X, num_iteration=clf.best_iteration)
            cv_result[:, 0] = score
        elif machine_method == 'XGB':
            score = clf.predict(xgb_train)
            train_result[:, 0] = score
            score = clf.predict(xgb_eval)
            cv_result[:, 0] = score
        else:
            score = clf.predict_proba(train_X)
            train_result[:, 0] = score[:, 1]
            score = clf.predict_proba(valid_X)
            cv_result[:, 0] = score[:, 1]

        # independent test
        if len(test_y) != 0:
            if machine_method == 'LGBM':
                test_result[:, 0] = clf.predict(test_X, num_iteration=clf.best_iteration)
            elif machine_method == 'XGB':
                test_result[:, 0] = clf.predict(xgb.DMatrix(test_X))
            else:
                test_result[:, 0] = clf.predict_proba(test_X)[:, 1]

        # CV
        train_output = pd.DataFrame(train_result, columns=['prob', 'label'])
        train_output.to_csv(out_path + "/" + str(i) + "/train_roc.csv")
        cv_output = pd.DataFrame(cv_result, columns=['prob', 'label'])
        cv_output.to_csv(out_path + "/" + str(i) + "/val_roc.csv")

        # independent test
        test_output = pd.DataFrame(test_result, columns=['prob', 'label'])
        test_output.to_csv(out_path + "/" + str(i) + "/test_roc.csv")

    print('elapsed time', time.time() - start)
