#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse
from valid_metrices_p22 import * #評価メトリクスを計算するための関数をインポート

def measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold, threshold=None):  
    for i in range(kfold):
        infile = inpath + '/' + str(i+1) + '/' + val_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)
        prob = result[:, 1]
        label = result[:, 2]
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(prob, label)
        valid_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_

        score_val.iloc[i, 0] = th_
        score_val.iloc[i, 1] = rec_
        score_val.iloc[i, 2] = spe_
        score_val.iloc[i, 3] = pre_
        score_val.iloc[i, 4] = acc_
        score_val.iloc[i, 5] = mcc_
        score_val.iloc[i, 6] = f1_
        score_val.iloc[i, 7] = auc_
        score_val.iloc[i, 8] = prauc_

        infile = inpath + '/' + str(i+1) + '/' + test_file
        result = np.loadtxt(infile, delimiter=',', skiprows=1)
        prob = result[:, 1]
        label = result[:, 2]
        
        th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, prob, label)
        test_matrices = th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, prauc_
            
        score_test.iloc[i, 0] = th_
        score_test.iloc[i, 1] = rec_
        score_test.iloc[i, 2] = spe_
        score_test.iloc[i, 3] = pre_
        score_test.iloc[i, 4] = acc_
        score_test.iloc[i, 5] = mcc_
        score_test.iloc[i, 6] = f1_
        score_test.iloc[i, 7] = auc_
        score_test.iloc[i, 8] = prauc_
        
    means = score_val.astype(float).mean(axis=0)
    means = pd.DataFrame(np.array(means).reshape(1, -1), index=['means'], columns=columns_measure)
    score_val = pd.concat([score_val, means])

    means = score_test.astype(float).mean(axis=0)
    means = pd.DataFrame(np.array(means).reshape(1, -1), index=['means'], columns=columns_measure)
    score_test = pd.concat([score_test, means])
    
    return score_val, score_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str, help='Machine learning methods to be evaluated')
    parser.add_argument('--encode_method', type=str, help='Encoding methods to be evaluated')
    parser.add_argument('--species', type=str, help='Species name for the dataset')
    args = parser.parse_args()

    machine_methods = args.machine_method.strip().split()
    encode_methods = args.encode_method.strip().split()
    species = args.species
    kfold = 5

    data_path = '../data/result_%s' % species
    test_file = 'test_roc.csv'
    val_file = 'val_roc.csv'
    val_measure = 'val_measures.csv'
    test_measure = 'test_measures.csv'

    index_fold = [i+1 for i in range(kfold)]
    columns_measure = ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

    for encode_method in encode_methods:
        for machine_method in machine_methods:
            inpath = data_path + '/' + machine_method + '/' + encode_method
            outpath = data_path + '/' + machine_method

            score_val = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
            score_test = pd.DataFrame(data=[], index=index_fold, columns=columns_measure)

            score_val, score_test = measure_evaluation(score_val, score_test, inpath, val_file, test_file, kfold, threshold=None)
            
            score_val.to_csv('%s/val_measures.csv' % inpath, header=True, index=True)
            score_test.to_csv('%s/test_measures.csv' % inpath, header=True, index=True)

            print(score_val)
            print()
            print(score_test)
