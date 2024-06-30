#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from valid_metrices_p22 import *

measure_column = ['Thre', 'Rec', 'Spe', 'Pre', 'Acc', 'MCC', 'F1', 'AUC', 'PRAUC']

def combine_model(train_data, valid_data, test_data, data_path, out_path, kfold, ml_list_label, combination, meta_class):  
    train_y, train_X = train_data[:, -1], train_data[:, :-1]
    valid_y, valid_X = valid_data[:, -1], valid_data[:, :-1]
    test_y, test_X = test_data[:, -1], test_data[:, :-1]

    train_result = np.zeros((len(train_y), 2))
    train_result[:, 1] = train_y            
    cv_result = np.zeros((len(valid_y), 2))
    cv_result[:, 1] = valid_y
    test_result = np.zeros((len(test_y), 2))
    test_result[:, 1] = test_y

    if meta_class == 'RF':
        model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
        clf = model.fit(train_X, train_y)
    elif meta_class == 'NB':
        model = GaussianNB()
        clf = model.fit(train_X, train_y)
    elif meta_class == 'KN':
        model = KNeighborsClassifier()
        clf = model.fit(train_X, train_y)
    elif meta_class == 'LR':
        model = LogisticRegression(random_state=0)
        clf = model.fit(train_X, train_y)
        os.makedirs('%s/%s/%s' % (data_path, out_path, kfold), exist_ok=True)
        pickle.dump(clf, open('%s/%s/%s' % (data_path, out_path, kfold) + "/lr_model", 'wb'))
    elif meta_class == 'SVM':    
        model = svm.SVC(probability=True)
        clf = model.fit(train_X, train_y)
    elif meta_class == 'XGB':
        xgb_train = xgb.DMatrix(train_X, train_y)
        xgb_eval = xgb.DMatrix(valid_X, valid_y)
        params = {
            "learning_rate": 0.01,
            "max_depth": 3
        }
        clf = xgb.train(params, xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")], 
                        num_boost_round=100, early_stopping_rounds=20)
    elif meta_class == 'LGBM':   
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)
        params = {         
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'random_state': 123,
        }
        clf = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                        num_boost_round=1000,
                        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)])
    else:
        print('No learning method')
        exit()

    if meta_class == 'LGBM':
        score = clf.predict(valid_X, num_iteration=clf.best_iteration)
        cv_result[:, 0] = score
    elif meta_class == 'XGB':  
        score = clf.predict(xgb_eval)
        cv_result[:, 0] = score
    else:
        score = clf.predict_proba(valid_X)
        cv_result[:, 0] = score[:, 1]

    if len(test_y) != 0:
        if meta_class == 'LGBM':
            test_result[:, 0] = clf.predict(test_X, num_iteration=clf.best_iteration)
        elif meta_class == 'XGB': 
            test_result[:, 0] = clf.predict(xgb.DMatrix(test_X))
        else:
            test_result[:, 0] = clf.predict_proba(test_X)[:, 1]

    valid_probs = cv_result[:, 0]
    valid_labels = cv_result[:, 1]
    test_probs = test_result[:, 0]
    test_labels = test_result[:, 1]

    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(valid_probs, valid_labels) 
    valid_matrices = th_, rec_, spe_, pre_, acc_, mcc_, f1_, auc_, prauc_
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, spe_, pre_, acc_, mcc_, f1_, auc_, prauc_

    df = pd.DataFrame([valid_matrices, test_matrices], index=['valid', 'test'], columns=measure_column)

    if meta_class == "LR":
        weight = [clf.intercept_[0]] + [clf.coef_[0, i] for i in range(clf.coef_.shape[1])]
    elif meta_class == "RF":
        weight = clf.feature_importances_
    elif meta_class == "LGBM":
        weight = clf.feature_importance()
    elif meta_class == "XGB":
        weight = clf.get_fscore()
    else:
        weight = -1
        
    return df, weight


def train_test(kfold, data_path, out_path, combination, ml_list_label, meta_class):
    train_data = []
    valid_data = []
    test_data = []
    for comb in combination:
        machine = comb[0]
        fea = comb[1]
        for datype in ['train', 'val', 'test']:
            fea_file = data_path + '/%s/%s/%s/%s_roc.csv' % (machine, fea, str(kfold), datype)
            fea_data = pd.read_csv(fea_file)
            if datype == 'train':
                train_data.append(fea_data['prob'].values.tolist())
                train_data.append(fea_data['label'].values.tolist())                
            elif datype == 'val':
                valid_data.append(fea_data['prob'].values.tolist())
                valid_data.append(fea_data['label'].values.tolist())
            elif datype == 'test':
                test_data.append(fea_data['prob'].values.tolist())
                test_data.append(fea_data['label'].values.tolist())
            else:
                pass

    train_data = np.array(train_data).T                    
    valid_data = np.array(valid_data).T
    test_data = np.array(test_data).T    

    train_data = np.delete(train_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    valid_data = np.delete(valid_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    test_data = np.delete(test_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)

    df, weight = combine_model(train_data, valid_data, test_data, data_path, out_path, kfold, ml_list_label, combination, meta_class)
    return df, weight
    

def ranking(measure_path, machine_methods, encode_methods):
    model_measure_column = ['Machine', 'Encode', 'Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']
    infile_name = ["val_measures.csv", "test_measures.csv"]

    val_measure = []
    for machine_method in machine_methods:
        for encode_method in encode_methods:
            infile_path = measure_path + "/%s/%s" % (machine_method, encode_method)
            infile1 = infile_path + '/' + infile_name[0]
            val_measure.append([machine_method, encode_method] + (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist()))

    df_val_measure = pd.DataFrame(data=val_measure, columns=model_measure_column)
    df_val_measure = df_val_measure[df_val_measure['Accuracy'] > 0.902]
    df_val_measure_sort = df_val_measure.sort_values('AUC', ascending=False)
    val_measure = df_val_measure_sort.values.tolist()
    
    combination = [[line[0], line[1]] for line in val_measure]
    return combination

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str, help='Machine learning methods')
    parser.add_argument('--encode_method', type=str, help='Encoding methods')
    parser.add_argument('--total_num', type=int, help='Total number of combinations to consider')
    parser.add_argument('--species', type=str, help='Species for the dataset')
    parser.add_argument('--meta', type=str, help='Meta classifier to use')
    parser.add_argument('--prefix', type=str, help='Prefix for output paths')
    args = parser.parse_args()
    
    machine_methods = args.machine_method.strip().split()
    encode_methods = args.encode_method.strip().split()
    meta_class = args.meta
    
    species = args.species
    total_number = args.total_num
    prefix = args.prefix
    kfold = 5
    data_path = "../data/result_%s" % species 
    df_all = pd.DataFrame(columns=measure_column) 
    
    combination_rank = ranking(data_path, machine_methods, encode_methods)
    
    combination = [comb for comb in combination_rank if not (comb[0] == "LGBM" and comb[1] == "ESM2" or comb[0] == "LR" and comb[1] == "ESM2")]

    total_number = len(combination)
    ml_list = [combination[i][0] + '-' + combination[i][1] for i in range(0, total_number)]
        
    for top_number in range(total_number, 0, -1):
        ml_list_label = ml_list + ['label']
        
        top_combination = [ml_list[i].split('-') for i in range(0, top_number)]
        out_path = '%s_%s/top%s' % (prefix, meta_class, top_number)
        
        df_valid = pd.DataFrame(columns=measure_column) 
        df_test = pd.DataFrame(columns=measure_column)
        
        if meta_class == "LR":
            df_weight = pd.DataFrame(columns=["intercept"] + ml_list)
        else:
            df_weight = pd.DataFrame(columns=ml_list)
            
        for k in range(1, kfold + 1):
            df, weight = train_test(k, data_path, out_path, top_combination, ml_list_label, meta_class)
            df_valid.loc[str(k) + "_valid"] = df.loc['valid']
            df_test.loc[str(k) + "_test"] = df.loc['test']  
            df_weight.loc[str(k) + "_weight"] = weight
    
        df_cat = pd.DataFrame(columns=measure_column)
        df_cat.loc["mean_valid"] = df_valid.mean() 
        df_cat.loc["sd_valid"] = df_valid.std()
        df_cat.loc["mean_test"] = df_test.mean()
        df_cat.loc["sd_test"] = df_test.std()
        df_cat.to_csv('%s/%s/average_measure.csv' % (data_path, out_path))
        
        if meta_class == 'LR':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/lr_weight.csv' % (data_path, out_path))
            df_weight = df_weight.drop(columns="intercept").T
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)
        
        elif meta_class == 'RF':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/rf_weight.csv' % (data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)
            
        elif meta_class == 'LGBM':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/lgbm_weight.csv' % (data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)
            
        elif meta_class == 'XGB':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/xgb_weight.csv' % (data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)

        else:
            print("No meta classifier")
            exit()

        del_feature = df_weight.index.tolist()[-1]
        ml_list.remove(del_feature)
        df_all.loc[str(top_number) + '_test'] = df_test.mean()  
        df_all.loc[str(top_number) + '_valid'] = df_valid.mean()
        
    df_all = df_all.iloc[::-1]
    
    maxAUC = 0
    maxStack = 0
    for i in range(int(df_all.shape[0] / 2)):    
        if df_all.iloc[2 * i, 7] > maxAUC:
            maxAUC = df_all.iloc[2 * i, 7]
            maxStack = i + 1
     
    df_all.loc['max_%s_valid' % maxStack] = df_all.iloc[2 * maxStack - 2, :]
    df_all.loc['max_%s_test' % maxStack] = df_all.iloc[2 * maxStack - 1, :]

    df_all.to_csv('%s/%s_%s/top_measure.csv' % (data_path, prefix, meta_class))
    print(df_all)                 
    print(df_all.iloc[:, 7])
