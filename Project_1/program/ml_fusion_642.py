#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 24, 2024
@author: Kurata Laboratory
"""

import os
import pickle
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from valid_metrices_p22 import *

measure_column =['Thre','Rec','Spe','Pre','Acc','MCC','F1','AUC','PRAUC'] 

def combine_model(train_data, valid_data, test_data, data_path, out_path, kfold, ml_list_label, combination, machine_method):  
    # ml_list_label: ML-encoding score,...., label
    prediction_result_cv = []
    prediction_result_test = []
    
    train_y, train_X = train_data[:,-1], train_data[:,:-1]
    valid_y, valid_X = valid_data[:,-1], valid_data[:,:-1]
    
    # option # train and valid data combined
    #valid_y = np.concatenate([train_data[:,-1], valid_data[:,-1]]) 
    #valid_X = np.concatenate([train_data[:,:-1], valid_data[:,:-1]])
    
    test_y, test_X = test_data[:,-1], test_data[:,:-1]

    train_result = np.zeros((len(train_y), 2))
    train_result[:, 1] = train_y            
    cv_result = np.zeros((len(valid_y), 2))
    cv_result[:, 1] = valid_y
    test_result = np.zeros((len(test_y), 2))
    test_result[:,1] = test_y   #score:one of two, label
       
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
        os.makedirs('%s/%s/%s' % (data_path, out_path, kfold), exist_ok=True) # file output directory
        pickle.dump(clf, open('%s/%s/%s' % (data_path, out_path, kfold) + "/lr_model",'wb'))
        
        """
        if os.path.isfile('%s/%s/%s/lr_model' % (data_path, out_path, kfold)):
            rfc = pickle.load(open('%s/%s/%s/lr_model' % (data_path, out_path, kfold)), 'rb'))
        else :
            print('No %s model' %data_path)
        """
        
    elif meta_class == 'SVM':    
        model = svm.SVC(probability=True)
        clf = model.fit(train_X, train_y)

    elif meta_class == 'XGB':
        xgb_train = xgb.DMatrix(train_X, train_y ) #feature_names= ml_list_label[:-1]
        xgb_eval  = xgb.DMatrix(valid_X , valid_y)
        params = {
        "learning_rate": 0.01,
        "max_depth": 3
        }
        clf = xgb.train(params, 
        xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")], 
        num_boost_round=100, early_stopping_rounds=20)
     
    elif meta_class == 'LGBM':   
        lgb_train = lgb.Dataset(train_X , train_y)
        lgb_eval = lgb.Dataset(valid_X , valid_y, reference=lgb_train)
        params = {         
        'objective': 'binary',# 二値分類問題         
        'metric': 'auc',# AUC の最大化を目指す          
        'verbosity': -1,# Fatal の場合出力
        'random_state': 123,
        }
        clf = lgb.train(params, lgb_train, valid_sets=lgb_eval,
              #verbose_eval=50,  # 50イテレーション毎に学習結果出力
              num_boost_round=1000,  # 最大イテレーション回数指定
              #early_stopping_rounds=100
              callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
             ) 
    else:
        print('No learning method')
        exit()

    if meta_class == 'LGBM':
        #train_result[:, 0] = clf.predict(train_X, num_iteration=clf.best_iteration)        
        score = clf.predict(valid_X, num_iteration=clf.best_iteration)
        cv_result[:, 0] = score
    elif meta_class == 'XGB':  
        #train_result[:, 0] = clf.predict(xgb_train)
        score = clf.predict(xgb_eval)
        cv_result[:, 0] = score
    else :
        #train_result[:, 0] = clf.predict_proba(train_X)[:,1]
        score = clf.predict_proba(valid_X)
        cv_result[:, 0] = score[:,1]
            
    #independent test
    if len(test_y) != 0:
        if meta_class == 'LGBM':
            test_result[:, 0] = clf.predict(test_X, num_iteration=clf.best_iteration)
        elif meta_class == 'XGB': 
            test_result[:, 0] = clf.predict(xgb.DMatrix(test_X))
        else:
            test_result[:, 0] = clf.predict_proba(test_X)[:,1]
     
    #CV
    cv_output = pd.DataFrame(cv_result,  columns=['prob', 'label'] )
    #cv_output.to_csv('%s/%s/%s/val_roc.csv' % (data_path, out_path, kfold))  
    df_valid = pd.DataFrame(valid_data, columns = ml_list_label)
    #df_valid.to_csv('%s/%s/%s/val_roc_com.csv' % (data_path, out_path, kfold))
    
    #independent test     
    test_output = pd.DataFrame(test_result,  columns=['prob', 'label'] )
    #test_output.to_csv('%s/%s/%s/test_roc.csv' % (data_path, out_path, kfold)) 
    df_test = pd.DataFrame(test_data, columns = ml_list_label)
    #df_test.to_csv('%s/%s/%s/test_roc_com.csv' % (data_path, out_path, kfold))
    
    valid_probs = cv_output['prob'].to_numpy()
    valid_labels = cv_output['label'].to_numpy()
    test_probs = test_output['prob'].to_numpy()
    test_labels = test_output['label'].to_numpy()
    #print(f'validation: prob label {valid_probs} {valid_labels} ')
    
    # metrics calculation
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = eval_metrics(valid_probs, valid_labels) 
    valid_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_
    th_, rec_, pre_, f1_, spe_, acc_, mcc_, auc_, pred_class, prauc_ = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, rec_, spe_, pre_,  acc_, mcc_, f1_, auc_, prauc_

    #print_results(valid_matrices, test_matrices) 
    #print(f'valid_matrices {valid_matrices}')  
    
    df  = pd.DataFrame([valid_matrices, test_matrices], index=['valid','test'], columns=measure_column)

    if meta_class == "LR":
        weight = [clf.intercept_[0]] + [ clf.coef_[0,i] for i in range(clf.coef_.shape[1]) ] #list
    elif meta_class == "RF":
        weight = clf.feature_importances_  # ndarray
    elif meta_class == "LGBM":
        weight = clf.feature_importance()  # ndarray
    elif meta_class == "XGB":
        weight = clf.get_fscore()  # ndarray
    else :
        weight = -1
        
    return df, weight


def train_test(kfold, data_path, out_path, combination, ml_list_label, meta_class):
    #feature combine for each fold
    train_data = []
    valid_data =[]
    test_data =[]
    for comb in combination:
        machine = comb[0]
        fea = comb[1] #encoding
        for datype in ['train', 'val','test']:
                fea_file = data_path + '/%s/%s/%s/%s_roc.csv' %(machine, fea, str(kfold), datype)
                fea_data = pd.read_csv(fea_file)
                if datype == 'train':
                    train_data.append(fea_data['prob'].values.tolist())
                    train_data.append(fea_data['label'].values.tolist())                
                elif datype =='val':
                    valid_data.append(fea_data['prob'].values.tolist())
                    valid_data.append(fea_data['label'].values.tolist())
                elif datype =='test':
                    test_data.append(fea_data['prob'].values.tolist())
                    test_data.append(fea_data['label'].values.tolist())
                else:
                    pass
    train_data = np.array(train_data).T                    
    valid_data = np.array(valid_data).T
    test_data = np.array(test_data).T    

    # Redundant labels [label,prob,label, prob,....] are removed, but one label in the last column must be left
    train_data = np.delete(train_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    valid_data = np.delete(valid_data, [i for i in range(1, 2*len(combination)-1, 2)], 1)
    test_data  = np.delete(test_data, [i for i in  range(1, 2*len(combination)-1, 2)], 1) 

    # training and testing
    df, weight = combine_model(train_data, valid_data, test_data, data_path, out_path, kfold, ml_list_label, combination, meta_class)

    return df, weight
    

def ranking(measure_path, machine_method_1, encode_method_1, machine_method_2, encode_method_2):
    model_measure_column = ['Machine','Encode','Threshold', 'Sensitivity', 'Specificity', 'Precision','Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']
    #print(f'encode_method {encode_method_1}')
    infile_name = ["val_measures.csv", "test_measures.csv" ]

    val_measure  = []
    for machine_method in machine_method_1:
        for i, encode_method in enumerate(encode_method_1):
            infile_path = measure_path + "/%s/%s" %(machine_method, encode_method)       
            infile1 = infile_path + '/' + infile_name[0] #val
            #print(encode_method)
            #print(infile1)
            val_measure.append( [machine_method, encode_method] +  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means

    for machine_method in machine_method_2:
        for i, encode_method in enumerate(encode_method_2):
            infile_path = measure_path + "/%s/%s" %(machine_method, encode_method)
            infile1 = infile_path + '/' + infile_name[0] #val
            val_measure.append( [machine_method, encode_method] +  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means

    df_val_measure  = pd.DataFrame(data=val_measure, columns=model_measure_column)
    
    df_val_measure = df_val_measure[df_val_measure['Accuracy'] > 0.902] #72
    
    # sort
    df_val_measure_sort = df_val_measure.sort_values('AUC', ascending=False)
    
    val_measure = df_val_measure_sort.values.tolist()
    #print(val_measure)
    #df_val_measure_sort.to_csv('ranking.csv')
     
    combination=[]
    for line in val_measure:
        combination.append([line[0], line[1]])        
    
    return combination   

if __name__ == '__main__':   

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method_1', type=str, help='term')
    parser.add_argument('--encode_method_1', type=str, help='term')
    parser.add_argument('--machine_method_2', type=str, help='term')
    parser.add_argument('--encode_method_2', type=str, help='term')
    parser.add_argument('--total_num', type=int, help='term')
    parser.add_argument('--species', type=str, help='term')
    parser.add_argument('--meta', type=str, help='term')
    parser.add_argument('--prefix', type=str, help='term')
    args = parser.parse_args()
    
    machine_method_1 = args.machine_method_1.strip().split()
    encode_method_1  = args.encode_method_1.strip().split()
    machine_method_2 = args.machine_method_2.strip().split()
    encode_method_2  = args.encode_method_2.strip().split() 
    meta_class = args.meta
    
    species = args.species
    total_number  =  args.total_num
    prefix  =  args.prefix
    kfold=5
    data_path = "../data/result_%s" %species 
    df_all = pd.DataFrame(columns=measure_column) 
    
    # ML-encoding ranking list based on AUC
    # Ranking is conducted in AUC, we get all the combinations in the descending order of AUC.
    combination_rank = ranking(data_path, machine_method_1, encode_method_1, machine_method_2, encode_method_2) #based on AUC
    
    comb = []
    total_number=len(combination_rank)
    for i in range(total_number):  # delete LGBM-ESM2
        if combination_rank[i][0] == "LGBM" and combination_rank[i][1] == "ESM2" : #21
            pass
        #elif combination_rank[i][0] == "LGBM" and combination_rank[i][1] == "W2V_4_128_100_40_1 : #22
        #    pass                       
        elif combination_rank[i][0] == "LR" and combination_rank[i][1] == "ESM2" : #23
            pass  
        else:
            comb.append(combination_rank[i]) 
    combination = comb
    
    total_number=len(combination)
    print(len(combination))
    #print(combination)
    
    # DataFrame initial ml_list definition         
    ml_list = [combination[i][0]+'-'+combination[i][1] for i in range(0, total_number)] 
        
    for top_number in range(total_number, 0, -1):
        print(f'top_number: {top_number}')
        ml_list_label = ml_list + ['label']
        
        print(f'1 {ml_list}')
        top_combination = [] 
        for i in range(0, top_number):
            top_combination.append(ml_list[i].split('-'))
      
        out_path ='%s_%s/top%s'%(prefix, meta_class, top_number)
        
        #print(top_combination)
        df_train = pd.DataFrame(columns=measure_column) 
        df_valid = pd.DataFrame(columns=measure_column) 
        df_test = pd.DataFrame(columns=measure_column)
        
        if meta_class == "LR":
            df_weight = pd.DataFrame(columns= ["intercept"] + ml_list)
        else :
            df_weight = pd.DataFrame(columns= ml_list)
            
        for k in range(1, kfold+1):
            df, weight = train_test(k, data_path, out_path, top_combination, ml_list_label , meta_class)  # calculation weight or importance 
            df_valid.loc[str(k) + "_valid"] = df.loc['valid']
            df_test.loc[str(k) + "_test"] = df.loc['test']  
            df_weight.loc[str(k) + "_weight"] = weight  #LR, RF
    
        # prediction performance of top X
        df_cat = pd.DataFrame(columns=measure_column)
        df_cat.loc["mean_train"] = df_valid.mean() 
        df_cat.loc["sd_train"] = df_valid.std()
        df_cat.loc["mean_test"] = df_test.mean()
        df_cat.loc["sd_test"] = df_test.std()
        df_cat.to_csv('%s/%s/average_measure.csv' %(data_path, out_path)) 
        
        # weight or importance contribution of each ML-encoding
        if meta_class == 'LR' :
            df_weight.loc["mean_weight"] = df_weight.abs().mean()  # Absolute coefficient
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/lr_weight.csv' %(data_path, out_path)) 
            df_weight = df_weight.drop(columns="intercept").T
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)  # Sort according to the importance.
            print(df_weight)
        
        elif meta_class =='RF':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()  # Absolute importance
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/rf_weight.csv' %(data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)
            
        elif meta_class =='LGBM':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()  # Absolute importance
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/lgbm_weight.csv' %(data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)
            
        elif meta_class =='XGB':
            df_weight.loc["mean_weight"] = df_weight.abs().mean()  # Absolute importance
            df_weight.loc["sd_weight"] = df_weight.std()
            df_weight.to_csv('%s/%s/xgb_weight.csv' %(data_path, out_path))
            df_weight = df_weight.T      
            df_weight.sort_values(ascending=False, by="mean_weight", inplace=True)

        else :
            print("No meta classifier")
            exit()

        # Remove the method with the lowest weight or importance
        del_feature = df_weight.index.tolist()[-1]
        print(f"del_feature: {del_feature}")    
        ml_list.remove(del_feature)        
        print(f'2 {ml_list}')
        df_all.loc[str(top_number) + '_test'] = df_test.mean()  
        df_all.loc[str(top_number) + '_valid'] = df_valid.mean()
        
    df_all = df_all.iloc[::-1] #  reverse        
           
    #selection of the best meta-model
    maxAUC=0
    maxStack=0
    for i in range(int(df_all.shape[0]/2)):    
        if df_all.iloc[2*i,7] > maxAUC :  #"AUC":7
            maxAUC=df_all.iloc[2*i,7]
            maxStack=i+1
     
    df_all.loc['max_%s_valid'%maxStack] = df_all.iloc[2*maxStack-2, :]
    df_all.loc['max_%s_test'%maxStack ] = df_all.iloc[2*maxStack-1, :]

    df_all.to_csv('%s/%s_%s/top_measure.csv' %(data_path, prefix, meta_class))   
    print(df_all)                 
    print(df_all.iloc[:,7])    
    
    

  
    



    
    
    
       
