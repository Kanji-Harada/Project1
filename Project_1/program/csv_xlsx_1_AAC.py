#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openpyxl as px
import pandas as pd
import argparse
import os

columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_method', type=str, help='Machine learning methods to be evaluated')
    parser.add_argument('--encode_method', type=str, help='Encoding methods to be evaluated')
    parser.add_argument('--outfile', type=str, help='Output file name')
    parser.add_argument('--species', type=str, help='Species name for the dataset')
    args = parser.parse_args()
    
    machine_methods = args.machine_method.strip().split()
    encode_methods  = args.encode_method.strip().split()
    species = args.species
    outfile_name = args.outfile

    infile_name = ["val_measures.csv", "test_measures.csv"]

    for machine_method in machine_methods:
        val_measure = []
        test_measure = []
        for encode_method in encode_methods:
            infile_path = "../data/result_%s/%s/%s" % (species, machine_method, encode_method)
            infile1 = infile_path + '/' + infile_name[0]  # val
            infile2 = infile_path + '/' + infile_name[1]  # test

            val_measure.append((pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist()))  # means
            test_measure.append((pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist()))  # means

        pd_val_measure = pd.DataFrame(data=val_measure, index=encode_methods, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=encode_methods, columns=columns_measure)

        pd_val_test = pd.concat([pd_val_measure, pd_test_measure], axis=0)

        if os.path.exists(outfile_name):
            mode_f = 'a'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode=mode_f) as writer:
                pd_val_test.to_excel(writer, sheet_name=machine_method)
        else:
            mode_f = 'w'
            with pd.ExcelWriter(outfile_name, engine="openpyxl", mode=mode_f) as writer:
                pd_val_test.to_excel(writer, sheet_name=machine_method)

    # # Adding ranking sheet
    # rank_file = './ranking.csv'
    # pd_rank = pd.read_csv(rank_file)
    # with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a') as writer:
    #     pd_rank.to_excel(writer, sheet_name='ranking')