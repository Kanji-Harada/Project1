#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openpyxl as px
import pandas as pd
import argparse
import os

columns_measure = ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, help='Output file name')
    parser.add_argument('--species', type=str, help='Species')
    parser.add_argument('--meta', type=str, help='Meta classifier')
    parser.add_argument('--prefix', type=str, help='Prefix for output paths')
    args = parser.parse_args()

    species = args.species
    meta_class = args.meta
    outfile_name = args.outfile
    prefix = args.prefix
    
    input_dir = "%s_%s" % (prefix, meta_class)
    infile_name = ["val_measures.csv", "test_measures.csv"]

    comb_file = '../data/result_%s/%s/top_measure.csv' % (species, input_dir)
    pd_comb = pd.read_csv(comb_file)
      
    if os.path.exists(outfile_name):
        with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer: 
            pd_comb.to_excel(writer, sheet_name='%s_stack_%s' % (prefix, meta_class))
    else:
        with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='w') as writer: 
            pd_comb.to_excel(writer, sheet_name='%s_stack_%s' % (prefix, meta_class))

    # Selection of the best meta-model
    maxAUC = 0
    maxStack = 1
    for i in range(int(pd_comb.shape[0] / 2)):
        if pd_comb.loc[2 * i, 'AUC'] > maxAUC:
            maxAUC = pd_comb.loc[2 * i, 'AUC']
            maxStack = 2 * i

    with pd.ExcelWriter(outfile_name, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer: 
        pd_comb[maxStack:maxStack + 2].to_excel(writer, sheet_name='%s_top_%s' % (prefix, meta_class), index=True)
