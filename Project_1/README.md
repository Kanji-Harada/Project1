# PredIL13

# Development environment
 >python 3.10.9   
 >anaconda 4.11.0  
 >pandas 1.2.5  
 >numpy 1.20.3  
 >lightgbm: 4.1.0  
 >xgboost: 1.7.6  
 >sklearn: 1.1.2  
 >gensim: 4.0.1  

# Execution
# 1 Setting directories
Users must keep the structure of the directories  

# 2 Construction of dataset
Before simulation, users build dataset files for cross validataion and independent test:   
seqwin(max_length)=35   
$sh data_const.sh
  
# 3 Baselin model construction and meta-classifier construction
seqwin(max_length)=35  
Note that users need to construct W2V_general_4_128_100_40_1.pt themselves or exclude it.  
We could not upload W2V_general_4_128_100_40_1.pt because its memory size is over 25 MB.
$cd program  
$sh process.sh  

## 3-1 Training and testing of the baseline models
./program/ml/ml_train_test_45.py  
./program/network/train_test_86.py  

## 3-2 Evaluation of the baseline models
analysis_622.py  

## 3-3 Meta-classifier construction
ml_fusion_642.py  

## 3-4 output of result
csv_xlsx_34.py  
csv_xlsx_37.py  

# References on RNA encodings
https://ilearn.erc.monash.edu/  

# History
from pa3/il13_24 in CERVO, mldl environment  

