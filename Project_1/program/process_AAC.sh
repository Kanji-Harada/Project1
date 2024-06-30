#!/bin/bash

species=md
outsuffix=1
kfold=5
seqwin=35

# esm2_dict_file�̃p�X���w��
#esm2_dict_file=/home/kurata/myproject/common/esm2_enc/il13_seq2esm2_dict.pkl

# �g�p����@�B�w�K�A���S���Y���ƃG���R�[�h���@���w��
machine_method="LGBM" #XGB RF SVM NB KN LR"
encode_method="AAC"
#encode_method_w=${encode_method}

total_num=21 # ���v���f����

# �v���W�F�N�g�̃��[�g�p�X��ݒ�
cd ..
main_path=`pwd`
echo ${main_path}

# �f�[�^�Z�b�g�̃p�X��ݒ�
test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv

# �@�B�w�K�̃g���[�j���O�ƃe�X�g�����s
cd program
cd ml

train_path=${main_path}/data/dataset/cross_val
result_path=${main_path}/data/result_${species}
#esm2_dict=${esm2_dict_file}

for machine_method in ${machine_method}
do
    for encode_method in ${encode_method}
    do
    kmer=1
    echo ${machine_method} ${encode_method}
    python ml_train_test_AAC.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method} --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} #--kmer ${kmer} --esm2 ${esm2_dict}
    done
done

cd ..
#cd network

# �[�w�w�K�̃g���[�j���O�ƃe�X�g�����s
# �g�p����[�w�w�K�A���S���Y�����w��
#machine_method_2="TX CNN bLSTM"

# for deep_method in ${machine_method}
# do
#     encode_method="AAC"
#     echo ${deep_method}: ${encode_method}
#     kmer=1
#     if [ $deep_method = TX ]; then 
#         size=128
#     else
#         size=1280
#     fi
#     epochs=-1
#     window=-1
#     sg=-1
#     w2v_model=None
#     w2v_bpe_model=None
#     bpe_model=None
#     #esm2_dict=${esm2_dict_file}

#     python train_test_AAC.py --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method} --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} #--kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} #--esm2 ${esm2_dict}
# done
#cd ..

# �A���T���u���w�K�̎��s
echo evaluation
python analysis_AAC.py --machine_method "${machine_method}" --encode_method "${encode_method}" --species ${species} 

# outfile=result_${outsuffix}.xlsx
# python csv_xlsx_1_AAC.py --machine_method "${machine_method}" --encode_method "${encode_method}" --species ${species} --outfile ${outfile}

# echo ensemble
# meta=LR
# prefix=seq_impdec23_combine
# python ml_fusion_AAC.py --machine_method "${machine_method}" --encode_method "${encode_method}" --species ${species} --total_num ${total_num} --meta ${meta} --prefix ${prefix}

# outfile=result_stack_${outsuffix}.xlsx
# python csv_xlsx_2_AAC.py --species ${species} --outfile ${outfile} --meta ${meta} --prefix ${prefix}