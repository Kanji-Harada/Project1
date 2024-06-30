#!/bin/sh

#���݂̃f�B���N�g���̎擾�ƕ\��
main_path=`pwd`
echo ${main_path}

#infile1=${main_path}/data/dataset/il13_train.txt
#infile2=${main_path}/data/dataset/il13_test.txt�i���̓t�@�C���Ƃ��Ďg���\�肾�����j

#���̓t�@�C���Əo�̓t�@�C���̃p�X�ݒ�
outfile1=${main_path}/data/dataset/il13_train.txt #�g���[�j���O�f�[�^
outfile2=${main_path}/data/dataset/il13_test.txt #�e�X�g�f�[�^

test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv
#�f�[�^�Z�b�g�f�B���N�g���̃p�X
data_path=${main_path}/data/dataset
#�N���X�o���f�[�V�����̐ݒ�
kfold=5

#python data_standard.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2} 
python ${main_path}/train_division_1.py --infile1 ${outfile1} --datapath ${data_path} --kfold ${kfold} #�g���[�j���O�f�[�^�̕���
python ${main_path}/test_fasta.py --infile1 ${outfile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} #�e�X�g�f�[�^���t�@�X�^�`����csv�`���̃t�@�C���ɕϊ�
