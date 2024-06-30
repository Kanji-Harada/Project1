#!/bin/sh

#現在のディレクトリの取得と表示
main_path=`pwd`
echo ${main_path}

#infile1=${main_path}/data/dataset/il13_train.txt
#infile2=${main_path}/data/dataset/il13_test.txt（入力ファイルとして使う予定だった）

#入力ファイルと出力ファイルのパス設定
outfile1=${main_path}/data/dataset/il13_train.txt #トレーニングデータ
outfile2=${main_path}/data/dataset/il13_test.txt #テストデータ

test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv
#データセットディレクトリのパス
data_path=${main_path}/data/dataset
#クロスバリデーションの設定
kfold=5

#python data_standard.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2} 
python ${main_path}/train_division_1.py --infile1 ${outfile1} --datapath ${data_path} --kfold ${kfold} #トレーニングデータの分割
python ${main_path}/test_fasta.py --infile1 ${outfile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} #テストデータをファスタ形式とcsv形式のファイルに変換
