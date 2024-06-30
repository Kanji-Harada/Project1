#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse
import collections
import time
import pickle
import sklearn.metrics as metrics
import json
from sklearn.metrics import precision_recall_curve
from loss_func import CBLoss
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
from sklearn.model_selection import StratifiedKFold
from CNN_network import CNN
from LSTM_network_bidirectional import bLSTM
from TX_network import TX
from encodingAA_23 import AAC

metrics_dict = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy, "auc": auc, "recall": recall, "f1": f1, "AUPRC": AUPRC}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def file_input_csv(filename, index_col=None):
    data = pd.read_csv(filename, index_col=index_col)
    return data

def trim_input_csv(filename, seqwin, index_col=None):
    df1 = pd.read_csv(filename, index_col=index_col)
    seq = df1.loc[:, 'seq'].tolist()
    for i in range(len(seq)):
        if len(seq[i]) > seqwin:
            seq[i] = seq[i][0:seqwin]
    for i in range(len(seq)):
        df1.loc[i, 'seq'] = seq[i]
    return df1

class pv_data_sets():
    def __init__(self, data_sets, encode_method, seqwin):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist()
        self.labels = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"].values.tolist()), 1]).astype(np.float32)
        self.encode_method = encode_method
        self.seqwin = seqwin

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.encode_method == 'AAC':
            emb_mat = np.array(AAC([self.seq[idx]], order='ARNDCQEGHILKMFPSTWYV', type='Protein'), dtype=float)
        else:
            print('no encoding method')
            exit()

        return torch.tensor(emb_mat).float().to(device), torch.tensor(label).to(device)

class train_test_process():
    def __init__(self, out_path, loss_type="balanced", tra_batch_size=128, val_batch_size=128, test_batch_size=32, lr=0.001, n_epoch=10000, early_stop=25, thresh=0.5):
        self.out_path = out_path
        self.tra_batch_size = tra_batch_size
        self.val_batch_size = val_batch_size
        self.lr = lr
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.thresh = thresh
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def training_testing(self, train_data_sets, val_data_sets, test_data_sets, deep_method, encode_method, seqwin):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)

        tra_data_all = pv_data_sets(train_data_sets, encode_method, seqwin)
        train_loader = DataLoader(dataset=tra_data_all, batch_size=self.tra_batch_size, shuffle=True)

        val_data_all = pv_data_sets(val_data_sets, encode_method, seqwin)
        val_loader = DataLoader(dataset=val_data_all, batch_size=self.val_batch_size, shuffle=True)

        test_data_all = pv_data_sets(test_data_sets, encode_method, seqwin)
        test_loader = DataLoader(dataset=test_data_all, batch_size=32, shuffle=False)

        if deep_method == 'CNN':
            net = CNN(features=20, time_size=seqwin).to(device)
        elif deep_method == 'bLSTM':
            net = bLSTM(features=20, lstm_hidden_size=128).to(device)
        elif deep_method == 'TX':
            net = TX(n_layers=3, d_model=20, n_heads=4, d_dim=100, d_ff=400, time_seq=seqwin).to(device)
        else:
            print('no net exist')
            exit()

        opt = optim.Adam(params=net.parameters(), lr=self.lr)

        if self.loss_type == "balanced":
            criterion = nn.BCELoss()

        min_loss = 1000
        early_stop_count = 0
        with open(self.out_path + "/cv_result.txt", 'w') as f:
            print(self.out_path, file=f, flush=True)
            print("The number of training data:" + str(len(train_data_sets)), file=f, flush=True)
            print("The number of validation data:" + str(len(val_data_sets)), file=f, flush=True)

            for epoch in range(self.n_epoch):
                train_losses, val_losses = [], []
                self.train_probs, self.train_labels = [], []

                print("epoch_" + str(epoch + 1) + "=====================", file=f, flush=True)
                print("train...", file=f, flush=True)
                net.train()

                for i, (emb_mat, label) in enumerate(train_loader):
                    opt.zero_grad()
                    outputs = net(emb_mat)

                    if self.loss_type == "balanced":
                        loss = criterion(outputs, label)
                    elif self.loss_type == "imbalanced":
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    loss.backward()
                    opt.step()

                    train_losses.append(float(loss.item()))
                    self.train_probs.extend(outputs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    self.train_labels.extend(label.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())

                train_thresh = 0.5
                print("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), file=f, flush=True)
                print("train_loss:: value: %f, epoch: %d, time: %f" % (sum(train_losses) / len(train_losses), epoch + 1, time.time() - start))
                print("val_threshold:: value: %f, epoch: %d" % (train_thresh, epoch + 1), file=f, flush=True)
                for key in metrics_dict.keys():
                    if key != "auc" and key != "AUPRC":
                        metrics = metrics_dict[key](self.train_labels, self.train_probs, thresh=train_thresh)
                    else:
                        metrics = metrics_dict[key](self.train_labels, self.train_probs)
                    print("train_" + key + ": " + str(metrics), file=f, flush=True)

                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.train_labels, self.train_probs, thresh=train_thresh)
                print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file=f, flush=True)
                print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file=f, flush=True)
                print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file=f, flush=True)
                print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file=f, flush=True)

                print("validation...", file=f, flush=True)

                net.eval()
                self.val_probs, self.val_labels = [], []
                for i, (emb_mat, label) in enumerate(val_loader):
                    with torch.no_grad():
                        outputs = net(emb_mat)

                    if self.loss_type == "balanced":
                        loss = criterion(outputs, label)
                    elif self.loss_type == "imbalanced":
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    if np.isnan(loss.item()) == False:
                        val_losses.append(float(loss.item()))

                    self.val_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    self.val_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist())

                loss_epoch = sum(val_losses) / len(val_losses)

                val_thresh = 0.5

                print("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), file=f, flush=True)
                print("val_threshold:: value: %f, epoch: %d" % (val_thresh, epoch + 1), file=f, flush=True)
                for key in metrics_dict.keys():
                    if key != "auc" and key != "AUPRC":
                        metrics = metrics_dict[key](self.val_labels, self.val_probs, thresh=val_thresh)
                    else:
                        metrics = metrics_dict[key](self.val_labels, self.val_probs)
                    print("validation_" + key + ": " + str(metrics), file=f, flush=True)

                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.val_labels, self.val_probs, thresh=val_thresh)
                print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file=f, flush=True)
                print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file=f, flush=True)
                print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file=f, flush=True)
                print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file=f, flush=True)

                if loss_epoch < min_loss:
                    early_stop_count = 0
                    min_loss = loss_epoch
                    os.makedirs(self.out_path + "/data_model", exist_ok=True)
                    os.chdir(self.out_path + "/data_model")
                    torch.save(net.state_dict(), "deep_model")

                    final_thresh = 0.5
                    final_val_probs = self.val_probs
                    final_val_labels = self.val_labels
                    final_train_probs = self.train_probs
                    final_train_labels = self.train_labels

                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop:
                        print('Training can not improve from epoch {}\tBest loss: {}'.format(epoch + 1 - self.early_stop, min_loss), file=f, flush=True)
                        break

            for key in metrics_dict.keys():
                if key != "auc" and key != "AUPRC":
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs, thresh=final_thresh)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs, thresh=final_thresh)
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print("train_" + key + ": " + str(train_metrics), file=f, flush=True)
                print("val_" + key + ": " + str(val_metrics), file=f, flush=True)

        with open(self.out_path + "/test_result.txt", 'w') as f:
            print(self.out_path, file=f, flush=True)
            print("The number of testing data:" + str(len(test_data_sets)), file=f, flush=True)

            self.test_probs, self.test_labels = [], []

            print("testing...", file=f, flush=True)
            net.eval()

            for i, (emb_mat, label) in enumerate(test_loader):
                with torch.no_grad():
                    outputs = net(emb_mat)

                self.test_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                self.test_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist())

            for key in metrics_dict.keys():
                if key != "auc" and key != "AUPRC":
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs, thresh=self.thresh)
                else:
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs)
                print("test_" + key + ": " + str(test_metrics), file=f, flush=True)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.test_labels, self.test_probs, thresh=self.thresh)
            print("test_true_negative:: value: %f" % (tn_t), file=f, flush=True)
            print("test_false_positive:: value: %f" % (fp_t), file=f, flush=True)
            print("test_false_negative:: value: %f" % (fn_t), file=f, flush=True)
            print("test_true_positive:: value: %f" % (tp_t), file=f, flush=True)

###############################################################################################################
if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intrain', help='Path')
    parser.add_argument('--intest', help='Path')
    parser.add_argument('--outpath', help='Path')
    parser.add_argument('--losstype', help='Path', default="balanced", choices=["balanced", "imbalanced"])
    parser.add_argument('--deeplearn', help='Path')
    parser.add_argument('--encode', help='Path')
    parser.add_argument('--kfold', type=int, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')

    args = parser.parse_args()
    path = args.intrain
    test_path = args.intest
    out_path = args.outpath
    loss_type = args.losstype
    deep_method = args.deeplearn
    encode_method = args.encode
    kfold = args.kfold
    seqwin = args.seqwin

    if encode_method != 'AAC':
        print("Only AAC encoding method is supported in this script.")
        exit()

    os.makedirs(out_path + '/' + deep_method + '/' + encode_method, exist_ok=True)
    out_path = out_path + '/' + deep_method + '/' + encode_method

    for i in range(1, kfold + 1):
        train_dataset = trim_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col=None)
        val_dataset = trim_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col=None)
        test_dataset = trim_input_csv(test_path, seqwin, index_col=None)

        net = train_test_process(out_path + "/" + str(i), loss_type=loss_type)
        net.training_testing(train_dataset, val_dataset, test_dataset, deep_method, encode_method, seqwin)

        output = pd.DataFrame([net.train_probs, net.train_labels], index=["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/train_roc.csv")
        output = pd.DataFrame([net.val_probs, net.val_labels], index=["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/val_roc.csv")

        output = pd.DataFrame([net.test_probs, net.test_labels], index=["prob", "label"]).transpose()
        output.to_csv(out_path + "/" + str(i) + "/test_roc.csv")

    print('total time:', time.time() - start)
