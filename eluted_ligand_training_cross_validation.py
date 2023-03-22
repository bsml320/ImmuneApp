# -*- coding: utf-8 -*-
"""
@author: HXu8
"""
import os, re, sys
import numpy as np
import pandas as pd
import random
import argparse
import h5py
import scipy.io
import json

from math import log
from collections import defaultdict

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform, glorot_normal
from keras.models import Model
from keras.layers import *
from keras.layers.core import  Dense, Dropout, Permute, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import load_model
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

np.random.seed(1234)

import matplotlib.pyplot as plt

import read_matrix
import pseudo_HLA_seq
import mhc_peptide_encoding_for_training

aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     

path_dict = 'supporting_file' #change as your supporting_file filepath
pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()

parameters = {"batch_size": 256,
        "fc1_size1": 256,
        "fc1_size2": 128,
        "filter1": 128,
        "filter2": 128,
        "filtersize1": 5,
        "optimizer": "adam"}

blosum_matrix = read_matrix(path_dict + r'blosum50.txt', 0)
pseq_dict_matrix = pseudo_HLA_seq(pseq_dict, blosum_matrix)

def creat_ligand_model(training_pep, training_mhc):
    filters1 = parameters['filter1']
    filters2 = parameters['filter2']
    kernel_size = parameters['filtersize1']
    fc1_size = parameters['fc1_size1']
    fc2_size = parameters['fc1_size2']
    fc3_size = 64
    fc4_size = 32

    inputs_1 = Input(shape = (np.shape(training_pep[0])[0],20))
    inputs_2 = Input(shape = (np.shape(training_mhc[0])[0],20))

    pep_conv = Conv1D(filters1,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_1)
    pep_lstm_out = Bidirectional(LSTM(64, return_sequences=True), merge_mode = 'concat')(pep_conv)

    mhc_conv_1 = Conv1D(filters2,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_2)
    mhc_maxpool_1 = MaxPooling1D()(mhc_conv_1)
    mhc_lstm_out = Bidirectional(LSTM(64, return_sequences=True), merge_mode = 'concat')(mhc_maxpool_1)

    flat_pep_2 = Flatten()(pep_lstm_out)
    flat_mhc_2 = Flatten()(mhc_lstm_out)


    cat_2 = Concatenate()([flat_pep_2, flat_mhc_2])        
    fc1_2 = Dense(fc1_size,activation = "relu")(cat_2)
    fc2 = Dense(fc2_size,activation = "relu")(fc1_2)
    fc3 = Dense(fc3_size,activation = "relu")(fc2)

    mhc_attention_weights = Flatten()(TimeDistributed(Dense(1))(mhc_conv_1))
    pep_attention_weights = Flatten()(TimeDistributed(Dense(1))(pep_conv))
    mhc_attention_weights = Activation('softmax')(mhc_attention_weights)
    pep_attention_weights = Activation('softmax')(pep_attention_weights)        
    mhc_conv_permute = Permute((2,1))(mhc_conv_1)
    pep_conv_permute = Permute((2,1))(pep_conv)
    mhc_attention = Dot(-1)([mhc_conv_permute, mhc_attention_weights])
    pep_attention = Dot(-1)([pep_conv_permute, pep_attention_weights])
    
    merge_2 = Concatenate()([mhc_attention,pep_attention,fc3])
    fc4 = Dense(fc4_size,activation = "relu")(merge_2)

    out = Dense(1,activation = "sigmoid")(fc4)
    model = Model(inputs=[inputs_1, inputs_2],outputs=out)  
    return model

def main_model_training_cross_validation():

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)

    folder = "your folder" #change as your saved folder
    if not os.path.isdir(folder):
        os.makedirs(folder)  

    path_train = "your training dataset" #change as your training dataset
    data_train_dict = mhc_peptide_encoding_for_training(path_train, pseq_dict, pseq_dict_matrix, blosum_matrix)    
      
    training_data = []
    for allele in data_train_dict.keys():
        allele_data = data_train_dict[allele]
        random.shuffle(allele_data)
        allele_data = np.array(allele_data)
        training_data.extend(allele_data)
    
    [all_pep, all_mhc, all_target] = [[i[j] for i in training_data] for j in range(3)]
    all_pep = np.array(all_pep)
    all_mhc = np.array(all_mhc)
    all_target = np.array(all_target)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)    
    allprobas_=np.array([]) 
    allylable=np.array([])

    for i, (train, test) in enumerate(kfold.split(all_pep, all_target)):
        training_pep = all_pep[train]
        training_mhc = all_mhc[train]
        training_target = all_target[train]
        
        validation_pep = all_pep[test]
        validation_mhc = all_mhc[test]
        validation_target = all_target[test]

        mc = ModelCheckpoint(folder + '/model_%s.h5' % str(i), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        model = creat_ligand_model(training_pep, training_mhc)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model_json = model.to_json()
        with open(folder + "model_"+str(i)+".json", "w") as json_file:
            json_file.write(model_json)

        training_pep1 = training_pep.reshape(training_pep.shape[0],-1)
        ros = RandomOverSampler(random_state=12345)
        X_resampled, y_resampled = ros.fit_resample(training_pep1, training_target)
        X_resampled1 = np.reshape(X_resampled, (-1, training_pep.shape[1], training_pep.shape[2]))

        training_mhc1 = training_mhc.reshape(training_mhc.shape[0],-1)
        X_resampled, y_resampled = ros.fit_resample(training_mhc1, training_target)
        X_resampled2 = np.reshape(X_resampled, (-1, training_mhc.shape[1], training_mhc.shape[2]))

        model.fit([X_resampled1,X_resampled2], 
                        y_resampled,
                        batch_size=5000,
                        epochs = 50,
                        shuffle=True,
                        callbacks=[es, mc],
                        validation_data=([validation_pep,validation_mhc], validation_target),
                        verbose=1)
        
        saved_model = load_model(folder + '/model_%s.h5' % str(i))
        probas_ = saved_model.predict([np.array(validation_pep),np.array(validation_mhc)])
        allprobas_ = np.append(allprobas_, probas_)           
        allylable = np.append(allylable, validation_target)
        del model

    lable_probas=np.c_[allylable, allprobas_]                
    with open(folder + 'Evalution_lable_probas.txt',"w+") as f:
        for j in range(len(lable_probas)):           
            f.write(str(lable_probas[j][0]) + '\t')
            f.write(str(lable_probas[j][1]) + '\n')

    font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16}
    figsize=6.2, 6.2

    #ROC_figure
    figure1, ax1 = plt.subplots(figsize=figsize)
    ax1.tick_params(labelsize=18)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  

    fpr, tpr, thresholds = roc_curve(allylable, allprobas_)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    
    ax1.plot(fpr, tpr, color='b',
        label=r'Mean ROC (AUC = %0.4f)' % (roc_auc),
        lw=2, alpha=.8)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate', font1)
    ax1.set_ylabel('True Positive Rate', font1)
    # title1 = 'Cross Validated ROC Curve'
    # ax1.set_title(title1, font1)
    ax1.legend(loc="lower right")
    figure1.savefig(folder + '5_fold_roc.png', dpi=300, bbox_inches = 'tight')

    #PR_figure
    figure2, ax2 = plt.subplots(figsize=figsize)
    ax2.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 

    precision, recall, _ = precision_recall_curve(allylable, allprobas_)
    ax2.plot(recall, precision, color='b',
            label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(allylable, allprobas_)),
            lw=2, alpha=.8)

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', font1)
    ax2.set_ylabel('Precision', font1)
    # title2 = 'Cross Validated PR Curve'
    # ax2.set_title(title2, font1)
    ax2.legend(loc="lower left")
    figure2.savefig(folder + '5_fold_pr.png', dpi=300, bbox_inches = 'tight')

if __name__ == '__main__':

    main_model_training_cross_validation()