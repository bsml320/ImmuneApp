# -*- coding: utf-8 -*-
"""
@author: HXu8
"""

import os, re, sys
import numpy as np
import pandas as pd
import random
import argparse
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

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

np.random.seed(1234)

import matplotlib.pyplot as plt

import read_matrix
import pseudo_HLA_seq
import transfomer_binding_affinity_matrix 
import transfomer_binding_affinity_matrix_external 

parameters = {
        "batch_size": 64,
        "fc1_size1": 256,
        "fc1_size2": 64,
        "filter1": 128,
        "filter2": 128,
        "filtersize1": 2,
        "optimizer": "adam"
    }

aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     

path_dict = 'supporting_file' #change as your supporting_file filepath
pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()
blosum_matrix = read_matrix(path_dict + r'blosum50.txt', 0)
pseq_dict_matrix = pseudo_HLA_seq(pseq_dict, blosum_matrix)

def creat_binding_affinity_model(training_pep, training_mhc):
    filters1 = parameters['filter1']
    filters2 = parameters['filter2']
    kernel_size = parameters['filtersize1']
    fc1_size = parameters['fc1_size1']
    fc2_size = parameters['fc1_size2']
    fc3_size = 16

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
   
    out = Dense(1,activation = "sigmoid")(merge_2)
    model = Model(inputs=[inputs_1, inputs_2],outputs=out)  
    model.summary()  
    return model

def main_model_training_cross_validation():
    
    path_train = "binding_affinity_train1" #change as your binding_affinity training dataset
    data_dict = transfomer_binding_affinity_matrix(path_train, pseq_dict, pseq_dict_matrix)    
    
    path_external = "binding_affinity_train2" #change as your external binding_affinity training dataset
    external_dict = transfomer_binding_affinity_matrix_external(path_external, pseq_dict, pseq_dict_matrix)
    
    es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=5)

    folder = "your folder" #change as your saved folder
    if not os.path.isdir(folder):
        os.makedirs(folder)  

    #merge the training datasets
    for allele in sorted(data_dict.keys()):
        if allele in external_dict.keys():
            print (allele)
            data_dict[allele] = data_dict[allele] + external_dict[allele]
            unique_seq = []
            unique_data = []
            for dt in data_dict[allele]:
                if dt[4] not in unique_seq:
                    unique_data.append(dt)
                    unique_seq.append(dt[4])
            print ("unique", len(unique_data))
            data_dict[allele] = unique_data

    n_splits = 5
    training_data = []
    test_dicts= []
    cross_validation = KFold(n_splits = n_splits, random_state=42)  
    
    for split in range(n_splits):
        training_data.append([])
        test_dicts.append([])    
    
    for allele in data_dict.keys():
        allele_data = data_dict[allele]
        random.shuffle(allele_data)
        allele_data = np.array(allele_data)
        split = 0

        if len(allele_data)< 5:
            continue

        for training_indices, test_indices in cross_validation.split(allele_data):
            training_data[split].extend(allele_data[training_indices])
            test_dicts[split].extend(allele_data[test_indices])
            split += 1 

    for split in range(n_splits):        
        random.shuffle(training_data[split])

    allprobas_=np.array([]) 
    allylable=np.array([])

    for i_splits in range(n_splits):
        train_splits = training_data[i_splits]    
        [training_pep, training_mhc, training_target] = [[i[j] for i in train_splits] for j in range(3)]

        test_splits = test_dicts[i_splits]    
        [validation_pep, validation_mhc, validation_target] = [[i[j] for i in test_splits] for j in range(3)]

        mc = ModelCheckpoint(folder + '/model_%s.h5' % str(i_splits), monitor='val_mse', mode='min', verbose=1, save_best_only=True)
        model = creat_binding_affinity_model(training_pep, training_mhc)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        model.summary()

        model_json = model.to_json()
        with open(folder + "model_"+str(i_splits)+".json", "w") as json_file:
            json_file.write(model_json)

        model.fit([training_pep,training_mhc], 
                        training_target,
                        batch_size=128,
                        epochs = 500,
                        shuffle=True,
                        callbacks=[es, mc],
                        validation_data=([validation_pep,validation_mhc], validation_target),
                        verbose=1)
        
        saved_model = load_model(folder + '/model_%s.h5' % str(i_splits))
        probas_ = saved_model.predict([validation_pep,validation_mhc])  
        test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in validation_target]
        allprobas_ = np.append(allprobas_, probas_)           
        allylable = np.append(allylable, np.array(test_label))
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
