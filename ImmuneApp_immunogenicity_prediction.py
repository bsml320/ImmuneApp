import sys, os, math, tempfile, datetime, time, copy, re
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from keras.models import model_from_json
from keras.layers import Input, Dense
from keras.models import Model 
common_aa = "ARNDCQEGHILKMFPSTWYV"
aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
######### change the path_dict path #########
path_dict = 'supporting_file/'
model_dict = 'models/'
pep_length = [8,9,10,11,12,13,14,15]
pseq_dict_blosum_matrix = np.load(path_dict + 'pseq_dict_blosum_matrix.npy', allow_pickle = True).item()
pseq_dict = list(pseq_dict_blosum_matrix.keys())
blosum_matrix = np.array([[0.5,  -0.2,  -0.1,  -0.2,  -0.1,  -0.1,  -0.1,  0.0,  -0.2,  -0.1,  -0.2,  -0.1,  -0.1,  -0.3,  -0.1,  0.1,  0.0,  -0.3,  -0.2,  0.0], 
[-0.2,  0.7,  -0.1,  -0.2,  -0.4,  0.1,  0.0,  -0.3,  0.0,  -0.4,  -0.3,  0.3,  -0.2,  -0.3,  -0.3,  -0.1,  -0.1,  -0.3,  -0.1,  -0.3], 
[-0.1,  -0.1,  0.7,  0.2,  -0.2,  0.0,  0.0,  0.0,  0.1,  -0.3,  -0.4,  0.0,  -0.2,  -0.4,  -0.2,  0.1,  0.0,  -0.4,  -0.2,  -0.3], 
[-0.2,  -0.2,  0.2,  0.8,  -0.4,  0.0,  0.2,  -0.1,  -0.1,  -0.4,  -0.4,  -0.1,  -0.4,  -0.5,  -0.1,  0.0,  -0.1,  -0.5,  -0.3,  -0.4],
[-0.1,  -0.4,  -0.2,  -0.4,  1.3,  -0.3,  -0.3,  -0.3,  -0.3,  -0.2,  -0.2,  -0.3,  -0.2,  -0.2,  -0.4,  -0.1,  -0.1,  -0.5,  -0.3,  -0.1],
[-0.1,  0.1,  0.0,  0.0,  -0.3,  0.7,  0.2,  -0.2,  0.1,  -0.3,  -0.2,  0.2,  0.0,  -0.4,  -0.1,  0.0,  -0.1,  -0.1,  -0.1,  -0.3],
[-0.1,  0.0,  0.0,  0.2,  -0.3,  0.2,  0.6,  -0.3,  0.0,  -0.4,  -0.3,  0.1,  -0.2,  -0.3,  -0.1,  -0.1,  -0.1,  -0.3,  -0.2,  -0.3],
[0.0,  -0.3,  0.0,  -0.1,  -0.3,  -0.2,  -0.3,  0.8,  -0.2,  -0.4,  -0.4,  -0.2,  -0.3,  -0.4,  -0.2,  0.0,  -0.2,  -0.3,  -0.3,  -0.4],
[-0.2,  0.0,  0.1,  -0.1,  -0.3,  0.1,  0.0,  -0.2,  1.0,  -0.4,  -0.3,  0.0,  -0.1,  -0.1,  -0.2,  -0.1,  -0.2,  -0.3,  0.2,  -0.4],
[-0.1,  -0.4,  -0.3,  -0.4,  -0.2,  -0.3,  -0.4,  -0.4,  -0.4,  0.5,  0.2,  -0.3,  0.2,  0.0,  -0.3,  -0.3,  -0.1,  -0.3,  -0.1,  0.4],
[-0.2,  -0.3,  -0.4,  -0.4,  -0.2,  -0.2,  -0.3,  -0.4,  -0.3,  0.2,  0.5,  -0.3,  0.3,  0.1,  -0.4,  -0.3,  -0.1,  -0.2,  -0.1,  0.1],
[-0.1,  0.3,  0.0,  -0.1,  -0.3,  0.2,  0.1,  -0.2,  0.0,  -0.3,  -0.3,  0.6,  -0.2,  -0.4,  -0.1,  0.0,  -0.1,  -0.3,  -0.2,  -0.3],
[-0.1,  -0.2,  -0.2,  -0.4,  -0.2,  0.0,  -0.2,  -0.3,  -0.1,  0.2,  0.3,  -0.2,  0.7,  0.0,  -0.3,  -0.2,  -0.1,  -0.1,  0.0,  0.1],
[-0.3,  -0.3,  -0.4,  -0.5,  -0.2,  -0.4,  -0.3,  -0.4,  -0.1,  0.0,  0.1,  -0.4,  0.0,  0.8,  -0.4,  -0.3,  -0.2,  0.1,  0.4,  -0.1],
[-0.1,  -0.3,  -0.2,  -0.1,  -0.4,  -0.1,  -0.1,  -0.2,  -0.2,  -0.3,  -0.4,  -0.1,  -0.3,  -0.4,  1.0,  -0.1,  -0.1,  -0.4,  -0.3,  -0.3],
[0.1,  -0.1,  0.1,  0.0,  -0.1,  0.0,  -0.1,  0.0,  -0.1,  -0.3,  -0.3,  0.0,  -0.2,  -0.3,  -0.1,  0.5,  0.2,  -0.4,  -0.2,  -0.2],
[0.0,  -0.1,  0.0,  -0.1,  -0.1,  -0.1,  -0.1,  -0.2,  -0.2,  -0.1,  -0.1,  -0.1,  -0.1,  -0.2,  -0.1,  0.2,  0.5,  -0.3,  -0.2,  0.0],
[-0.3,  -0.3,  -0.4,  -0.5,  -0.5,  -0.1,  -0.3,  -0.3,  -0.3,  -0.3,  -0.2,  -0.3,  -0.1,  0.1,  -0.4,  -0.4,  -0.3,  1.5,  0.2,  -0.3],
[-0.2,  -0.1,  -0.2,  -0.3,  -0.3,  -0.1,  -0.2,  -0.3,  0.2,  -0.1,  -0.1,  -0.2,  0.0,  0.4,  -0.3,  -0.2,  -0.2,  0.2,  0.8,  -0.1],
[0.0,  -0.3,  -0.3,  -0.4,  -0.1,  -0.3,  -0.3,  -0.4,  -0.4,  0.4,  0.1,  -0.3,  0.1,  -0.1,  -0.3,  -0.2,  0.0,  -0.3,  -0.1,  0.5]])

def ig_model(path_model):
    json_f = open(path_model + 'immunogenicity/model_immunogenicity.json', 'r')
    loaded_model_json = json_f.read()
    json_f.close()
    pan_ligand = model_from_json(loaded_model_json)
    pan_ligand.load_weights(path_model + 'immunogenicity/model_immunogenicity.h5') 
    return pan_ligand

def mhc_peptide_pair_parallel(allele, peptides, pseq_dict_matrix, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    data_dict = {}
    pep_length = [8,9,10,11,12,13,14,15]
    for line in peptides:
        if allele in pseq_dict:
            pep = line
            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                print('Illegal peptides')
                continue   
            if len(pep) not in pep_length:
                print('Illegal peptides')
                continue 
            pep_blosum = []
            for residue_index in range(15):
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(15):
                if 15 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20)) 
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 15 + residue_index]]])
            new_data = [pep_blosum, pseq_dict_matrix[allele], pep]
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
    return data_dict    

def immunogenicity_prediction(allele, peptides, model_immunogenicity):
    prediction_data = mhc_peptide_pair_parallel(allele, peptides, pseq_dict_blosum_matrix, blosum_matrix)
    all_prediction_data = []
    for allele in prediction_data.keys():
        allele_data = prediction_data[allele]
        allele_data = np.array(allele_data)
        [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
        mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  
        
        lignad_scores = model_immunogenicity.predict(mp_pairs1)

        for i_peptides in range(len(peplist)):
            all_prediction_data.append([allele, peplist[i_peptides],
             lignad_scores[i_peptides]])

    return all_prediction_data 

def mk_predictions_immunogenicity(allele_peptides):
    all_predictions = {}
    model_ligand = ig_model(model_dict)
    for allele, peptides in allele_peptides.items():  
        all_predictions[allele] = {pep: {} for pep in peptides}
        all_prediction_data = immunogenicity_prediction(allele, peptides, model_ligand)
        peptide_idx = 1
        el_score_idx = 2

        for line in all_prediction_data:
            peptide = line[peptide_idx]
            el_score = float(line[el_score_idx])
                
            all_predictions[allele][peptide] = {'Immunogenicity_score': el_score}
    return all_predictions

def mk_predictions_immunogenicity_datatable(samples, allele_peptides, sample_peptides1, all_predictions): # add all predictions to the binding_predictions DataTable
    binding_predictions: pd.DataFrame = pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                                                'Immunogenicity_score', 'Binder'])
    for sample in samples:
        rows = []
        for allele, peptides in allele_peptides.items():
            for pep in peptides:
                if pep not in sample_peptides1[sample]:
                    continue
                rows.append([sample,
                                pep,
                                allele,
                                all_predictions[allele][pep]['Immunogenicity_score']])
                
        binding_predictions = binding_predictions.append(
            pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                    'Immunogenicity_score'], data=rows),
            ignore_index=True)       

    binding_predictions['Immunogenicity_score'] = binding_predictions['Immunogenicity_score'].apply(lambda x: format(x, '.4%'))   
    
    with open(tmp_folder + 'ImmuneApp_Immunogenicity_predictions.tsv', 'w') as f:
        for sample in samples:
            peptides = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Peptide'].unique())
            alleles = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Allele'].unique())
            keys = list(all_predictions[alleles[0]][peptides[0]].keys())
            header = ['Allele', 'Peptide', 'Sample'] + keys
            f.write('\t'.join(header) + '\n')
            for allele in alleles:
                for peptide in peptides:
                    keys = all_predictions[allele][peptide].keys()
                    to_write = [allele, peptide, sample] + [str(all_predictions[allele][peptide][k]) for k in keys]
                    f.write('\t'.join(to_write) + '\n') 
    
    return binding_predictions 

def read_peplist(peplist_file):
    try:
        fp = open(peplist_file)
    except IOError:
        exit()
    else:
        sample_peptides = {}
        fp = open(peplist_file)
        lines = fp.readlines()
        peplist = []
        for line in lines:
            peplist.append(line.strip())
        sample_peptides['peplist'] = peplist 
    return sample_peptides    

def main(args):  
    alleles = args.alleles
    peplist_file = args.peplist
    sample_peptides = read_peplist(peplist_file)

    sample_peptides1 = {}
    for sample, peptides in sample_peptides.items():
        sample_peptides1[sample] = list(set([p for p in peptides if 8 <= len(p) <= 15]))
    samples = list(sample_peptides1.keys())

    allele_peptides = {}
    for allele in alleles:
        allele_peps = []
        for sample in samples:
            allele_peps += sample_peptides1[sample]
        allele_peptides[allele] = list(set(allele_peps)) 

    all_predictions = mk_predictions_immunogenicity(allele_peptides)
    binding_predictions = mk_predictions_immunogenicity_datatable(samples, allele_peptides, sample_peptides1, all_predictions)
    
    print('ImmuneApp immunogenicity prediction done, Please check!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial intelligence-based epitope discovery and personalized tumor immune peptide analysis")
    parser.add_argument('-f', '--peplist', type=str, 
                        help='One file containing peptide list for prediction.')
    parser.add_argument('-a', '--alleles', type=str, nargs='+',
                        help='MHC alleles, spaces separated if more than one.')                       
    parser.add_argument('-o', '--output', type=str,
                        help='Output folder for the results.')     
    args = parser.parse_args()
    
    tmp_folder = args.output
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)  
    tmp_folder = args.output + "/"
    main(args)

    # python ImmuneApp_immunogenicity_prediction.py -f '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/test_immunogenicity.txt' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -o '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test'

