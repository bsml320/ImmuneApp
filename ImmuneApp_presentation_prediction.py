## ImmuneApp(version 1.0) for antigen presentation prediction and immunopeptidome analysis.
## Through ImmuneApp, we hope to serve the community to push forward our understandings 
## of mechanism of T cell-mediated immunity and yield new insight in both personalized 
## immune treatment and development of targeted vaccines. 
## ImmuneApp also provides the online server, which is freely available at https://bioinfo.uth.edu/iapp/.
#############################################################################
import sys, os, math, tempfile, datetime, time, copy, re, argparse
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.layers import Input, Dense
from keras.models import Model 
from scipy.stats import percentileofscore
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## import defined functions for data processing and encoding
#############################################################################
from read_matrix import read_matrix
from pseudo_HLA_seq import pseudo_HLA_seq
from transform_affinity import transform_affinity
from affinity_transform import affinity_transform
from model_import import import_model
from scoring import scoring
from scoring_all import scoring_all
from get_final_ligand_rank import get_final_ligand_rank
from get_final_binding_rank import get_final_binding_rank
from data_dict_extract import data_dict_extract
from read_peplist import read_peplist
from read_fasta_files import read_fasta
from sample_fasta_peptides import sample_fasta_peptides
from import_el_ba_model import import_ba_model, import_el_model, import_el_model_m, import_ba_model_m
from mhc_peptide_pair_parallel import mhc_peptide_pair_parallel
from merge_dicts import merge_dicts
from merge_dicts import merge_background_scores, merge_background_scores_K

## Load data that the model depends on for prediction
#############################################################################
common_aa = "ARNDCQEGHILKMFPSTWYV"
aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19} 
pep_length = [8,9,10,11,12,13,14,15]
path_dict = 'supporting_file/'
model_dict = 'models/'
# dict_score_background = merge_background_scores(path_dict, 'score')
# dict_score_background = np.load(path_dict + 'dict_score_background.npy', allow_pickle=True).item()
print('Loading data from '+ path_dict)
# tmp_folder = 'jobs/' #please creat and specify a new folder for to store analysis results
blosum_matrix = read_matrix(path_dict + 'blosum50.txt', 0)
pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()
pseq_dict_blosum_matrix = pseudo_HLA_seq(pseq_dict, blosum_matrix) 

## Defined functions for prediction
#############################################################################
def ligands_prediction(allele, peptides, mode=None):
    prediction_data = mhc_peptide_pair_parallel(pseq_dict, allele, peptides, pseq_dict_blosum_matrix, blosum_matrix)
    all_prediction_data = []
    
    if mode:
        model_ligand = import_el_model_m(model_dict, 'ligands', 5) 
        for allele in prediction_data.keys():
            allele_data = prediction_data[allele]
            allele_data = np.array(allele_data)
            [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
            mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  
            lignad_scores = scoring(model_ligand, mp_pairs1)
            lignad_rank = get_final_ligand_rank(allele, lignad_scores, dict_score_background)
            for i_peptides in range(len(peplist)):
                all_prediction_data.append([allele, peplist[i_peptides],
                lignad_scores[i_peptides], lignad_rank[i_peptides]])
    else:   
        model_ligands = import_el_model(model_dict) 
        for allele in prediction_data.keys():
            allele_data = prediction_data[allele]
            allele_data = np.array(allele_data)
            [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
            mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  
            lignad_scores = model_ligands.predict(mp_pairs1)
            lignad_rank = get_final_ligand_rank(allele, lignad_scores, dict_score_background)
            for i_peptides in range(len(peplist)):
                all_prediction_data.append([allele, peplist[i_peptides],
                lignad_scores[i_peptides], lignad_rank[i_peptides]])

    return all_prediction_data 

def all_prediction(allele, peptides, mode=None):
    prediction_data = mhc_peptide_pair_parallel(pseq_dict, allele, peptides, pseq_dict_blosum_matrix, blosum_matrix)
    all_prediction_data = []
    
    if mode:   
        model_ligands = import_el_model_m(model_dict, 'ligands', 5) 
        model_binding = import_ba_model_m(model_dict, 'binding', 5) 
        for allele in prediction_data.keys():
            allele_data = prediction_data[allele]
            allele_data = np.array(allele_data)
            [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
            mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  
            
            lignad_scores = scoring(model_ligands, mp_pairs1)
            lignad_rank = get_final_ligand_rank(allele, lignad_scores, dict_score_background)
            binding_scores = scoring(model_binding, mp_pairs1)
            binding_aff = list(map(affinity_transform, binding_scores))

            for i_peptides in range(len(peplist)):
                all_prediction_data.append([allele, peplist[i_peptides],
                lignad_scores[i_peptides], lignad_rank[i_peptides],
                binding_scores[i_peptides], binding_aff[i_peptides]])
    else:
        model_ligands = import_el_model(model_dict)
        model_binding = import_ba_model(model_dict) 
        for allele in prediction_data.keys():
            allele_data = prediction_data[allele]
            allele_data = np.array(allele_data)
            [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
            mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  
            
            lignad_scores = model_ligands.predict(mp_pairs1)
            lignad_rank = get_final_ligand_rank(allele, lignad_scores, dict_score_background)
            binding_scores = model_binding.predict(mp_pairs1)
            binding_aff = list(map(affinity_transform, binding_scores))

            for i_peptides in range(len(peplist)):
                all_prediction_data.append([allele, peplist[i_peptides],
                lignad_scores[i_peptides], lignad_rank[i_peptides],
                binding_scores[i_peptides], binding_aff[i_peptides]])
                
    return all_prediction_data 

def mk_predictions_el(allele_peptides, mode):
    all_predictions = {}
    for allele, peptides in allele_peptides.items():  
        all_predictions[allele] = {pep: {} for pep in peptides}
        all_prediction_data = ligands_prediction(allele, peptides, mode)
        peptide_idx = 1
        el_score_idx = 2
        el_rank_idx = 3
        strong_cutoff = 0.5 if not mode else 0.2
        weak_cutoff = 2 if not mode else 0.5
        for line in all_prediction_data:
            peptide = line[peptide_idx]
            el_rank = float(line[el_rank_idx])
            el_score = float(line[el_score_idx])
            
            if float(el_rank) <= strong_cutoff:
                binder = 'Strong'
            elif float(el_rank) > strong_cutoff and float(el_rank) <= weak_cutoff:
                binder = 'Weak'
            else:
                binder = 'Non-binder'
                
            all_predictions[allele][peptide] = {'El_rank': el_rank,
                                                'El_score': el_score,
                                                'Binder': binder}
    print('Prediction done')

    return all_predictions

def mk_predictions_all(allele_peptides, mode):
    all_predictions = {}
    for allele, peptides in allele_peptides.items():  
        all_predictions[allele] = {pep: {} for pep in peptides}
        all_prediction_data = all_prediction(allele, peptides, mode)
        peptide_idx = 1
        el_score_idx = 2
        el_rank_idx = 3
        aff_score_idx = 4
        aff_nM_idx = 5
        strong_cutoff = 0.5 if not mode else 0.1
        weak_cutoff = 2 if not mode else 0.5
        for line in all_prediction_data:
            peptide = line[peptide_idx]
            el_rank = float(line[el_rank_idx])
            el_score = float(line[el_score_idx])
            aff_score = float(line[aff_score_idx])
            aff_nM  = float(line[aff_nM_idx])
            
            if float(el_rank) <= strong_cutoff:
                binder = 'Strong'
            elif float(el_rank) > strong_cutoff and float(el_rank) <= weak_cutoff:
                binder = 'Weak'
            else:
                binder = 'Non-binder'
                
            all_predictions[allele][peptide] = {'El_rank': el_rank,
                                                'El_score': el_score,
                                                'Aff_score': aff_score,
                                                'Aff_nM': aff_nM,
                                                'Binder': binder}
    print('Prediction done')
    
    return all_predictions

def mk_predictions_all_datatable(samples, allele_peptides, sample_peptides1, all_predictions): # add all predictions to the binding_predictions dataTable
    binding_predictions: pd.DataFrame = pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                                                'Score_EL', 'Rank_EL',
                                                                'Score_Aff', 'Aff_nM',  'Binder'])
    for sample in samples:
        rows = []
        for allele, peptides in allele_peptides.items():
            for pep in peptides:
                if pep not in sample_peptides1[sample]:
                    continue
                rows.append([sample,
                                pep,
                                allele,
                                all_predictions[allele][pep]['El_score'],
                                all_predictions[allele][pep]['El_rank'],
                                all_predictions[allele][pep]['Aff_score'],
                                all_predictions[allele][pep]['Aff_nM'],
                                all_predictions[allele][pep]['Binder']])
        binding_predictions = binding_predictions.append(pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                    'Score_EL', 'Rank_EL',
                                    'Score_Aff', 'Aff_nM', 'Binder'], data=rows),
            ignore_index=True)       

    with open(tmp_folder + 'ImmuneApp_presentation_predictions.tsv', 'w') as f:
        m = 0
        for sample in samples:
            peptides = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Peptide'].unique())
            alleles = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Allele'].unique())
            keys = list(all_predictions[alleles[0]][peptides[0]].keys())
            header = ['Allele', 'Peptide', 'Sample'] + keys
            if m == 0:
                f.write('\t'.join(header) + '\n')
            for allele in alleles:
                for peptide in peptides:
                    keys = all_predictions[allele][peptide].keys()
                    to_write = [allele, peptide, sample] + [str(all_predictions[allele][peptide][k]) for k in keys]
                    f.write('\t'.join(to_write) + '\n') 
            m = m + 1
    return binding_predictions

def mk_predictions_el_datatable(samples, allele_peptides, sample_peptides1, all_predictions): # add all predictions to the binding_predictions DataTable
    binding_predictions: pd.DataFrame = pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                                                'Score_EL', 'Rank_EL', 'Binder'])
    for sample in samples:
        rows = []
        for allele, peptides in allele_peptides.items():
            for pep in peptides:
                if pep not in sample_peptides1[sample]:
                    continue
                rows.append([sample,
                                pep,
                                allele,
                                all_predictions[allele][pep]['El_score'],
                                all_predictions[allele][pep]['El_rank'],
                                all_predictions[allele][pep]['Binder']])
        binding_predictions = binding_predictions.append(
            pd.DataFrame(columns=['Sample', 'Peptide', 'Allele',
                                    'Score_EL', 'Rank_EL',
                                    'Binder'], data=rows),
            ignore_index=True)       

    with open(tmp_folder + 'ImmuneApp_presentation_predictions.tsv', 'w') as f:
        m = 0
        for sample in samples:
            peptides = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Peptide'].unique())
            alleles = list(binding_predictions.loc[binding_predictions['Sample'] == sample, 'Allele'].unique())
            keys = list(all_predictions[alleles[0]][peptides[0]].keys())
            header = ['Allele', 'Peptide', 'Sample'] + keys
            if m == 0:
                f.write('\t'.join(header) + '\n')
            for allele in alleles:
                for peptide in peptides:
                    keys = all_predictions[allele][peptide].keys()
                    to_write = [allele, peptide, sample] + [str(all_predictions[allele][peptide][k]) for k in keys]
                    f.write('\t'.join(to_write) + '\n') 
            m = m + 1
    return binding_predictions

def mk_pred_and_write_metrics_datatable(samples, sample_peptides1, allele_peptides, binding_predictions):
    preds = binding_predictions.drop_duplicates()
    peptide_numbers = {}
    for sample in samples:
        peptide_numbers[sample] = {}
        peptide_numbers[sample]['within_length'] = len(set(sample_peptides1[sample]))
        for allele in allele_peptides.keys():
            peptide_numbers[sample][allele] = {}
            for strength in ['Strong', 'Weak', 'Non-binder']:
                peptide_numbers[sample][allele][strength] = len(
                    preds.loc[(preds['Sample'] == sample) &
                                    (preds['Allele'] == allele) &
                                    (preds['Binder'] == strength), 'Peptide'].unique())

    all_sample_binder_data = []
    for sample in samples:
        Total_peptides = peptide_numbers[sample]['within_length']
        for allele in allele_peptides.keys():
            ratio = [round(peptide_numbers[sample][allele][strength] * 100 / Total_peptides, 1) for strength in ['Strong', 'Weak', 'Non-binder']]
            row_data = [sample, Total_peptides, allele] + [peptide_numbers[sample][allele][strength] for strength in ['Strong', 'Weak', 'Non-binder']] + ratio
            all_sample_binder_data.append(row_data)

    with open(str(tmp_folder + 'sample_annotation_results.txt'), 'w') as f:
        f.write(f'sample\ttotal peptides\tallele\tstrong binders\tweak binders\tnon-binders\tstrong binders(%)\tweak binders(%)\tnon-binders(%)\n')
        for data in all_sample_binder_data:
            f.write('\t'.join(map(str, data)) + '\n')    

def main(args):  
    alleles = args.alleles
    mode = args.mode
    if args.fastafile:
        peptide_lengths = args.peptide_lengths
        fasta_file = args.fastafile
        sequences = read_fasta(fasta_file)
        sample_peptides = sample_fasta_peptides(sequences, peptide_lengths)
    else:
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

    if args.BA:    
        all_predictions = mk_predictions_all(allele_peptides, mode)
        binding_predictions = mk_predictions_all_datatable(samples, allele_peptides, sample_peptides1, all_predictions)
        mk_pred_and_write_metrics_datatable(samples, sample_peptides1, allele_peptides, binding_predictions)        
    else:
        all_predictions = mk_predictions_el(allele_peptides, mode)
        binding_predictions = mk_predictions_el_datatable(samples, allele_peptides, sample_peptides1, all_predictions)
        mk_pred_and_write_metrics_datatable(samples, sample_peptides1, allele_peptides, binding_predictions)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial intelligence-based epitope discovery and personalized tumor immune peptide analysis")

    parser.add_argument('-f', '--peplist', type=str, 
                        help='One file containing peptide list for prediction.')
    parser.add_argument('-fa', '--fastafile', type=str,
                        help='One file containing protein with fasta format for prediction.')
    parser.add_argument('-a', '--alleles', type=str, nargs='+',
                        help='MHC alleles, spaces separated if more than one.')
    parser.add_argument('-l', '--peptide_lengths', type=int, default=[9,10], nargs='*',
                        help='Peptide lengths extracted for prediction from protein.')                        
    parser.add_argument('-b', '--BA', action="store_true",
                        help='Whether to include the binding affinity presentation.')  
    parser.add_argument('-m', '--mode', action="store_true",
                    help='Whether to use multiple models (up to 25 models) to make predictions.')  
    parser.add_argument('-o', '--output', type=str,
                        help='Output folder for the results.')     
    args = parser.parse_args()
    
    tmp_folder = args.output
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)  
    tmp_folder = args.output + "/"
    mode = args.mode
    dict_score_background = merge_background_scores_K(path_dict, '500K') if mode else merge_background_scores_K(path_dict, '100K')
    main(args)
    
# python ImmuneApp_presentation_prediction.py -f '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/test_peplist.txt' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -o '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test'
# python ImmuneApp_presentation_prediction.py -b -m -fa '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/test.fasta' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -o '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test'
# python ImmuneApp_presentation_prediction.py -b -m -f '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/test_peplist.txt' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -o '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test'
# python ImmuneApp_presentation_prediction.py -b -f '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/test_peplist.txt' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -o '/public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test'
