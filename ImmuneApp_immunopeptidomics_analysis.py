## ImmuneApp(version 1.0) for antigen presentation prediction and immunopeptidome analysis.
## Through ImmuneApp, we hope to serve the community to push forward our understandings 
## of mechanism of T cell-mediated immunity and yield new insight in both personalized 
## immune treatment and development of targeted vaccines. 
## ImmuneApp also provides the online server, which is freely available at https://bioinfo.uth.edu/iapp/.
#############################################################################
import sys, os, math, tempfile, datetime, time, copy, re, argparse
import numpy as np
import pandas as pd
from os import PathLike
import shutil
from typing import Union, List, Tuple
import subprocess
from multiprocessing import Pool
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from keras.models import model_from_json
import logomaker
from scipy.stats import percentileofscore
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## import defined functions for prediction task and immunopeptidome analysis
#############################################################################
from read_matrix import read_matrix
from pseudo_HLA_seq import pseudo_HLA_seq
from model_import import import_model
from import_el_ba_model import import_ba_model, import_el_model
from scoring import scoring
from mhc_peptide_pair_parallel import mhc_peptide_pair_parallel
from run_multiple_jobs import *
from plot_motif_logomaker import plot_motif_logomaker
from correction_sample_name import correction_sample_name
from load_standard_file import load_standard_file
from filter_clean_peptide import *
from get_highest_binding import get_highest_binding
from load_immunopeptidome_files import load_immunopeptidome_files
from mk_folder import mk_folder
from merge_dicts import merge_dicts
from merge_dicts import merge_background_scores, merge_background_scores_K

# %%
## Load data that the model depends on for prediction
#############################################################################
common_aa = "ARNDCQEGHILKMFPSTWYV"
aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
path_dict = 'supporting_file/'
model_dict = 'models/'
dict_score_background = merge_background_scores_K(path_dict, '100K')
# dict_score_background = np.load(path_dict + 'dict_score_background.npy', allow_pickle=True).item()
print('Loading data from '+ path_dict)
blosum_matrix = read_matrix(path_dict + 'blosum50.txt', 0)
pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()
pseq_dict_blosum_matrix = pseudo_HLA_seq(pseq_dict, blosum_matrix) 

## Defined functions for prediction and analysis
#############################################################################
def get_final_rank(allele, list_predicted, dict_distr):
    list_out = list(map(lambda x: np.around(100 - percentileofscore(dict_distr[allele],x),4), list_predicted))
    return list_out  

def get_ranks_binder(row):
    strong_cutoff, weak_cutoff = 0.5, 2
    if float(row['Best_allele_rank']) <= strong_cutoff:
        binder = 'Strong'
    elif float(row['Best_allele_rank']) > strong_cutoff and float(row['Best_allele_rank']) <= weak_cutoff:
        binder = 'Weak'
    else:
        binder = 'Non-binder'
    return binder 

def mk_allele_peptides(sample_info, sample_peptides1):# get sets of peptides per allele
    alleles = []
    for sample in sample_info:
        alleles += [x.strip() for x in sample['sample-alleles'].split(',')]
    alleles = set(alleles)
    allele_peptides = {}
    for allele in alleles:
        allele_peps = []
        for sample in sample_info:
            if allele in sample['sample-alleles']:
                allele_peps += sample_peptides1[sample['sample-name']]
        allele_peptides[allele] = list(set(allele_peps)) 
    return allele_peptides  

def ligands_prediction(allele, peptides, model_ligands):
    # prediction_data = mhc_peptide_pair_parallel(allele, peptides, pseq_dict_blosum_matrix, blosum_matrix)
    prediction_data = mhc_peptide_pair_parallel(pseq_dict, allele, peptides, pseq_dict_blosum_matrix, blosum_matrix)
    all_prediction_data = []
    for allele in prediction_data.keys():
        allele_data = prediction_data[allele]
        allele_data = np.array(allele_data)
        [validation_pep, validation_mhc, peplist] = [[i[j] for i in allele_data] for j in range(3)] 
        mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]   
        scores = model_ligands.predict(mp_pairs1)
        rank = get_final_rank(allele, scores, dict_score_background)
        for i_peptides in range(len(peplist)):
            all_prediction_data.append([allele, peplist[i_peptides], scores[i_peptides], rank[i_peptides]])

    return all_prediction_data   

def mk_predictions(allele_peptides):
    all_predictions = {}
    model_ligand = import_el_model(model_dict)
    for allele, peptides in allele_peptides.items():  
        all_predictions[allele] = {pep: {} for pep in peptides}
        all_prediction_data = ligands_prediction(allele, peptides, model_ligand)
        peptide_idx = 1
        el_score_idx = 2
        el_rank_idx = 3
        strong_cutoff = 0.5
        weak_cutoff = 2
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
                
            all_predictions[allele][peptide] = {'el_rank': el_rank,
                                                    'el_score': el_score,
                                                    'binder': binder}
    return all_predictions

def mk_predictions_deconvolution_datatable(sample_info, allele_peptides, sample_peptides1, binding_predictions, all_predictions):
    global tmp_folder
    for sample in sample_info:
        rows = []
        for allele, peptides in allele_peptides.items():
            if allele not in sample['sample-alleles']:
                continue
            for pep in peptides:
                if pep not in sample_peptides1[sample['sample-name']]:
                    continue
                rows.append([sample['sample-name'],
                                pep,
                                allele,
                                all_predictions[allele][pep]['el_score'],
                                all_predictions[allele][pep]['el_rank'],
                                all_predictions[allele][pep]['binder']])
                
        binding_predictions = binding_predictions.append(
            pd.DataFrame(columns=['Sample', 'Peptide', 'Allele', 'Score', 'Rank', 'Binder'], data=rows),
            ignore_index=True)       

        samples = binding_predictions['Sample'].unique()
        for sample in samples:
            sample_decon_file = tmp_folder + f'{sample}_ImmuneApp_predictions.tsv'
            pivot_query = binding_predictions.loc[binding_predictions['Sample'] == sample, :]
            pivot = pivot_query.pivot(index=['Peptide'], columns='Allele', values=['Rank', 'Score']).astype(float)
            pivot.columns = [n[1]+ '_' +n[0] for n in pivot.columns.to_list()]
            pivot1 = pivot.reset_index()
            pivot2 = pivot1.filter(regex='_Rank$', axis=1)
            pivot1['Best_allele'] = pivot2.idxmin(axis=1)
            pivot1['Best_allele_rank'] = pivot2.min(axis=1)   
            pivot1['Best_allele'] = pivot1['Best_allele'].str[:-5]
            pivot1["Binder"] = pivot1.apply(lambda row: get_ranks_binder(row),axis=1)
            pivot1.to_csv(sample_decon_file, sep='\t', index=False)
            
    return binding_predictions

def mk_cluster_with_gibbscluster(GIBBSCLUSTER, samples, sample_peptides, not_enough_peptides):
    global tmp_folder
    jds = []
    min_length = 8
    max_length = 15
    os.chdir(tmp_folder)
    for sample in samples:
        fname = Path(tmp_folder, f'{sample}_forgibbs.csv')
        peps = np.array(cleaning_ligands(sample_peptides[sample]))
        if len(peps) < 20:
            not_enough_peptides.append(sample)
            continue
        lengths = np.vectorize(len)(peps)
        peps = peps[(lengths >= min_length) & (lengths <= max_length)]
        peps.tofile(str(fname), '\n', '%s')
        n_groups = 6  # search for up to 6 motifs
        for groups in range(1, n_groups+1):
            jds.append({"command" : f'{GIBBSCLUSTER} -f {fname} -P {groups}groups ' \
                                    f'-g {groups} -k 1 -T -j 2 -C -D 4 -I 1 -G'.split(' '),
                    "working_directory" : tmp_folder +'gibbscluster/' + sample + '/unsupervised',
                    "id" : f'gibbscluster_{groups}groups'})
    gibbs_result = run_multiple_processes(jds, 6)  

def mk_cluster_with_gibbscluster_by_allele(GIBBSCLUSTER, samples, sample_alleles, supervised_gibbs_directories, binding_predictions):
    global tmp_folder
    jds = []
    min_length = 8
    max_length = 15
    not_enough_peptides = []
    os.chdir(tmp_folder)
    # current_path = os.path.abspath(os.path.dirname(__file__))
    for sample in samples:
        alleles = sample_alleles[sample]
        supervised_gibbs_directories[sample] = {}
        sample_peps = binding_predictions.loc[binding_predictions['Sample'] == sample, :]
        allele_peps = {}
        for allele in alleles:
            allele_peps[allele] = set(list(sample_peps.loc[(sample_peps['Allele'] == allele) &
                                                            (sample_peps['Binder'] == 'Strong'), 'Peptide'].unique()))

        allele_peps['unannotated'] = set(list(sample_peps['Peptide']))
        for allele in alleles:
            allele_peps['unannotated'] = allele_peps['unannotated'] - allele_peps[allele]

        for allele, peps in allele_peps.items():
            fname = Path(tmp_folder, f"{allele}_{sample}_forgibbs.csv")
            peps = np.array(list(allele_peps[allele]))
            if len(peps) < 20:
                not_enough_peptides.append(f'{allele}_{sample}')
            else:
                lengths = np.vectorize(len)(peps)
                peps = peps[(lengths >= min_length) & (lengths <= max_length)]

                peps.tofile(str(fname), '\n', '%s')

                n_groups = 2 if allele == 'unannotated' else 1
                for groups in range(1, n_groups+1):
                    jds.append({"command" : f'{GIBBSCLUSTER} -f {fname} -P {groups}groups ' \
                                            f'-l {str(9)} -g {groups} -k 1 -T -j 2 -C -D 4 -I 1 -G'.split(' '),
                            "working_directory" : tmp_folder +'gibbscluster/' + sample + '/' + allele,
                            "id" : f'gibbscluster_{groups}groups'})
                    
    gibbs_result = run_multiple_processes(jds, 6)

def find_best_match(samples, sample_alleles, gibbs_files):
    global tmp_folder
    for sample in samples:
        gibbs_files[sample] = {}
        for run in ['unannotated', 'unsupervised']:
            sample_dirs = list(Path(tmp_folder + 'gibbscluster/' + sample + '/' + run).glob('*'))
            if len(sample_dirs) == 0:
                gibbs_files[sample][run] = None
                continue
            high_score = 0
            best_grouping = ''
            best_n_motifs = 0
            best_grouping_dir = ''
            for grouping in sample_dirs:
                with open(str(grouping) + '/images' + '/gibbs.KLDvsClusters.tab', 'r') as f:
                    klds = np.array(f.readlines()[1].strip().split()[1:], dtype=float)
                    score = np.sum(klds)
                    n_motifs = np.sum(klds != 0)
                if score > high_score:
                    best_grouping = grouping.stem[0]
                    best_grouping_dir = Path(grouping)
                    high_score = score
                    best_n_motifs = n_motifs

            if best_grouping == '':
                gibbs_files[sample][run] = None
                continue
            gibbs_files[sample][run] = {}
            gibbs_files[sample][run]['n_groups'] = best_grouping
            gibbs_files[sample][run]['directory'] = best_grouping_dir
            gibbs_files[sample][run]['n_motifs'] = best_n_motifs
            gibbs_files[sample][run]['cores'] = [x for x in list(Path(str(best_grouping_dir) + '/cores').glob('*'))
                                                                    if 'of' in x.name]
            gibbs_files[sample][run]['pep_groups_file'] = str(best_grouping_dir) + '/res/' + f'gibbs.{best_grouping}g.ds.out'

    for sample in samples:
        for allele in sample_alleles[sample]:
            gibbs_files[sample][allele] = {}
            ls = list(Path(tmp_folder + 'gibbscluster/' + sample + '/' + allele).glob('*'))
            if len(ls) == 0:
                gibbs_files[sample][allele] = None
                continue

            gibbs_files[sample][allele]['n_groups'] = '1'
            gibbs_files[sample][allele]['directory'] = list(Path(tmp_folder + 'gibbscluster/' + sample + '/' + allele).glob('*'))[0]
            gibbs_files[sample][allele]['n_motifs'] = 1
            gibbs_files[sample][allele]['cores'] = \
                str(gibbs_files[sample][allele]['directory']) + '/cores/' + 'gibbs.1of1.core'
            gibbs_files[sample][allele]['pep_groups_file'] = \
                str(gibbs_files[sample][allele]['directory']) + '/res/' + f'gibbs.1g.ds.out'

    return gibbs_files

def mk_pred_and_write_metrics_datatable(samples, sample_alleles, binding_predictions, original_peptides, sample_peptides1):
    global tmp_folder
    preds = binding_predictions.drop_duplicates()
    peptide_numbers = {}
    for sample in samples:
        peptide_numbers[sample] = {}
        peptide_numbers[sample]['original_total'] = len(set(original_peptides[sample]))
        peptide_numbers[sample]['within_length'] = len(set(sample_peptides1[sample]))
        for allele in sample_alleles[sample]:
            peptide_numbers[sample][allele] = {}
            for strength in ['Strong', 'Weak', 'Non-binder']:
                peptide_numbers[sample][allele][strength] = len(
                    preds.loc[(preds['Sample'] == sample) &
                                    (preds['Allele'] == allele) &
                                    (preds['Binder'] == strength), 'Peptide'].unique())

    all_sample_binder_data = []
    for sample in samples:
        Total_peptides = peptide_numbers[sample]['within_length']
        for allele in sample_alleles[sample]:
            ratio = [round(peptide_numbers[sample][allele][strength] * 100 / Total_peptides, 1) for strength in ['Strong', 'Weak', 'Non-binder']]
            row_data = [sample, Total_peptides, allele] + [peptide_numbers[sample][allele][strength] for strength in ['Strong', 'Weak', 'Non-binder']] + ratio
            all_sample_binder_data.append(row_data)

    with open(str(tmp_folder + 'sample_annotation_results.txt'), 'w') as f:
        f.write(f'sample\ttotal peptides\tallele\tstrong binders\tweak binders\tnon-binders\tstrong binders(%)\tweak binders(%)\tnon-binders(%)\n')
        for data in all_sample_binder_data:
            f.write('\t'.join(map(str, data)) + '\n')      

    for sample in samples:      
        pivot = preds.loc[preds['Sample'] == sample, :].pivot(index='Peptide', columns='Allele', values='Score').astype(float)
        pivot[pivot < 0.5] = 0
        data = pivot.sort_values(list(pivot.columns), ascending=False)
        data.to_csv(tmp_folder + "%s_heatmap.txt" % sample, index=True,float_format='%g')    

        figsize=8, 8
        figure_h, ax_h = plt.subplots(figsize=figsize)
        labels = ax_h.get_xticklabels() + ax_h.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels] 
        ax_h = sns.heatmap(data, cmap="OrRd", ax=ax_h, yticklabels=[])
        figure_h.savefig(tmp_folder + "%s_deconvolution_heatmap.png" % sample, dpi=300, bbox_inches = 'tight')  

    metrics = {}
    min_len = 8
    max_len = 15
    metrics['acceptable_length_key'] = f'n_peptides_{min_len}-{max_len}_mers'
    min_length = 8
    max_length = 15
    for sample in samples:
        all_peps = list(set(original_peptides[sample]))  # all peptides
        n_all_peps = len(all_peps)  # number of peptides in original list
        lengths = np.vectorize(len)(all_peps)  # lengths of those peptides
        length_peptides, counts = np.unique(lengths, return_counts=True)
        length_peptides_counts = {length: count for length, count in zip(list(length_peptides), (list(counts)))}
        to_write = [str(k) + ':' + str(length_peptides_counts[k]) for k in list(length_peptides_counts.keys())] 

        n_with_acceptable_length = np.sum(
            (lengths >= min_length) & (lengths <= max_length))

        counts_df = preds.loc[preds['Sample'] == sample, :]
        counts_df = counts_df.pivot(index='Peptide', columns='Allele', values='Binder')
        bindings = counts_df.apply(get_highest_binding, axis=1).values
        binders, counts = np.unique(bindings, return_counts=True)
        binder_counts = {binder: count for binder, count in zip(list(binders), (list(counts)))}
        binder_name = ['Strong', 'Weak', 'Non-binder']

        SWN_counts = [counts[list(binders).index('Strong')] if 'Strong' in binders else 0,
                    counts[list(binders).index('Weak')] if 'Weak' in binders else 0,
                    counts[list(binders).index('Non-binding')] if 'Non-binding' in binders else 0]
        binders = ['Strong', 'Weak', 'Non-binder']
        to_write1 = [binder_name[k] + ':' + str(SWN_counts[k]) for k in range(len(binder_name))] 

        n_binders = n_with_acceptable_length - binder_counts['Non-binding']
        lf_score = round(n_with_acceptable_length / n_all_peps, 2)
        bf_score = round(n_binders / n_with_acceptable_length, 2)

        metrics[sample] = {}
        metrics[sample]['n_peptides'] = n_all_peps
        metrics[sample][metrics['acceptable_length_key']] = n_with_acceptable_length
        metrics[sample]['lf_score'] = lf_score
        metrics[sample]['bf_score'] = bf_score
        metrics[sample]['pep_distribution'] = ','.join(to_write)
        metrics[sample]['types_distribution'] = ','.join(to_write1)

    with open(str(tmp_folder + 'sample_metrics.txt'), 'w') as f:
        f.write(f'sample\tn_peptides\tn_peptides_{min_len}-{max_len}_mers\tlength_score\tbinding_score\tlengths\tbinder_counts\n')
        for sample in samples:
            f.write(f'{sample}\t')
            f.write(f"{metrics[sample]['n_peptides']}\t")
            f.write(f"{metrics[sample][f'n_peptides_{min_len}-{max_len}_mers']}\t")
            f.write(f"{metrics[sample]['lf_score']}\t")
            f.write(f"{metrics[sample]['bf_score']}\t")
            f.write(f"{metrics[sample]['pep_distribution']}\t")
            f.write(f"{metrics[sample]['types_distribution']}\n")
    return preds 

def mk_sequence_genetate_unsupervised_logos_and_decomposition(samples, sample_alleles, preds, gibbs_files):
    global tmp_folder
    gibbs_peps = {}
    # get the peptides in each group
    for sample in samples:
        if gibbs_files[sample]['unsupervised'] is not None:
            pep_groups_file = gibbs_files[sample]['unsupervised']['pep_groups_file']
            if not Path(pep_groups_file).exists():
                raise FileNotFoundError(f'The GibbsCluster output file {pep_groups_file} does not exist.')
            with open(pep_groups_file, 'r') as f:
                pep_lines = f.readlines()[1:]
            n_motifs = gibbs_files[sample]['unsupervised']['n_groups']
            gibbs_peps[sample] = {str(x): [] for x in range(1, int(n_motifs)+1)}  # <---- there is a problem here because gibbcluster can find weird groups (like 2of2 but no 1of2) so we need to account for this somehow

            for line in pep_lines:
                line = [x for x in line.split(' ') if x != '']
                group = str(int(line[1]) + 1)
                pep = line[3]
                gibbs_peps[sample][group].append(pep)  # will contain the peptides belonging to each group

    for sample in samples:
        logo_dir = tmp_folder + 'gibbscluster/' + sample + '/' + 'unsupervised_logos/'
        if not os.path.exists(logo_dir):
            os.makedirs(logo_dir)
        if gibbs_files[sample]['unsupervised'] is not None:
            cores = gibbs_files[sample]['unsupervised']['cores']
            if not isinstance(cores, list):
                cores = [cores]
            for core in cores:
                cluster_name = logo_dir + str(core).split('/')[-1].split('of')[0] + '.core.png'
                f2 = open(str(core),"r")
                lines = f2.readlines()
                seqs = [seq.strip().upper() for seq in lines]
                plot_motif_logomaker(cluster_name, seqs)

    pep_binding_dict = {}
    for sample in samples:
        counts_df = preds.loc[preds['Sample'] == sample, :]
        counts_df = counts_df.pivot(index='Peptide', columns='Allele', values='Binder')
        pep_binding_dict[sample] = counts_df.copy(deep=True)

    composition = {}       
    for sample in samples:
        if gibbs_files[sample]['unsupervised'] is not None:
            cores = gibbs_files[sample]['unsupervised']['cores']
            pep_groups = []
            for x in range(len(cores)):
                pep_groups.append(cores[x].name.replace('gibbs.', '')[0])
            p_df: pd.DataFrame = pep_binding_dict[sample]
            composition[sample] = {str(x): {} for x in range(1, len(pep_groups)+1)}  # <---- there is a problem here because gibbcluster can find weird groups (like 2of2 but no 1of2) so we need to account for this somehow

            for i in range(len(pep_groups)):
                g_peps = set(gibbs_peps[sample][pep_groups[i]])  # the set of peptides found in the group
                strong_binders = {allele: round(len(g_peps & set(p_df[p_df[allele] == "Strong"].index)) * 100 /
                                                len(g_peps)) for allele in sample_alleles[sample]}

                composition[sample][pep_groups[i]] = strong_binders

    for sample in samples:
        logo_dir = tmp_folder + 'gibbscluster/' + sample + '/' + 'unsupervised_logos/'
        with open(logo_dir + 'cluster_composition.tsv', 'w') as f:
            f.write(sample + '\n')
            for cluster in composition[sample].keys():
                cluster_composition = composition[sample][cluster]
                top_binder = np.max(list(cluster_composition.values()))
                keys = cluster_composition.keys()
                to_write = [cluster, str(top_binder)] + [k + ': ' +str(composition[sample][cluster][k]) for k in keys]
                f.write('\t'.join(to_write) + '\n') 

    for sample in samples:
        logo_dir = tmp_folder + 'gibbscluster/' + sample + '/' + 'unsupervised_logos/'

        all_cluster = []
        for cluster in composition[sample].keys():
            cluster_composition = composition[sample][cluster]
            keys = cluster_composition.keys()
            for k in keys:
                all_cluster.append([cluster, k[4:], cluster_composition[k]])

        all_cluster = pd.DataFrame(all_cluster)
        all_cluster.columns = ['Cluster', 'Allele', 'cluster_composition']
        
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

        for c in list(composition[sample].keys()):
            data = all_cluster.loc[all_cluster['Cluster'] == c, :] 
            figsize=(4, 3)
            figure, ax = plt.subplots(figsize=figsize)
            plt.tick_params(labelsize=8)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels] 
            sns.barplot(x="Allele", y="cluster_composition", data=data, palette='Set2')
            plt.xlabel('Allele',font1, fontsize=10) 
            plt.ylabel('Composition', font1, fontsize=10)
            plt.savefig(logo_dir + '/gibbs.%s.com.png' % c, dpi=300, bbox_inches = 'tight')

def mk_sequence_genetate_alleles_logos(samples, sample_alleles, gibbs_files):
    global tmp_folder
    gibbs_peps = {}
    for sample in samples:
        for allele in sample_alleles[sample] + ['unannotated']:
            if gibbs_files[sample][allele] is not None:
                pep_groups_file = gibbs_files[sample][allele]['pep_groups_file']
                with open(pep_groups_file, 'r') as f:
                    pep_lines = f.readlines()[1:]
                n_motifs = gibbs_files[sample][allele]['n_groups']
                gibbs_peps[f'{allele}_{sample}'] = {str(x): [] for x in range(1, int(n_motifs)+1)}
                for line in pep_lines:
                    line = [x for x in line.split(' ') if x != '']
                    group = str(int(line[1]) + 1)
                    pep = line[3]
                    gibbs_peps[f'{allele}_{sample}'][group].append(pep)

    for sample in samples:
        for allele in sample_alleles[sample] + ['unannotated']:
            if gibbs_files[sample][allele] is not None:
                cores = gibbs_files[sample][allele]['cores']
                if not isinstance(cores, list):
                    cores = [cores]
                logo_dir = tmp_folder + 'gibbscluster/' + sample + '/' + 'supervised_sequence_logos/' + allele + '/'
                if not os.path.exists(logo_dir):
                    os.makedirs(logo_dir)
                for core in cores:
                    cluster_name = logo_dir + str(core).split('/')[-1].split('of')[0] + '.core.png'
                    cluster_name_seqs = logo_dir + str(core).split('/')[-1].split('of')[0] + '.core.txt'
                    f2 = open(str(core),"r")
                    lines = f2.readlines()
                    seqs = [seq.strip().upper() for seq in lines]
                    plot_motif_logomaker(cluster_name, seqs)
                    with open(cluster_name_seqs, 'w') as f:
                        f.write(str(len(seqs)) + '\n')    

def main(args):
    min_length=8 
    max_length=15
    sample_info = []
    sample_peptides = {}

    if args.template:
        files_alleles = load_standard_file(args.template)
        for file in files_alleles:
            sample_name = correction_sample_name(Path(file['file']).name)
            sample_info.append({'sample-name': sample_name,
                                'sample-description': '',
                                'sample-alleles': ', '.join(file['alleles'])})
            sample_peptides[sample_name] = cleaning_ligands(load_immunopeptidome_files(file['file']))
    else:
        files = args.files
        alleles = args.alleles
        for idx, file in enumerate(files):
            sample_name = correction_sample_name(Path(file).name)
            sample_info.append({'sample-name': sample_name,
                                'sample-description': '',
                                'sample-alleles': alleles[idx]})
            sample_peptides[sample_name] = cleaning_ligands(load_immunopeptidome_files(file))

    print('Immunopeptidome datasets are being processed!')
    original_peptides = sample_peptides
    sample_peptides1 = {}
    for sample, peptides in sample_peptides.items():
        sample_peptides1[sample] = [p for p in peptides if min_length <= len(p) <= max_length]

    samples = list(sample_peptides1.keys())
    sample_alleles = {}
    binding_predictions: pd.DataFrame = pd.DataFrame(columns=['Sample', 'Peptide', 'Allele', 'Rank', 'Binder'])
    supervised_gibbs_directories = {}
    gibbs_files = {}
    not_enough_peptides = []

    # current_path = os.path.abspath(os.path.dirname(__file__))
    # GIBBSCLUSTER = current_path + '/gibbscluster/gibbscluster'
    GIBBSCLUSTER = '/public/home/hxu6/projects/HLA_Prediction/python_202309/python_202309_old_server/iapp_share/gibbscluster/gibbscluster'

    mk_folder(tmp_folder, sample_info, sample_alleles)
    allele_peptides = mk_allele_peptides(sample_info, sample_peptides1) 
    all_predictions = mk_predictions(allele_peptides)
    binding_predictions = mk_predictions_deconvolution_datatable(sample_info, allele_peptides, sample_peptides1, binding_predictions, all_predictions)                                                                    
    mk_cluster_with_gibbscluster(GIBBSCLUSTER, samples, sample_peptides, not_enough_peptides)
    print('Motif analysis, reconstruction and discovey are ongoing!')
    mk_cluster_with_gibbscluster_by_allele(GIBBSCLUSTER, samples, sample_alleles, supervised_gibbs_directories, binding_predictions)
    gibbs_files = find_best_match(samples, sample_alleles, gibbs_files)  
    preds = mk_pred_and_write_metrics_datatable(samples, sample_alleles, binding_predictions, original_peptides, sample_peptides1) 
    mk_sequence_genetate_unsupervised_logos_and_decomposition(samples, sample_alleles, preds, gibbs_files)
    mk_sequence_genetate_alleles_logos(samples, sample_alleles, gibbs_files)
    print('Immunopeptidome analysis done, please check!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Artificial intelligence-based epitope discovery and personalized tumor immune peptide analysis")
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True,
                        help='One or more files containing peptide lists to analyze.')
    parser.add_argument('-a', '--alleles', type=str, nargs='+',
                        help='MHC alleles, spaces separated if more than one.')
    parser.add_argument('-t', '--template', type=str,
                        help='A tab-separated file containing file paths and alleles. Can be used to process peptide lists '
                        'with different alleles at the same time. The first column must be the file paths, and the '
                        'respective alleles can be put in the following columns, up to 6 per sample.')
    parser.add_argument('-o', '--output', type=str,
                        help='Output folder for the results.')
    args = parser.parse_args()
    tmp_folder = args.output + "/"
    main(args)

# python ImmuneApp_immunopeptidomics_analysis.py -f /public/home/hxu6/projects/HLA_Prediction/python_202309/python_202309_old_server/iapp_share/testdata/Melanoma_tissue_sample_of_patient_5.txt /public/home/hxu6/projects/HLA_Prediction/python_202309/python_202309_old_server/iapp_share/testdata/Melanoma_tissue_sample_of_patient_8.txt -a HLA-A*01:01,HLA-A*25:01,HLA-B*08:01,HLA-B*18:01 HLA-A*01:01,HLA-A*03:01,HLA-B*07:02,HLA-B*08:01,HLA-C*07:02,HLA-C*07:01 -o /public/home/hxu6/projects/HLA_Prediction/python_202309/web_model/prediction_test

