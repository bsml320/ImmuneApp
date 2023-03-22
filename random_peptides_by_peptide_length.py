# -*- coding: utf-8 -*-
"""
@author: HXu8
"""

import re
import random
import numpy as np
import pandas as pd
from collections import Counter 

import random_pick
import read_proteome_uniprot

def random_peptides_by_peptide_length():  #randomly generate peptides from the proteome
    proteome = read_proteome_uniprot()
    iedb_csv = "eluted ligands or binding affinity datasets" #change as your training dataset filepath (only including positive)
    iedb_df = pd.read_csv(iedb_csv, sep=',', skiprows=0, low_memory=False, dtype=object)
    iedb_df = np.array(iedb_df)
    
    all_positive_peptide = list(set([p[0] for p in iedb_df]))
    
    data_dict = {}
    for i in range(len(iedb_df)):
        allele = iedb_df[i][4]        
        if allele not in data_dict.keys():
            data_dict[allele] = [iedb_df[i].tolist()]
        else:
            data_dict[allele].append(iedb_df[i].tolist())
    
    all_neg = []
    for allele in data_dict.keys():
        print(allele) 
        traing_data = data_dict[allele]
        all_length = [len(traing_data[j][0]) for j in range(len(traing_data))]      
        all_length_times = Counter(all_length)

        all_probabilities = []
        for kmer in [8,9,10,11,12,13]:
            try:              
                probabilities = all_length_times[kmer]
            except:
                probabilities = 0   
            
            all_probabilities.append(probabilities)
    
        pep_seq = []
        while len(pep_seq) < 10*len(traing_data): # you can change the fold number 
            length = random_pick([8,9,10,11,12,13],all_probabilities)  
            accession = random.choice(list(proteome.keys()))
            protein = proteome[accession]
            # protein = random.choice(list(proteome.values()))
            if len(protein) < length:
                    continue
            pep_start = random.randint(0, len(protein) - length)
            pep = protein[pep_start:pep_start + length]
            
            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                continue             
            if pep in all_positive_peptide:
                print('In positive peptides')
                continue
            if pep not in pep_seq:
                pep_seq.append([accession, pep])
    
        for k in pep_seq:
            all_neg.append([allele, k[0], k[1]])
            
    return all_neg

if __name__ == '__main__':
    
    all_neg = random_peptides_by_peptide_length()    
