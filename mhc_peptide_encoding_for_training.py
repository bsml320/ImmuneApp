import os, re, sys
import numpy as np
import pandas as pd
import random

def mhc_peptide_encoding_for_training(path, pseq_dict, pseq_dict_matrix, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    data_dict = {}
    pep_length = [8,9,10,11,12,13]
    m = 0
    f = open(path,"r")
    for line in f:
        info = re.split("\t",line)
        allele = info[1].strip()
        if allele in pseq_dict.keys():
            affinity = int(info[-1].strip())
            pep = info[0].strip()
            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                continue   
            if len(pep) not in pep_length:
                continue 
            pep_blosum = []
            for residue_index in range(13):
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(13):
                if 13 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20)) 
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 13 + residue_index]]])

            new_data = [pep_blosum, pseq_dict_matrix[allele], affinity]
            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)
            m = m + 1          
    return data_dict