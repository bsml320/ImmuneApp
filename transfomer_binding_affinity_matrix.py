import re
import numpy as np
from math import log

def read_binding_data(path, pseq_dict, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
    data_dict = {}
    f = open(path,"r")
    for line in f:
        info = re.split("\t",line)
        allele = info[1]
        if allele in pseq_dict.keys():
            affinity = 1-log(float(info[5]))/log(50000)
            pep = info[3]
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
            new_data = [pep_blosum, pseq_dict[allele], affinity, len(pep), pep, allele]

            if allele not in data_dict.keys():
                data_dict[allele] = [new_data]
            else:
                data_dict[allele].append(new_data)

    return data_dict