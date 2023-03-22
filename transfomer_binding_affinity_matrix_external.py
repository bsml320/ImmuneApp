import re
import numpy as np
from math import log

def read_external_train(path_external, pseq_dict, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    external_dict = {}
    f = open(path_external)
    invalid = 0
    valid = 0
    for line in f:
        info = re.split("\t", line[:-1])
        pep = info[0]
        affinity = 1-log(float(info[2]))/log(50000)
        allele = info[3][:8]+":"+info[3][-2:]
        if allele in pseq_dict.keys():
            valid += 1
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

            if allele not in external_dict.keys():
                external_dict[allele] = [new_data]
            else:
                external_dict[allele].append(new_data)
        else:
            invalid += 1
    print ("valid", valid, "invalid", invalid)
    
    return external_dict