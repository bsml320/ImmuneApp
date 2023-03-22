

import os, re, sys
import numpy as np
import pandas as pd

def pseudo_HLA_seq(seq_dict,blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
    pseq_dict = {}
    for allele in seq_dict.keys():
        new_pseq = []
        for index in range(34):
            new_pseq.append(blosum_matrix[aa[seq_dict[allele][index]]]) 
        pseq_dict[allele] = new_pseq
    return pseq_dict
