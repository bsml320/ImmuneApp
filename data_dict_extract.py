import os, re, sys
import numpy as np
import pandas as pd

def data_dict_extract(file):
    pep_length = [8,9,10,11,12,13]
    data_dict = {}
    f = open(file,"r")
    for line in f:
        info = re.split(",|\t",line)
        allele = info[0].strip()
        pep = info[1].strip()
        if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
            continue    
        if len(pep) not in pep_length:
            continue 
        new_data = pep
        if allele not in data_dict.keys() :
            data_dict[allele] = [new_data]
        else:
            data_dict[allele].append(new_data)
    return data_dict