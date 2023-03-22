import os, re, sys
import numpy as np
import pandas as pd

def filter_non_standard_aa(peptide: str):
    if peptide[1] == '.':
        peptide = peptide[2:]
    if peptide[-2] == '.':
        peptide = peptide[:-2]
    return peptide

def cleaning_ligands(peptide_list, verbose=False):
    common_aa = "ARNDCQEGHILKMFPSTWYV"
    unmodified_peps = []
    if verbose:
        print('Removing peptide modifications')
    for pep in peptide_list:
        pep = filter_non_standard_aa(pep)
        pep = ''.join(re.findall('[a-zA-Z]+', pep))
        pep = pep.upper()
        incompatible_aa = False
        for aa in pep:
            if aa not in common_aa:
                incompatible_aa = True
                break
        if not incompatible_aa:
            unmodified_peps.append(pep)
    return unmodified_peps