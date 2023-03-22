import os, re, sys
import numpy as np
import pandas as pd

def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()
        fasta_dict = {} 
        gene_id = ""
        for line in lines:
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq
                seq = ""
                gene_id = line.strip() 
            else:
                seq += line.strip()        
        fasta_dict[gene_id] = seq 
    return fasta_dict  