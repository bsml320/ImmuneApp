import os, re, sys
import numpy as np
import pandas as pd

def read_peplist(peplist_file):
    try:
        fp = open(peplist_file)
    except IOError:
        exit()
    else:
        sample_peptides = {}
        fp = open(peplist_file)
        lines = fp.readlines()
        peplist = []
        for line in lines:
            peplist.append(line.strip())
        sample_peptides['peplist'] = peplist 
    return sample_peptides  