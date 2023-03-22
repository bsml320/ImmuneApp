import os, re, sys
import numpy as np
import pandas as pd

def merge_dicts(*dict_args):
    result = {}
    for item in dict_args:
        result.update(item)

    return result

def merge_background_scores(filepath, identifier):
    all_dict_background = []
    for n in range(5):
        dict_background = np.load(filepath + 'dict_%s_background_%s.npy' % (identifier, str(n)), allow_pickle=True).item()
        all_dict_background.append(dict_background)   
         
    all_dict_background1 =  merge_dicts(*all_dict_background)    
    
    return all_dict_background1
