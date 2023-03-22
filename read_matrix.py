import os, re, sys
import numpy as np
import pandas as pd
import random

def read_matrix(path, identifier):
    f = open(path,"r")
    blosum = []
    if identifier == 0: #(blosum 50)
       for line in f:
           blosum.append([(float(i))/10 for i in re.split("\t",line)])
    else:
        for line in f: #(one-hot)
           blosum.append([float(i) for i in re.split("\t",line)])
    f.close()
    
    return blosum