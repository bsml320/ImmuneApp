import os, re, sys
import numpy as np
import pandas as pd

def transform_affinity(x):
    x = np.clip(x, a_min=None, a_max=50000)
    
    return 1 - np.log(x) / np.log(50000)