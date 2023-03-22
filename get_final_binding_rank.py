import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def get_final_binding_rank(allele, list_predicted, dict_distr):
    list_out = list(map(lambda x: np.around(percentileofscore(dict_distr[allele[4:]],x),4), list_predicted))

    return list_out 
