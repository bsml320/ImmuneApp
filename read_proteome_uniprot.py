import re
import random
import numpy as np
import pandas as pd
from collections import Counter 

def read_proteome_uniprot(): 
    path = "proteome_uniprot_filepath" #change as your uniprot proteome filepath
    reference_df = pd.read_csv(path, index_col=0)
    reference_df1 = reference_df.set_index(['accession'])['seq'].to_dict()
    
    return reference_df1