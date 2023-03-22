import re
import random
import numpy as np
import pandas as pd
from collections import Counter 

def random_pick(seq,probabilities):
    # x = random.uniform(0, 1)
    x = random.randint(1, sum(probabilities))
    cumulative_probability = 0
    for item, item_probability in zip(seq,probabilities):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
        
    return item  