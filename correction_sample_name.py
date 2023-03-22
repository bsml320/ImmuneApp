import os, re, sys
import numpy as np
import pandas as pd

def correction_sample_name(sample_name: str):
    for bad_character in [' ', ':', ';', '/', '\\', '$', '@', '*', '!', '^', '(', ')', '{', '}', '[', ']']:
        sample_name = sample_name.replace(bad_character, '_')
    sample_name = sample_name.replace('&', 'AND').replace('%', 'percent')
    return sample_name