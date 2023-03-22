import os, re, sys
import numpy as np
import pandas as pd

def affinity_transform(x):
    return 50000**(1-x)