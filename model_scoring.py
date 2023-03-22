import os, re, sys
import numpy as np
import pandas as pd

def model_scoring(models, data):
    probas_ = [np.transpose(model.predict(data))[0] for model in models]
    probas_ = [np.mean(scores) for scores in zip(*probas_)]
    return probas_  