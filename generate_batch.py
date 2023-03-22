import os, re, sys
import numpy as np
import pandas as pd
import random

def generate_batch(x_train,y_train,batch_size,x_train2,randomFlag=True):
    ylen = len(y_train)
    loopcount = ylen // batch_size
    i=-1
    while True:
        if randomFlag:
            i = random.randint(0,loopcount-1)
        else:
            i=i+1
            i=i%loopcount
        print(i)
        yield ({'inputs_1': x_train[i*batch_size:(i+1)*batch_size], 
        'inputs_2': x_train2[i*batch_size:(i+1)*batch_size]}, 
        {'outputs': y_train[i*batch_size:(i+1)*batch_size]}) 