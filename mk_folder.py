import sys, os, math, tempfile, datetime, time, copy, re, argparse
import numpy as np
import pandas as pd
from os import PathLike
import shutil
from pathlib import Path

def mk_folder(tmp_folder, sample_info, sample_alleles):
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    if Path(tmp_folder + 'gibbscluster').exists() and Path(tmp_folder + 'gibbscluster').is_dir():
        shutil.rmtree(f'{Path(tmp_folder + "gibbscluster")}')

    Path(tmp_folder + 'gibbscluster').mkdir()
 
    for sample in sample_info:
        sample_name = sample['sample-name']
        alleles = [x.strip() for x in sample['sample-alleles'].split(',')]
        sample_alleles[sample_name] = alleles
        Path(tmp_folder + 'gibbscluster/' + sample_name).mkdir()
        Path(tmp_folder + 'gibbscluster/' + sample_name + '/unsupervised').mkdir()
        Path(tmp_folder + 'gibbscluster/' + sample_name + '/unannotated').mkdir()
        for allele in alleles:
            Path(tmp_folder + 'gibbscluster/' + sample_name + '/' + allele).mkdir()