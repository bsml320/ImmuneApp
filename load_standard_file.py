from typing import Union, List, Tuple
from os import PathLike

def load_standard_file(filepath: Union[str, PathLike]):
    with open(filepath, 'r') as f:
        lines = [l.strip().split() for l in f.readlines()]
    samples = []
    for line in lines:
        if len(line) == 1:
            raise ValueError('Each file in the template must have at least one alleles assigned to it.')
        if line == '':
            continue
        samples.append({'file': line[0], 'alleles': line[1:]})
    return samples