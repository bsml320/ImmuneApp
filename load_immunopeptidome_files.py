from typing import Union, List, Tuple
from os import PathLike

def load_immunopeptidome_files(filepath: Union[str, PathLike], peptide_column: str = None, delimiter: str = None):
    if not peptide_column == delimiter and None in [peptide_column, delimiter]:
        raise ValueError('Both peptide and delimiter must be defined for multi-column files.')

    peptides = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if delimiter:
            if delimiter == 'comma':
                delimiter = ','
            else:
                delimiter = '\t'
            header_index = lines[0].strip().index(peptide_column)
            for line in lines[1:]:
                peptides.append(line.strip().split(delimiter)[header_index])
        else:
            for line in lines:
                peptides.append(line.strip())

    return peptides  