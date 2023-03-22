import os, re, sys
import numpy as np
import pandas as pd

def sample_fasta_peptides(sequences, peptide_lengths):
    sample_peptides = {}
    for (i, (name, sequence)) in enumerate(sequences.items()):
        if not isinstance(sequence, str):
            raise ValueError("Expected string, not %s (%s)" % (
                sequence, type(sequence)))
        for peptide_start in range(len(sequence) - min(peptide_lengths) + 1):
            for peptide_length in peptide_lengths:
                peptide = sequence[
                    peptide_start: peptide_start + peptide_length
                ]
                if len(peptide) != peptide_length:
                    continue
                if name not in sample_peptides.keys() :
                    sample_peptides[name] = [peptide]
                else:
                    sample_peptides[name].append(peptide)
    return sample_peptides     