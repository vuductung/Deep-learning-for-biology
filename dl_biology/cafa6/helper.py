from itertools import islice

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.sparse import csr_matrix

from dl_biology.cafa6.constants import amino_acids


def extract_data_from_fasta(path):
    """
    Extracts sequence data from a FASTA file and returns it in a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the FASTA file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per sequence and columns:
        - 'entryID': The sequence identifier.
        - 'sequence': The nucleotide or amino acid sequence as a string.
        - 'length': The length of the sequence.
        - 'species': Species name
    """
    entries = SeqIO.parse(open(path), "fasta")

    data = []
    for entry in entries:
        data.append(
            {
                "EntryID": entry.id.split("|")[1],
                "sequence": str(entry.seq),
                "length": len(entry.seq),
                "species": entry.id.split("|")[-1].split("_")[-1],
            }
        )

    return pd.DataFrame(data)


def get_csr_matrix_from_terms(data):
    entry_code = data["EntryID"].astype("category").cat.codes
    term_code = data["EntryID"].astype("category").cat.codes

    entry_labels = data["EntryID"].astype("category").cat.categories
    term_labels = data["term"].astype("category").cat.categories
    sparse_matrix = csr_matrix((np.ones(len(data)), (entry_code, term_code)))
    return sparse_matrix.toarray(), entry_labels, term_labels
