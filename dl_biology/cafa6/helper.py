import pandas as pd
from Bio import SeqIO

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
