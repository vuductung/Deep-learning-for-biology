import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from dl_biology.cafa6.constants import amino_acids
from dl_biology.cafa6.helper import extract_data_from_fasta


def aa_encoder(seq, vocab=None):
    """
    Encode the aa in a numerical representation
    e.g. "ACD" -> 012
    """
    if vocab is None:
        vocab = amino_acids
    amino_acid_dict = {aa: idx for idx, aa in enumerate(vocab)}
    if isinstance(seq, list):
        return [amino_acid_dict[aa] for aa in seq]

    elif isinstance(seq, str):
        return [amino_acid_dict[aa] for aa in list(seq)]


def get_hidden_embedding(model, tokenizer, sequences, device):
    """
    Obtains hidden embeddings from a model for a given list of sequences.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to obtain embeddings from.
    tokenizer : callable
        Tokenizer compatible with the model for processing the sequences.
    sequences : list of str
        Protein or nucleotide sequences for which to compute embeddings.
    device : torch.device or str
        Device on which to run the computations (e.g., 'cpu' or 'cuda').

    Returns
    -------
    numpy.ndarray
        Hidden state embeddings as output by the model, with shape
        (batch_size, sequence_length, embedding_dim).
    """
    model_inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(**model_inputs)
        output = output.last_hidden_state

    return output.detach().cpu().numpy()


def get_input_embedding(model, tokenizer, sequences):
    """
    Returns the input embeddings for a list of sequences using the provided model and tokenizer.

    Parameters
    ----------
    model : EsmModel
        The pretrained ESM model from which to extract input embeddings.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model for encoding the sequences.
    sequences : list of str
        List of protein sequences as strings.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is the input embedding matrix (shape: sequence_length x embedding_dim)
        for the corresponding sequence.
    """
    input_embeddings = []
    input_embedding = model.get_input_embeddings().weight.detach().numpy()
    for seq in sequences:
        indices = tokenizer(seq)["input_ids"]
        embedding = input_embedding[indices]
        input_embeddings.append(embedding)

    return input_embeddings
