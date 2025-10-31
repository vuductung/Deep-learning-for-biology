from itertools import islice
from pathlib import Path

import h5py
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from dl_biology.cafa6.constants import amino_acids


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


def get_hidden_embedding(model, tokenizer, sequences, device, truncation=True, max_length=1024):
    """
    Obtains hidden embeddings from a model for a given list of sequences.
    This is a simple implementation using a small set of sequences.

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
    model_inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=truncation, max_length=max_length)
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
    This is a simple implementation using a small set of sequences.

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


class EmbeddingGenerator:
    def __init__(self, device: torch.device, model_checkpoint="facebook/esm2_t33_650M_UR50D"):
        self.model = EsmModel.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate_embedding(self, output_dir, fasta_file, batch_size, max_length, max_n_sequences=1000):
        # create the path to save the embeddings
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # parse the fasta file
        sequences = SeqIO.parse(fasta_file, "fasta")

        if max_n_sequences:
            sequences = islice(sequences, max_n_sequences)

        # generate embedding from batches
        h5_path = output_dir / "train_embeddings.h5"

        with h5py.File(h5_path, "w") as h5f:
            for batch in tqdm(self._batch_iterator(sequences, batch_size)):
                seq_ids = [rec.id.split("|")[1] for rec in batch]
                seq_strs = [str(rec.seq) for rec in batch]

                with torch.no_grad():
                    inputs = self.tokenizer(
                        seq_strs,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(self.device)

                    output = self.model(**inputs)
                    hidden_states = output.last_hidden_state.cpu().numpy()
                    attention_mask = inputs["attention_mask"].cpu().numpy()

                    # Save each sequence to HDF5 (much faster!)
                    for i, seq_id in enumerate(seq_ids):
                        grp = h5f.create_group(seq_id)
                        valid_embedding_idx = attention_mask[i].sum()
                        grp.create_dataset("embedding", data=hidden_states[i][:valid_embedding_idx], compression="gzip")

    def _batch_iterator(self, sequences, batch_size):
        while True:
            batch = list(islice(sequences, batch_size))
            if not batch:
                break
            else:
                yield batch


class Esm2EmbeddingDataset(Dataset):
    def __init__(self, label_dir, embedding_dir):
        super().__init__()
        self.label_data = pd.read_csv(label_dir, delimiter="\t")
        self.embedding_dir = embedding_dir

        # Open the HDF5 file in read mode to obtain sequence IDs
        with h5py.File(self.embedding_dir, "r") as h5f:
            self.seq_ids = list(h5f.keys())

        # Truncate the label data to only those entries with embeddings
        self.label_data_trunc = self.label_data[self.label_data["EntryID"].isin(self.seq_ids)]
        # Compute multi-hot labels as a tensor
        self.y = torch.from_numpy(
            pd.crosstab(self.label_data_trunc["EntryID"], self.label_data_trunc["term"])
            .reindex(self.seq_ids, fill_value=0)
            .values
        ).float()
        # Store terms to recover class order, if needed
        self.terms = pd.unique(self.label_data_trunc["term"])

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]

        with h5py.File(self.embedding_dir, "r") as h5f:
            embedding = torch.from_numpy(h5f[seq_id]["embedding"][()], dtype=torch.float32)

        label = self.y[idx]

        return {"seq_len": embedding.shape[0], "embedding": embedding, "label": label}
