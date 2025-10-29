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


def get_hidden_embedding_batch(model, tokenizer, sequences, device, batch_size=32):
    """
    Efficiently computes hidden embeddings for large numbers of sequences in batches.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to obtain embeddings from.
    tokenizer : callable
        Tokenizer compatible with the model for processing the sequences.
    sequences : list of str
        List of protein sequences as strings.
    device : torch.device or str
        Device on which to run the computations (e.g., 'cpu' or 'cuda').
    batch_size : int, default=32
        Number of sequences to process in each batch. Lower than input embeddings
        because this requires model forward pass.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is the hidden embedding matrix (shape: sequence_length x embedding_dim)
        for the corresponding sequence. Padding tokens are removed.
    """
    model.to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch = sequences[i : i + batch_size]

        # Tokenize batch with padding
        batch_inputs = tokenizer(batch, return_tensors="pt", padding=True)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        # Get attention mask to remove padding tokens
        attention_mask = batch_inputs["attention_mask"]

        # Forward pass
        with torch.no_grad():
            output = model(**batch_inputs)
            hidden_states = output.last_hidden_state.detach().cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()

        # Remove padding tokens for each sequence
        for j in range(len(batch)):
            seq_length = attention_mask_np[j].sum()
            embedding = hidden_states[j, :seq_length]
            all_embeddings.append(embedding)

    return all_embeddings


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


def get_input_embedding_batch(model, tokenizer, sequences, batch_size=1000):
    """
    Efficiently computes input embeddings for large numbers of sequences in batches.

    Parameters
    ----------
    model : EsmModel
        The pretrained ESM model from which to extract input embeddings.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model for encoding the sequences.
    sequences : list of str
        List of protein sequences as strings.
    batch_size : int, default=1000
        Number of sequences to process in each batch.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is the input embedding matrix for the corresponding sequence.
    """
    input_embedding_weight = model.get_input_embeddings().weight.detach().numpy()
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch = sequences[i : i + batch_size]

        # Tokenize batch
        batch_inputs = tokenizer(batch, return_tensors=None, padding=False)

        # Extract embeddings for each sequence in batch
        for input_ids in batch_inputs["input_ids"]:
            embedding = input_embedding_weight[input_ids]
            all_embeddings.append(embedding)

    return all_embeddings


def save_embeddings(embeddings_dict, output_path):
    """Save embeddings to disk using pickle."""
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved embeddings to {output_path}")


def load_embeddings(input_path):
    """Load embeddings from disk."""
    with open(input_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    print(f"Loaded embeddings from {input_path}")
    return embeddings_dict


def process_fasta_to_embeddings(fasta_path, model, tokenizer, output_path=None, batch_size=1000, force_recompute=False):
    """
    End-to-end function to process a FASTA file and generate/save input embeddings.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    model : EsmModel
        The pretrained ESM model.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model.
    output_path : str, optional
        Path to save embeddings. If None, auto-generates from fasta_path.
    batch_size : int, default=1000
        Number of sequences to process in each batch.
    force_recompute : bool, default=False
        If True, recompute embeddings even if cached file exists.

    Returns
    -------
    dict
        Dictionary mapping sequence IDs to their embeddings.
    """
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = str(Path(fasta_path).with_suffix(".embeddings.pkl"))

    # Check if embeddings already exist
    if Path(output_path).exists() and not force_recompute:
        print(f"Loading cached embeddings from {output_path}")
        return load_embeddings(output_path)

    # Extract sequence data from FASTA
    print("Extracting sequences from FASTA...")
    df = extract_data_from_fasta(fasta_path)

    # Compute embeddings
    print(f"Computing embeddings for {len(df)} sequences...")
    embeddings_list = get_input_embedding_batch(model, tokenizer, df["sequence"].tolist(), batch_size=batch_size)

    # Create dictionary mapping IDs to embeddings
    embeddings_dict = dict(zip(df["EntryID"], embeddings_list))

    # Save embeddings
    save_embeddings(embeddings_dict, output_path)

    return embeddings_dict


def get_input_embedding_parallel(model, tokenizer, sequences, n_workers=4, batch_size=1000):
    """
    Parallel version of get_input_embedding using threading.

    Note: This function uses ThreadPoolExecutor for parallelization. For input embeddings,
    the batched version (get_input_embedding_batch) is typically faster and more memory
    efficient. This parallel version may cause kernel crashes in Jupyter notebooks due
    to threading issues. Consider using get_input_embedding_batch instead.

    Parameters
    ----------
    model : EsmModel
        The pretrained ESM model from which to extract input embeddings.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model for encoding the sequences.
    sequences : list of str
        List of protein sequences as strings.
    n_workers : int, default=4
        Number of parallel workers to use.
    batch_size : int, default=1000
        Number of sequences to process in each chunk.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is the input embedding matrix for the corresponding sequence.
    """
    input_embedding_weight = model.get_input_embeddings().weight.detach().numpy()

    def _process_single_sequence(seq):
        """Helper function for parallel processing of a single sequence."""
        # Tokenize
        inputs = tokenizer(seq, return_tensors=None, padding=False)
        input_ids = inputs["input_ids"]

        # Extract embedding
        embedding = input_embedding_weight[input_ids]

        return embedding

    all_embeddings = []

    # Process in batches to show progress
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch = sequences[i : i + batch_size]

        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            batch_embeddings = list(executor.map(_process_single_sequence, batch))

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def process_fasta_to_embeddings_parallel(
    fasta_path, model, tokenizer, output_path=None, batch_size=1000, n_workers=8, force_recompute=False
):
    """
    Parallel version of process_fasta_to_embeddings for faster processing.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    model : EsmModel
        The pretrained ESM model.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model.
    output_path : str, optional
        Path to save embeddings. If None, auto-generates from fasta_path.
    batch_size : int, default=1000
        Number of sequences to process in each batch.
    n_workers : int, default=8
        Number of parallel workers to use.
    force_recompute : bool, default=False
        If True, recompute embeddings even if cached file exists.

    Returns
    -------
    dict
        Dictionary mapping sequence IDs to their embeddings.
    """
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = str(Path(fasta_path).with_suffix(".embeddings.pkl"))

    # Check if embeddings already exist
    if Path(output_path).exists() and not force_recompute:
        print(f"Loading cached embeddings from {output_path}")
        return load_embeddings(output_path)

    # Extract sequence data from FASTA
    print("Extracting sequences from FASTA...")
    df = extract_data_from_fasta(fasta_path)

    # Compute embeddings in parallel
    print(f"Computing embeddings for {len(df)} sequences using {n_workers} workers...")
    embeddings_list = get_input_embedding_parallel(
        model, tokenizer, df["sequence"].tolist(), n_workers=n_workers, batch_size=batch_size
    )

    # Create dictionary mapping IDs to embeddings
    embeddings_dict = dict(zip(df["EntryID"], embeddings_list))

    # Save embeddings
    save_embeddings(embeddings_dict, output_path)

    return embeddings_dict


def process_fasta_to_hidden_embeddings(
    fasta_path, model, tokenizer, device, output_path=None, batch_size=32, force_recompute=False
):
    """
    End-to-end function to process a FASTA file and generate/save hidden embeddings.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    model : torch.nn.Module
        The pretrained ESM model.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model.
    device : torch.device or str
        Device on which to run the computations (e.g., 'cpu' or 'cuda').
    output_path : str, optional
        Path to save embeddings. If None, auto-generates from fasta_path.
    batch_size : int, default=32
        Number of sequences to process in each batch.
    force_recompute : bool, default=False
        If True, recompute embeddings even if cached file exists.

    Returns
    -------
    dict
        Dictionary mapping sequence IDs to their hidden embeddings.
    """
    # Auto-generate output path if not provided
    if output_path is None:
        output_path = str(Path(fasta_path).with_suffix(".hidden_embeddings.pkl"))

    # Check if embeddings already exist
    if Path(output_path).exists() and not force_recompute:
        print(f"Loading cached hidden embeddings from {output_path}")
        return load_embeddings(output_path)

    # Extract sequence data from FASTA
    print("Extracting sequences from FASTA...")
    df = extract_data_from_fasta(fasta_path)

    # Compute embeddings
    print(f"Computing hidden embeddings for {len(df)} sequences...")
    embeddings_list = get_hidden_embedding_batch(
        model, tokenizer, df["sequence"].tolist(), device, batch_size=batch_size
    )

    # Create dictionary mapping IDs to embeddings
    embeddings_dict = dict(zip(df["EntryID"], embeddings_list))

    # Save embeddings
    save_embeddings(embeddings_dict, output_path)

    return embeddings_dict


def process_fasta_to_both_embeddings(
    fasta_path,
    model,
    tokenizer,
    device,
    input_output_path=None,
    hidden_output_path=None,
    input_batch_size=1000,
    hidden_batch_size=32,
    force_recompute=False,
):
    """
    End-to-end function to process a FASTA file and generate/save both input and hidden embeddings.

    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    model : torch.nn.Module
        The pretrained ESM model.
    tokenizer : AutoTokenizer
        Tokenizer compatible with the model.
    device : torch.device or str
        Device on which to run the computations (e.g., 'cpu' or 'cuda').
    input_output_path : str, optional
        Path to save input embeddings. If None, auto-generates from fasta_path.
    hidden_output_path : str, optional
        Path to save hidden embeddings. If None, auto-generates from fasta_path.
    input_batch_size : int, default=1000
        Number of sequences to process in each batch for input embeddings.
    hidden_batch_size : int, default=32
        Number of sequences to process in each batch for hidden embeddings.
    force_recompute : bool, default=False
        If True, recompute embeddings even if cached files exist.

    Returns
    -------
    tuple of dict
        Tuple of (input_embeddings_dict, hidden_embeddings_dict), each mapping sequence IDs to embeddings.
    """
    # Auto-generate output paths if not provided
    if input_output_path is None:
        input_output_path = str(Path(fasta_path).with_suffix(".input_embeddings.pkl"))
    if hidden_output_path is None:
        hidden_output_path = str(Path(fasta_path).with_suffix(".hidden_embeddings.pkl"))

    # Extract sequence data from FASTA (only once)
    print("Extracting sequences from FASTA...")
    df = extract_data_from_fasta(fasta_path)
    sequences = df["sequence"].tolist()
    entry_ids = df["EntryID"].tolist()

    # Compute input embeddings
    input_embeddings_dict = None
    if not Path(input_output_path).exists() or force_recompute:
        print(f"Computing input embeddings for {len(df)} sequences...")
        input_embeddings_list = get_input_embedding_batch(model, tokenizer, sequences, batch_size=input_batch_size)
        input_embeddings_dict = dict(zip(entry_ids, input_embeddings_list))
        save_embeddings(input_embeddings_dict, input_output_path)
    else:
        print(f"Loading cached input embeddings from {input_output_path}")
        input_embeddings_dict = load_embeddings(input_output_path)

    # Compute hidden embeddings
    hidden_embeddings_dict = None
    if not Path(hidden_output_path).exists() or force_recompute:
        print(f"Computing hidden embeddings for {len(df)} sequences...")
        hidden_embeddings_list = get_hidden_embedding_batch(
            model, tokenizer, sequences, device, batch_size=hidden_batch_size
        )
        hidden_embeddings_dict = dict(zip(entry_ids, hidden_embeddings_list))
        save_embeddings(hidden_embeddings_dict, hidden_output_path)
    else:
        print(f"Loading cached hidden embeddings from {hidden_output_path}")
        hidden_embeddings_dict = load_embeddings(hidden_output_path)

    return input_embeddings_dict, hidden_embeddings_dict
