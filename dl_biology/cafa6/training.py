from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def generate_dataloader(
    label_data,
    entry_labels,
    embedding_dir,
    train_seq_ids,
    test_seq_ids,
    collate_fn,
    batch_size,
    shuffle=True,
):
    train_dataset = Esm2EmbeddingDataset(label_data, entry_labels, train_seq_ids, embedding_dir)
    test_dataset = Esm2EmbeddingDataset(label_data, entry_labels, test_seq_ids, embedding_dir)

    return DataLoader[Any](train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle), DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )


class Esm2EmbeddingDataset(Dataset):
    def __init__(self, label_data, entry_labels, seq_ids, embedding_dir):
        super().__init__()
        self.embedding_dir = embedding_dir
        self.label_data = label_data
        self.seq_id_to_label_idx = {entry_id: idx for idx, entry_id in enumerate(entry_labels)}
        self.seq_ids = seq_ids
        self._h5_file = None

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]

        if self._h5_file is None:
            self._h5_file = h5py.File(self.embedding_dir, "r", swmr=True)
        embedding = torch.from_numpy(self._h5_file[seq_id]["embedding"][()]).float()  # (seq_len, embedding_size)
        label_idx = self.seq_id_to_label_idx[seq_id]
        label = torch.as_tensor(self.label_data[label_idx], dtype=torch.float32)

        return {"seq_len": embedding.shape[0], "embedding": embedding, "label": label}


def calculate_fmax(predictions, y_targets, weights=None, thresholds=np.arange(0.01, 1.01, 0.01)):
    """
    Calculate F-max (or weighted F-max if weights provided)

    Parameters:
    -----------
    predictions : array-like, shape (n_proteins, n_terms)
        Prediction scores for each protein-term pair
    y_targets : array-like, shape (n_proteins, n_terms)
        Binary ground truth (1 if protein has term, 0 otherwise)
    weights : array-like, shape (n_terms,), optional
        Information accretion weight for each GO term
        If None, calculates standard (unweighted) F-max
    thresholds : array-like
        Threshold values to evaluate (default: 0.01 to 1.00, step 0.01)

    Returns:
    --------
    fmax : float
        Maximum F1 score across all thresholds
    best_threshold : float
        Threshold that achieves F-max
    pr_rc_curve : array, shape (n_thresholds, 4)
        Array of [threshold, precision, recall, f1] for each threshold
    """

    predictions = np.asarray(predictions)
    y_targets = np.asarray(y_targets)

    _, n_terms = predictions.shape

    # If weights not provided, use uniform weights (standard F-max)
    if weights is None:
        weights = np.ones(n_terms)
    else:
        weights = np.asarray(weights)

    pr_rc_curve = []
    best_f = 0
    best_threshold = 0

    for tau in thresholds:
        # Binarize predictions at threshold tau
        pred_binary = (predictions >= tau).astype(float)

        # Number of proteins with at least one prediction at this threshold
        m_tau = np.sum(np.any(pred_binary, axis=1))

        if m_tau == 0:
            continue

        # True positives: predictions that match ground truth
        tp = pred_binary * y_targets  # element-wise multiplication

        # Calculate weighted precision
        # Numerator: sum of weights for true positives
        wpr_numerator = np.sum(tp * weights)
        # Denominator: sum of weights for all predictions
        wpr_denominator = np.sum(pred_binary * weights)

        # Calculate weighted recall
        # Numerator: sum of weights for true positives (same as precision)
        wrc_numerator = np.sum(tp * weights)
        # Denominator: sum of weights for all ground truth terms
        wrc_denominator = np.sum(y_targets * weights)

        # Compute precision and recall
        pr = wpr_numerator / wpr_denominator if wpr_denominator > 0 else 0
        rc = wrc_numerator / wrc_denominator if wrc_denominator > 0 else 0

        # Calculate F1
        if pr + rc > 0:
            f1 = 2 * pr * rc / (pr + rc)
        else:
            f1 = 0

        pr_rc_curve.append([tau, pr, rc, f1])

        # Track best F1
        if f1 > best_f:
            best_f = f1
            best_threshold = tau

    pr_rc_curve = np.array(pr_rc_curve)

    return best_f, best_threshold, pr_rc_curve
