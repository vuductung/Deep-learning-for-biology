"""
Generate protein embeddings using ESM2 on multiple GPUs.
Usage:
    # Generate: python generate_embeddings.py --fasta sequences.fasta --output embeddings/ --rank 0 --world_size 4
    # Merge:    python generate_embeddings.py --merge --output embeddings/ --world_size 4
"""

import pickle
import sys
from itertools import islice
from pathlib import Path

import lmdb
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


class EmbeddingGenerator:
    def __init__(self, device, model_checkpoint="facebook/esm2_t33_650M_UR50D"):
        print(f"Loading model on {device}...")
        self.model = EsmModel.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.model_name = model_checkpoint.split("/")[-1]
        print("Model loaded!")

    def count_sequences(self, fasta_file):
        """Count sequences without loading into memory."""
        return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))

    def generate_embeddings(
        self,
        fasta_file,
        output_dir,
        rank,
        world_size,
        batch_size=32,
        max_length=1024,
    ):
        """Generate embeddings for this rank's portion of sequences."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count and split sequences
        n_total = self.count_sequences(fasta_file)
        seqs_per_gpu = n_total // world_size
        start_idx = rank * seqs_per_gpu
        end_idx = n_total if rank == world_size - 1 else (rank + 1) * seqs_per_gpu
        n_seqs = end_idx - start_idx

        print(f"Rank {rank}: Processing sequences {start_idx}-{end_idx} ({n_seqs} total)")

        # Open LMDB
        lmdb_path = output_dir / f"{self.model_name}_embeddings_rank{rank}.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=200 * 1024**3, writemap=True)

        # Stream through FASTA (memory efficient!)
        seq_iter = SeqIO.parse(fasta_file, "fasta")

        # Skip to start position
        for _ in range(start_idx):
            next(seq_iter)

        # Process this rank's sequences
        my_seqs = islice(seq_iter, n_seqs)

        write_count = 0
        txn = env.begin(write=True)

        for batch in tqdm(
            self._batch_iter(my_seqs, batch_size),
            total=(n_seqs + batch_size - 1) // batch_size,
            desc=f"GPU {rank}",
        ):
            seq_ids = [rec.id.split("|")[1] if "|" in rec.id else rec.id for rec in batch]
            seq_strs = [str(rec.seq) for rec in batch]

            with torch.no_grad():
                inputs = self.tokenizer(
                    seq_strs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                embeddings = self.model(**inputs).last_hidden_state.cpu().numpy()
                masks = inputs["attention_mask"].cpu().numpy()

            for i, seq_id in enumerate(seq_ids):
                valid_len = int(masks[i].sum())
                emb = embeddings[i][:valid_len]
                txn.put(seq_id.encode(), pickle.dumps(emb, protocol=5))
                write_count += 1

                # Commit every 1000
                if write_count % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn.commit()
        env.close()

        print(f"Rank {rank}: Saved {write_count} embeddings to {lmdb_path}")

    def _batch_iter(self, sequences, batch_size):
        """Yield batches from sequence iterator."""
        sequences = iter(sequences)
        while True:
            batch = list(islice(sequences, batch_size))
            if not batch:
                break
            yield batch


def merge_lmdb(output_dir, world_size, model_name):
    """Merge rank files into single LMDB."""

    output_dir = Path(output_dir)
    final_path = output_dir / "embeddings.lmdb"

    print(f"Merging {world_size} rank files into {final_path}...")

    env_final = lmdb.open(str(final_path), map_size=200 * 1024**3, writemap=True)

    total = 0
    txn = env_final.begin(write=True)

    for rank in range(world_size):
        rank_path = output_dir / f"{model_name}_embeddings_rank{rank}.lmdb"

        if not rank_path.exists():
            print(f"WARNING: {rank_path} not found, skipping")
            continue

        print(f"Merging rank {rank}...")
        env_rank = lmdb.open(str(rank_path), readonly=True)

        count = 0
        with env_rank.begin() as txn_rank:
            cursor = txn_rank.cursor()
            for key, value in cursor:
                txn.put(key, value)
                count += 1

                # Commit every 10k
                if count % 10000 == 0:
                    txn.commit()
                    txn = env_final.begin(write=True)

        env_rank.close()
        total += count
        print(f"  Merged {count} sequences from rank {rank}")

    txn.commit()
    env_final.close()

    print(f"SUCCESS: Merged {total} total sequences into {final_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate ESM2 embeddings")
    parser.add_argument("--fasta", type=str, help="Input FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--world_size", type=int, default=1, help="Total GPUs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--merge", action="store_true", help="Merge mode")

    args = parser.parse_args()

    try:
        if args.merge:
            model_name = args.model.split("/")[-1]
            merge_lmdb(args.output, args.world_size, model_name)
        else:
            if not args.fasta:
                raise ValueError("--fasta required for generation mode")

            device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
            generator = EmbeddingGenerator(device, args.model)
            generator.generate_embeddings(
                args.fasta,
                args.output,
                args.rank,
                args.world_size,
                args.batch_size,
                args.max_length,
            )

        print("SUCCESS!")
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
