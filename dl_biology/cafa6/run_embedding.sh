#!/bin/bash
#SBATCH --job-name=cafa6_embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4                    # 4 GPUs
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G                    # 32GB per GPU
#SBATCH --time=24:00:00               # 24 hours
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

# Setup
module purge
module load python-waterboa/2024.06
module load cuda/12.1
source ~/projects/cafa6/venv/bin/activate  # ← CHANGE THIS

mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Configuration - CHANGE THESE FOR YOUR PROJECT
FASTA_FILE="/ptmp/dtvu/data/cafa6/train_sequences.fasta"  # ← CHANGE
OUTPUT_DIR="/ptmp/dtvu/data/cafa6/embeddings"             # ← CHANGE
BATCH_SIZE=32                                          # Adjust if OOM
MODEL="facebook/esm2_t33_650M_UR50D"                  # Good default

# Job info
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_NTASKS"
echo "FASTA: $FASTA_FILE"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"
echo "======================================"

# Generate embeddings
srun --ntasks=$SLURM_NTASKS \
     --output=logs/embed_%j_%t.out \
     --error=logs/embed_%j_%t.err \
     python generate_embeddings.py \
        --fasta $FASTA_FILE \
        --output $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --model $MODEL \
        --rank $SLURM_PROCID \
        --world_size $SLURM_NTASKS

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed!"
    exit 1
fi

echo "Generation complete! Merging..."

# Merge
python generate_embeddings.py \
    --merge \
    --output $OUTPUT_DIR \
    --world_size $SLURM_NTASKS

if [ $? -ne 0 ]; then
    echo "ERROR: Merge failed!"
    exit 1
fi

echo "======================================"
echo "SUCCESS!"
echo "Output: ${OUTPUT_DIR}/embeddings.lmdb"
echo "End: $(date)"
echo "======================================"
