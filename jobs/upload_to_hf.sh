#!/usr/bin/env bash
#SBATCH --job-name=upload_to_hf
#SBATCH --partition=long-cpu	
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
module load gcc/9.3.0

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# PYANNOTE_TOKEN must come from the submitting shell (sbatch exports it by
# default): export PYANNOTE_TOKEN="..." && sbatch upload_to_hf.sh
# Without it upload_to_hf.py silently skips diarization and drops speaker_id.
if [[ -z "${PYANNOTE_TOKEN:-}" ]]; then
  echo "ERROR: PYANNOTE_TOKEN is not set. Export it before submitting." >&2
  exit 1
fi

conda activate ReadAlongs

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-resources

##################################################################
# Run
##################################################################
python upload_to_hf.py

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"