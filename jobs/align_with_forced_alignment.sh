#!/usr/bin/env bash
#SBATCH --job-name=align_with_forced_alignment
#SBATCH --partition=long-cpu	
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
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

conda activate ReadAlongs

cd /home/mila/g/guzmand/scratch/Repositories/bible-tts-resources

##################################################################
# Run force alignment for all Swahili books
##################################################################
# Debug books that failed
# python utils/force_align_book.py \
#     --audio_folder "data/audios/Apali/New Testament - mp3/Acts" \
#     --book_usx "data/texts/Apali/USX/release/USX_1/ACT.usx" \
#     --output "Apali" \
#     --language "und" \
#     --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Apali" \
    --usfm_folder "data/texts/Apali/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Arabic Standard" \
    --usfm_folder "data/texts/Arabic Standard/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Chichewa" \
    --usfm_folder "data/texts/Chichewa/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Dawro" \
    --usfm_folder "data/texts/Dawro/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Gamo" \
    --usfm_folder "data/texts/Gamo/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Gofa" \
    --usfm_folder "data/texts/Gofa/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Haitian Creole" \
    --usfm_folder "data/texts/Haitian Creole/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Kikuyu" \
    --usfm_folder "data/texts/Kikuyu/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Luganda" \
    --usfm_folder "data/texts/Luganda/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Shona" \
    --usfm_folder "data/texts/Shona/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Swahili" \
    --usfm_folder "data/texts/Swahili/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"


python utils/process_all_books_force_align.py \
    --base_path "data/audios/Turkish" \
    --usfm_folder "data/texts/Turkish/USX/release/USX_1" \
    --language "und" \
    --workers 32 \
    --chapter-intro "chapter introduction"

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
