#!/usr/bin/env bash
#SBATCH --job-name=align_01
#SBATCH --partition=long-cpu	
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
# Run
##################################################################
python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Assamese" \
    --timing_folder "data/audios/Assamese/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Assamese/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Bengali" \
    --timing_folder "data/audios/Bengali/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Bengali/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Central Kurdish" \
    --timing_folder "data/audios/Central Kurdish/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Central Kurdish/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Chhattisgarhi" \
    --timing_folder "data/audios/Chhattisgarhi/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Chhattisgarhi/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Dholuo" \
    --timing_folder "data/audios/Dholuo/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Dholuo/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Ewe" \
    --timing_folder "data/audios/Ewe/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Ewe/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Gujarati" \
    --timing_folder "data/audios/Gujarati/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Gujarati/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Haryanvi" \
    --timing_folder "data/audios/Haryanvi/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Haryanvi/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Hausa" \
    --timing_folder "data/audios/Hausa/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Hausa/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Hiligaynon" \
    --timing_folder "data/audios/Hiligaynon/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Hiligaynon/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Hindi" \
    --timing_folder "data/audios/Hindi/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Hindi/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Igbo" \
    --timing_folder "data/audios/Igbo/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Igbo/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Kannada" \
    --timing_folder "data/audios/Kannada/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Kannada/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Lingala" \
    --timing_folder "data/audios/Lingala/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Lingala/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Malayalam" \
    --timing_folder "data/audios/Malayalam/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Malayalam/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Marathi" \
    --timing_folder "data/audios/Marathi/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Marathi/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Ndebele" \
    --timing_folder "data/audios/Ndebele/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Ndebele/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Oromo" \
    --timing_folder "data/audios/Oromo/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Oromo/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Punjabi" \
    --timing_folder "data/audios/Punjabi/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Punjabi/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Tamil" \
    --timing_folder "data/audios/Tamil/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Tamil/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Telugu" \
    --timing_folder "data/audios/Telugu/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Telugu/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Toma" \
    --timing_folder "data/audios/Toma/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Toma/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Twi (Akuapem)" \
    --timing_folder "data/audios/Twi (Akuapem)/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Twi (Akuapem)/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Twi (Asante)" \
    --timing_folder "data/audios/Twi (Asante)/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Twi (Asante)/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Ukrainian" \
    --timing_folder "data/audios/Ukrainian/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Ukrainian/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Urdu" \
    --timing_folder "data/audios/Urdu/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Urdu/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Vietnamese" \
    --timing_folder "data/audios/Vietnamese/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Vietnamese/Paratext (USFM)/release/USX_1" \
    --workers 16


python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Yoruba" \
    --timing_folder "data/audios/Yoruba/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Yoruba/Paratext (USFM)/release/USX_1" \
    --workers 16



echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"