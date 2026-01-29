#!/bin/bash
#SBATCH --job-name=label_csv_nltk
#SBATCH --output=/home/szhan81/chexpert-labeler/logs/label_csv_nltk_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/logs/label_csv_nltk_%j_%x.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

python -u label_csv_nltk.py\
    --reports_path /home/szhan81/chexpert-labeler/demo/data/input/Atelectasis_mention_sample_analysis.csv\
    --output_path /home/szhan81/chexpert-labeler/demo/data/output/Atelectasis_mention_sample_analysis_results.csv\
    --batch_size 10 \
    --enable_sentences \
    --min_sentence_length 5 \
    --print_sentences \
    --print_timing \
    --verbose