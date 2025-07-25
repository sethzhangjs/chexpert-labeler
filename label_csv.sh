#!/bin/bash
#SBATCH --job-name=label_csv
#SBATCH --output=/home/szhan81/chexpert-labeler/demo/logs/label_csv_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/demo/logs/label_csv_%j_%x.err
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

python -u label_csv.py\
    --reports_path "/home/szhan81/chexpert-labeler/demo/data/Atelectasis_mention_sample_analysis.csv"\
    --output_path "/home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_results.csv"\
    --batch_size 10 \
    --max_text_length 350 \
    --verbose