#!/bin/bash
#SBATCH --job-name=label_csv
#SBATCH --output=/home/szhan81/chexpert-labeler/logs/label_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/logs/label_%j_%x.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

python label.py \
    --reports_path /home/szhan81/chexpert-labeler/discharge_summary.csv \
    --output_path /home/szhan81/chexpert-labeler/discharge_summary_label.csv \
    --verbose