#!/bin/bash
#SBATCH --job-name=label_csv
#SBATCH --output=/home/szhan81/chexpert-labeler/logs/label_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/logs/label_%j_%x.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

# Record start time
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="
start_time=$(date +%s)

python label.py \
    --reports_path /home/szhan81/chexpert-labeler/discharge_summary.csv \
    --output_path /home/szhan81/chexpert-labeler/discharge_summary_label.csv \
    --verbose

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Total seconds: ${duration}"
echo "=========================================="