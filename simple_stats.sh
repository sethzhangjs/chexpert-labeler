#!/bin/bash
#SBATCH --job-name=count_tokens
#SBATCH --output=/home/szhan81/chexpert-labeler/logs/count_tokens_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/logs/count_tokens_%j_%x.err
#SBATCH --mem=16G
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

python simple_stats.py \
    --reports_path /home/szhan81/chexpert-labeler/discharge_summary.csv \
    --output_path /home/szhan81/chexpert-labeler/discharge_summary_token_counts.csv \
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