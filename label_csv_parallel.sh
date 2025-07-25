#!/bin/bash
#SBATCH --job-name=parallel_chexpert
#SBATCH --output=/home/szhan81/chexpert-labeler/demo/logs/parallel_chexpert_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/demo/logs/parallel_chexpert_%j_%x.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

echo "Starting parallel CheXpert processing..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Calculate optimal batch size and worker count
# Recommendation: batch_size = num_workers * 50-100 for balanced memory usage
NUM_WORKERS=$SLURM_CPUS_PER_TASK
BATCH_SIZE=$((NUM_WORKERS * 100))  # Adjust multiplier based on your data size

echo "Using $NUM_WORKERS worker processes"
echo "Using batch size: $BATCH_SIZE"

# Run the parallel processing
python -u label_csv_parallel.py \
    --reports_path "/home/szhan81/chexpert-labeler/demo/data/Atelectasis_mention_sample_analysis.csv" \
    --output_path "/home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_parallel_results.csv" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --enable_segmentation \
    --segment_length 350 \
    --segment_overlap 50 \
    --verbose

echo "End time: $(date)"
echo "Job completed!"

# Optional: Calculate processing statistics
if [ -f "/home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_parallel_results.csv" ]; then
    echo "Output file created successfully"
    echo "Output file size: $(du -h /home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_parallel_results.csv)"
else
    echo "WARNING: Output file not found!"
fi