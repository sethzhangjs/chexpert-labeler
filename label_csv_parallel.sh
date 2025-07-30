#!/bin/bash
#SBATCH --job-name=batch_parallel_chexpert
#SBATCH --output=/home/szhan81/chexpert-labeler/demo/logs/batch_parallel_%j_%x.out
#SBATCH --error=/home/szhan81/chexpert-labeler/demo/logs/batch_parallel_%j_%x.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Enable conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chexpert-label

cd /home/szhan81/chexpert-labeler

echo "Starting batch-level parallel CheXpert processing..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Set parameters based on CPU count
NUM_WORKERS=$SLURM_CPUS_PER_TASK
# For batch parallel processing, smaller batch sizes work better
BATCH_SIZE=$((NUM_WORKERS * 50))  # Smaller multiplier for batch processing

echo "Using $NUM_WORKERS worker processes"
echo "Using batch size: $BATCH_SIZE"

# Run the batch parallel processing
python -u label_csv_parallel.py \
    --reports_path "/home/szhan81/chexpert-labeler/demo/data/Atelectasis_mention_sample_analysis.csv" \
    --output_path "/home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_batch_parallel_results.csv" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --enable_segmentation \
    --segment_length 350 \
    --segment_overlap 50 \
    --verbose

echo "End time: $(date)"
echo "Job completed!"

# Check results
if [ -f "/home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_batch_parallel_results.csv" ]; then
    echo "Output file created successfully"
    echo "Output file size: $(du -h /home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention_sample_analysis_batch_parallel_results.csv)"
    echo "Number of output rows: $(wc -l /home/szhan81/chexpert-labeler/demo/data/result/Atelectasis_mention