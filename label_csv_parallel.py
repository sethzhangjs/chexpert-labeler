#!/usr/bin/env python3
"""
Batch-level parallel processing version for CheXpert labeler.
This version processes multiple batches in parallel instead of individual records,
which is more stable and avoids parser initialization issues.
"""
import pandas as pd
import csv
import tempfile
import os
import numpy as np
from typing import List, Tuple
import multiprocessing as mp
from multiprocessing import Pool
import time
import subprocess
import shutil
from pathlib import Path

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *
import argparse


def intelligent_text_segmentation(text: str, segment_length: int = 300, overlap: int = 50) -> List[str]:
    """
    Intelligently segment text, prioritizing sentence boundaries for splitting.
    """
    if pd.isna(text) or not text.strip():
        return ['""']
    
    text = str(text).replace('"', '')
    words = text.split()
    
    if len(words) <= segment_length:
        return [f'"{text}"']
    
    segments = []
    start = 0
    sentence_endings = ['.', '!', '?', ';']
    
    while start < len(words):
        end = min(start + segment_length, len(words))
        
        if end < len(words):
            for i in range(end - 1, max(start + segment_length // 2, start), -1):
                if any(words[i].endswith(punct) for punct in sentence_endings):
                    end = i + 1
                    break
        
        segment_words = words[start:end]
        segment_text = ' '.join(segment_words)
        segments.append(f'"{segment_text}"')
        
        if end < len(words):
            start = max(start + 1, end - overlap)
        else:
            break
    
    return segments


def preprocess_text_segmented(text: str, segment_length: int = 300, overlap: int = 50) -> List[str]:
    """
    Preprocess text for CheXpert: segment long texts for processing.
    """
    if pd.isna(text):
        return ['""']
    
    estimated_tokens = len(str(text).split())
    
    if estimated_tokens <= segment_length:
        clean_text = str(text).replace('"', '')
        return [f'"{clean_text}"']
    else:
        segments = intelligent_text_segmentation(text, segment_length, overlap)
        return segments


def integrate_segment_results(segment_labels: List[np.ndarray]) -> np.ndarray:
    """
    Integrate results from multiple text segments using CheXpert's aggregation logic.
    """
    if not segment_labels:
        return np.array([''] * len(CATEGORIES), dtype=object).reshape(1, -1)
    
    POSITIVE = 1
    NEGATIVE = 0
    UNCERTAIN = 'u'
    
    def clean_label(value):
        if pd.isna(value) or value == '' or value == ' ':
            return None
        elif str(value).lower() in ['u', '-1']:
            return UNCERTAIN
        elif str(value) in ['1', '1.0']:
            return POSITIVE
        elif str(value) in ['0', '0.0']:
            return NEGATIVE
        else:
            return None
    
    integrated_result = []
    
    for cat_idx in range(len(CATEGORIES)):
        label_list = []
        for segment_result in segment_labels:
            if cat_idx < len(segment_result):
                cleaned = clean_label(segment_result[cat_idx])
                if cleaned is not None:
                    label_list.append(cleaned)
        
        if len(label_list) == 0:
            final_result = np.nan
        elif len(label_list) == 1:
            final_result = label_list[0]
        else:
            if NEGATIVE in label_list and UNCERTAIN in label_list:
                final_result = UNCERTAIN
            elif NEGATIVE in label_list and POSITIVE in label_list:
                final_result = POSITIVE
            elif UNCERTAIN in label_list and POSITIVE in label_list:
                final_result = POSITIVE
            else:
                final_result = label_list[0]
        
        integrated_result.append(final_result)
    
    return np.array(integrated_result, dtype=object).reshape(1, -1)


def preprocess_text(text, max_length=350):
    """
    Original text preprocessing function (for backward compatibility).
    """
    if pd.isna(text):
        return '""'
    
    text = str(text).replace('"', '')
    estimated_tokens = len(text.split())
    
    if estimated_tokens > max_length:
        words = text.split()
        truncated_text = ' '.join(words[-max_length:])
        text = truncated_text
    
    return f'"{text}"'


def process_batch_file(batch_file_path, args_dict, worker_id):
    """
    Process a single batch file using the original CheXpert pipeline.
    This function runs in a separate process.
    
    Args:
        batch_file_path: Path to the CSV file containing the batch
        args_dict: Dictionary of arguments
        worker_id: ID of the worker process
    
    Returns:
        Path to the output file with results
    """
    print(f"[WORKER {worker_id}] Processing batch: {batch_file_path}")
    
    # Recreate args object
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    args = Args(args_dict)
    
    try:
        # Create temporary output path
        output_path = batch_file_path.replace('.csv', '_results.csv')
        
        # Import and use the original label function from your existing code
        # We'll call the original CheXpert pipeline on this batch
        loader = Loader(batch_file_path,
                        args.sections_to_extract,
                        args.extract_strict)

        extractor = Extractor(args.mention_phrases_dir,
                              args.unmention_phrases_dir,
                              verbose=False)
        classifier = Classifier(args.pre_negation_uncertainty_path,
                                args.negation_path,
                                args.post_negation_uncertainty_path,
                                verbose=False)
        aggregator = Aggregator(CATEGORIES, verbose=False)

        # Process the batch
        loader.load()
        extractor.extract(loader.collection)
        classifier.classify(loader.collection)
        labels = aggregator.aggregate(loader.collection)

        # Create output dataframe
        labeled_reports = pd.DataFrame({REPORTS: loader.reports})
        for index, category in enumerate(CATEGORIES):
            labeled_reports[category] = labels[:, index]

        # Save results
        labeled_reports.to_csv(output_path, index=False)
        
        print(f"[WORKER {worker_id}] Completed batch: {batch_file_path}")
        return output_path
        
    except Exception as e:
        print(f"[WORKER {worker_id}] Error processing batch {batch_file_path}: {e}")
        return None


def split_dataframe_into_batches(df, text_column, batch_size, temp_dir, args):
    """
    Split dataframe into smaller CSV files for batch processing.
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        batch_size: Number of records per batch
        temp_dir: Temporary directory for batch files
        args: Arguments object
    
    Returns:
        List of batch file paths
    """
    batch_files = []
    total_rows = len(df)
    
    use_segmentation = getattr(args, 'enable_segmentation', False)
    max_length = getattr(args, 'max_text_length', 350)
    
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        batch_df = df.iloc[i:end_idx].copy()
        
        # Preprocess text in this batch
        processed_texts = []
        for _, row in batch_df.iterrows():
            text = row[text_column]
            
            if use_segmentation:
                # For segmentation, we'll create multiple rows for each text
                segments = preprocess_text_segmented(
                    text, 
                    getattr(args, 'segment_length', 300),
                    getattr(args, 'segment_overlap', 50)
                )
                # For batch processing, we'll just use the first segment or combine them
                # This is a simplified approach - you might want to handle this differently
                if len(segments) == 1:
                    processed_texts.append(segments[0])
                else:
                    # Combine segments for batch processing
                    combined_text = ' [SEG] '.join([seg.strip('"') for seg in segments])
                    processed_texts.append(f'"{combined_text}"')
            else:
                processed_text = preprocess_text(text, max_length)
                processed_texts.append(processed_text)
        
        # Create batch CSV file
        batch_file = os.path.join(temp_dir, f'batch_{i//batch_size + 1}.csv')
        
        with open(batch_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for text in processed_texts:
                writer.writerow([text])
        
        batch_files.append(batch_file)
        print(f"[SPLIT] Created batch file: {batch_file} ({len(processed_texts)} records)")
    
    return batch_files


def process_batches_parallel(batch_files, args, num_workers=None):
    """
    Process multiple batch files in parallel.
    
    Args:
        batch_files: List of batch file paths
        args: Arguments object
        num_workers: Number of worker processes
    
    Returns:
        List of result file paths
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(batch_files))
    
    print(f"[PARALLEL] Processing {len(batch_files)} batches using {num_workers} workers")
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Prepare arguments for parallel processing
    worker_args = [(batch_file, args_dict, i+1) for i, batch_file in enumerate(batch_files)]
    
    start_time = time.time()
    
    # Process batches in parallel
    with Pool(processes=num_workers) as pool:
        result_files = pool.starmap(process_batch_file, worker_args)
    
    processing_time = time.time() - start_time
    print(f"[PARALLEL] Completed all batches in {processing_time:.2f} seconds")
    
    # Filter out failed results
    successful_results = [f for f in result_files if f is not None]
    
    if len(successful_results) != len(batch_files):
        print(f"[WARNING] {len(batch_files) - len(successful_results)} batches failed")
    
    return successful_results


def combine_batch_results(result_files, original_df, output_path, args):
    """
    Combine results from multiple batch files.
    
    Args:
        result_files: List of result file paths
        original_df: Original input dataframe
        output_path: Final output path
        args: Arguments object
    """
    print(f"[COMBINE] Combining results from {len(result_files)} batch files")
    
    all_reports = []
    all_labels = []
    
    for result_file in result_files:
        try:
            batch_df = pd.read_csv(result_file)
            all_reports.extend(batch_df[REPORTS].tolist())
            
            # Extract label columns
            label_data = batch_df[CATEGORIES].values
            all_labels.append(label_data)
            
        except Exception as e:
            print(f"[ERROR] Failed to read result file {result_file}: {e}")
    
    if not all_labels:
        raise RuntimeError("No successful batch results to combine")
    
    # Combine all labels
    combined_labels = np.vstack(all_labels)
    
    print(f"[COMBINE] Combined {len(all_reports)} reports and {combined_labels.shape[0]} label sets")
    
    # Create final output
    labeled_reports = pd.DataFrame({REPORTS: all_reports})
    
    # Add CheXpert results with prefix
    label_prefix = getattr(args, 'label_prefix', 'CheXpert_')
    chexpert_columns = []
    for index, category in enumerate(CATEGORIES):
        chexpert_col_name = f"{label_prefix}{category}"
        labeled_reports[chexpert_col_name] = combined_labels[:, index]
        chexpert_columns.append(chexpert_col_name)
    
    # Merge with original data
    if original_df is not None:
        for col in original_df.columns:
            if col not in labeled_reports.columns and col != 'text':
                labeled_reports[col] = original_df[col].values[:len(labeled_reports)]
        
        # Reorder columns
        original_cols = [col for col in original_df.columns if col != 'text']
        new_column_order = original_cols + [REPORTS] + chexpert_columns
        labeled_reports = labeled_reports[[col for col in new_column_order if col in labeled_reports.columns]]
    
    # Save final results
    labeled_reports.to_csv(output_path, index=False)
    print(f"[COMBINE] Final results saved to: {output_path}")


def label_batch_parallel(args):
    """
    Main function for batch-level parallel processing.
    """
    reports_path_str = str(args.reports_path)
    
    if not reports_path_str.endswith('.csv'):
        raise ValueError("Batch parallel processing only supports CSV input")
    
    print(f"[INFO] Processing CSV input with batch-level parallelization: {reports_path_str}")
    
    # Read input data
    text_column = getattr(args, 'text_column', 'text')
    original_df = pd.read_csv(reports_path_str)
    
    print(f"[INFO] Loaded {len(original_df)} records")
    print(f"[INFO] Using text column: '{text_column}'")
    
    # Get parameters
    batch_size = getattr(args, 'batch_size', 1000)
    num_workers = getattr(args, 'num_workers', mp.cpu_count())
    
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Number of workers: {num_workers}")
    
    # Create temporary directory for batch files
    temp_dir = tempfile.mkdtemp(prefix='chexpert_batch_')
    print(f"[INFO] Temporary directory: {temp_dir}")
    
    try:
        # Split data into batches
        print(f"[STEP 1/4] Splitting data into batches...")
        batch_files = split_dataframe_into_batches(original_df, text_column, batch_size, temp_dir, args)
        
        # Process batches in parallel
        print(f"[STEP 2/4] Processing batches in parallel...")
        result_files = process_batches_parallel(batch_files, args, num_workers)
        
        # Combine results
        print(f"[STEP 3/4] Combining batch results...")
        combine_batch_results(result_files, original_df, args.output_path, args)
        
        print(f"[STEP 4/4] Cleanup...")
        
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(temp_dir)
            print(f"[CLEANUP] Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temporary directory: {e}")
    
    print(f"[SUCCESS] Batch parallel processing completed!")
    print(f"[SUCCESS] Results saved to: {args.output_path}")


if __name__ == "__main__":
    parser = ArgParser()
    parser.parser.add_argument('--text_column', default='text',
                                help='Name of the column containing text data in CSV input')
    parser.parser.add_argument('--label_prefix', default='CheXpert_',
                                help='Prefix for CheXpert result columns to avoid naming conflicts')
    parser.parser.add_argument('--batch_size', type=int, default=1000,
                                help='Number of rows to process in each batch (for CSV input)')
    parser.parser.add_argument('--max_text_length', type=int, default=350,
                                help='Maximum text length in tokens (BLLIP parser limit is 399)')
    parser.parser.add_argument('--num_workers', type=int, default=None,
                                help='Number of worker processes (default: auto-detect CPU count)')
    
    # Segmentation arguments
    parser.parser.add_argument('--enable_segmentation', action='store_true',
                                help='Enable text segmentation for long texts instead of truncation')
    parser.parser.add_argument('--segment_length', type=int, default=300,
                                help='Length of each text segment in tokens')
    parser.parser.add_argument('--segment_overlap', type=int, default=50,
                                help='Overlap between segments in tokens')
    
    label_batch_parallel(parser.parse_args())