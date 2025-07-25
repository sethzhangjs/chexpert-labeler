#!/usr/bin/env python3
"""
Parallel version of CheXpert labeler with multiprocessing support.
"""
import pandas as pd
import csv
import tempfile
import os
import numpy as np
from typing import List, Tuple
import multiprocessing as mp
from multiprocessing import Pool, Manager
import pickle
import time
from functools import partial

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


def create_chexpert_components(args):
    """
    Create CheXpert components (extractor, classifier, aggregator).
    This function is used to initialize components in each worker process.
    """
    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=False)  # Disable verbose in workers
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=False)
    aggregator = Aggregator(CATEGORIES, verbose=False)
    
    return extractor, classifier, aggregator


def process_single_record_parallel(record_data, args_dict):
    """
    Process a single record in parallel worker process.
    
    Args:
        record_data: Tuple of (index, text) for the record
        args_dict: Dictionary containing arguments
    
    Returns:
        Tuple of (index, labels, processed_report)
    """
    idx, text = record_data
    
    # Recreate args object from dictionary
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    args = Args(args_dict)
    
    # Create CheXpert components for this worker
    extractor, classifier, aggregator = create_chexpert_components(args)
    
    try:
        if getattr(args, 'enable_segmentation', False):
            # Use segmentation processing
            record_labels, record_report = process_text_with_segmentation_worker(
                text, args, extractor, classifier, aggregator
            )
        else:
            # Use traditional truncation approach
            max_length = getattr(args, 'max_text_length', 350)
            processed_text = preprocess_text(text, max_length)
            record_labels, record_report = process_single_segment_worker(
                processed_text, args, extractor, classifier, aggregator
            )
        
        return idx, record_labels, record_report
        
    except Exception as e:
        print(f"Error processing record {idx}: {e}")
        # Return empty results for failed records
        empty_labels = np.array([np.nan] * len(CATEGORIES), dtype=object)
        return idx, empty_labels, '""'


def process_single_segment_worker(segment_text: str, args, extractor, classifier, aggregator) -> Tuple[np.ndarray, str]:
    """
    Process a single text segment through the CheXpert pipeline (worker version).
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow([segment_text])
        
        segment_loader = Loader(temp_path, 
                              args.sections_to_extract,
                              args.extract_strict)
        
        segment_loader.load()
        extractor.extract(segment_loader.collection)
        classifier.classify(segment_loader.collection)
        segment_labels = aggregator.aggregate(segment_loader.collection)
        
        return segment_labels[0], segment_loader.reports[0]
        
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def process_text_with_segmentation_worker(text: str, args, extractor, classifier, aggregator) -> Tuple[np.ndarray, str]:
    """
    Process text with segmentation support for long texts (worker version).
    """
    segment_length = getattr(args, 'segment_length', 300)
    overlap = getattr(args, 'segment_overlap', 50)
    
    segments = preprocess_text_segmented(text, segment_length, overlap)
    
    if len(segments) == 1:
        return process_single_segment_worker(segments[0], args, extractor, classifier, aggregator)
    
    # Long text: process segments separately
    all_segment_labels = []
    all_segment_reports = []
    
    for segment in segments:
        segment_labels, segment_report = process_single_segment_worker(segment, args, extractor, classifier, aggregator)
        all_segment_labels.append(segment_labels)
        all_segment_reports.append(segment_report.strip('"'))
    
    # Integrate results from all segments
    integrated_labels = integrate_segment_results(all_segment_labels)
    combined_report = " [SEG] ".join(all_segment_reports)
    
    return integrated_labels[0], f'"{combined_report}"'


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


def process_batch_parallel(batch_df, text_column, batch_num, total_batches, args, num_workers=None):
    """
    Process a batch of records using multiprocessing.
    
    Args:
        batch_df: DataFrame containing the batch of records to process
        text_column: Name of the text column
        batch_num: Current batch number
        total_batches: Total number of batches
        args: Command line arguments
        num_workers: Number of worker processes (None for auto-detect)
    
    Returns:
        Tuple of (label array, list of processed reports)
    """
    print(f"[BATCH {batch_num}/{total_batches}] Processing {len(batch_df)} reports in parallel...")
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(batch_df))
    
    print(f"[BATCH {batch_num}/{total_batches}] Using {num_workers} worker processes")
    
    # Prepare data for parallel processing
    record_data = [(idx, row[text_column]) for idx, row in batch_df.iterrows()]
    
    # Convert args to dictionary for pickling
    args_dict = vars(args)
    
    # Process records in parallel
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        worker_func = partial(process_single_record_parallel, args_dict=args_dict)
        results = pool.map(worker_func, record_data)
    
    processing_time = time.time() - start_time
    print(f"[BATCH {batch_num}/{total_batches}] Parallel processing completed in {processing_time:.2f} seconds")
    
    # Sort results by original index and extract labels and reports
    results.sort(key=lambda x: x[0])
    
    all_labels = np.array([result[1] for result in results])
    all_reports = [result[2] for result in results]
    
    print(f"[BATCH {batch_num}/{total_batches}] Generated {len(all_labels)} labels")
    
    return all_labels, all_reports


def prepare_csv_input(input_csv_path, text_column='text'):
    """
    Read CSV file and prepare it for CheXpert processing.
    """
    print(f"[INFO] Reading CSV file: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] CSV loaded successfully. Shape: {df.shape}")
    print(f"[INFO] Available columns: {list(df.columns)}")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"[INFO] Using text column: '{text_column}'")
    
    return df


def write_enhanced(reports, labels, output_path, original_df=None, verbose=False, label_prefix="CheXpert_"):
    """
    Write labeled reports to specified path with original data preserved.
    """
    print(f"[OUTPUT] Creating output dataframe...")
    
    labeled_reports = pd.DataFrame({REPORTS: reports})
    print(f"[OUTPUT] Added {len(reports)} reports to output")
    
    chexpert_columns = []
    for index, category in enumerate(CATEGORIES):
        chexpert_col_name = f"{label_prefix}{category}"
        labeled_reports[chexpert_col_name] = labels[:, index]
        chexpert_columns.append(chexpert_col_name)
    
    print(f"[OUTPUT] Added {len(chexpert_columns)} CheXpert result columns")

    if original_df is not None:
        print(f"[OUTPUT] Merging with {len(original_df.columns)} original columns...")
        for col in original_df.columns:
            if col not in labeled_reports.columns and col != 'text':
                labeled_reports[col] = original_df[col].values[:len(labeled_reports)]
        
        original_cols = [col for col in original_df.columns if col != 'text']
        new_column_order = original_cols + [REPORTS] + chexpert_columns
        labeled_reports = labeled_reports[[col for col in new_column_order if col in labeled_reports.columns]]
        print(f"[OUTPUT] Final dataframe shape: {labeled_reports.shape}")

    print(f"[OUTPUT] Saving to CSV: {output_path}")
    labeled_reports.to_csv(output_path, index=False)
    
    if verbose:
        print(f"[OUTPUT] CheXpert results saved with prefix: '{label_prefix}'")
        print(f"[OUTPUT] Total columns in output: {len(labeled_reports.columns)}")
        
    print(f"[OUTPUT] File saved successfully")


def label_parallel(args):
    """
    Main labeling function with parallel processing support.
    """
    reports_path_str = str(args.reports_path)
    
    if reports_path_str.endswith('.csv'):
        verbose = getattr(args, 'verbose', False)
        if verbose:
            print(f"Processing CSV input: {reports_path_str}")
            
        # Get processing parameters
        batch_size = args.batch_size
        num_workers = getattr(args, 'num_workers', None)
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        print(f"[INFO] Using batch size: {batch_size}")
        print(f"[INFO] Using {num_workers} worker processes per batch")
        
        # Check processing mode
        use_segmentation = getattr(args, 'enable_segmentation', False)
        if use_segmentation:
            segment_length = getattr(args, 'segment_length', 300)
            segment_overlap = getattr(args, 'segment_overlap', 50)
            print(f"[INFO] Segmentation enabled - segment_length: {segment_length}, overlap: {segment_overlap}")
        else:
            max_text_length = getattr(args, 'max_text_length', 350)
            print(f"[INFO] Using traditional truncation - max_length: {max_text_length}")
        
        # Read CSV file
        text_column = getattr(args, 'text_column', 'text')
        original_df = prepare_csv_input(reports_path_str, text_column)
        
        # Process in batches with parallel processing
        total_rows = len(original_df)
        total_batches = (total_rows + batch_size - 1) // batch_size
        print(f"[INFO] Processing {total_rows} rows in {total_batches} batches")
        
        all_labels = []
        all_reports = []
        
        total_start_time = time.time()
        
        for i in range(0, total_rows, batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, total_rows)
            batch_df = original_df.iloc[i:end_idx].copy()
            
            try:
                batch_labels, batch_reports = process_batch_parallel(
                    batch_df, text_column, batch_num, total_batches, args, num_workers
                )
                
                all_labels.append(batch_labels)
                all_reports.extend(batch_reports)
                
                print(f"[PROGRESS] Completed {batch_num}/{total_batches} batches ({end_idx}/{total_rows} rows)")
                
            except Exception as e:
                print(f"[ERROR] Failed to process batch {batch_num}: {e}")
                raise
        
        total_processing_time = time.time() - total_start_time
        print(f"[INFO] Total processing time: {total_processing_time:.2f} seconds")
        print(f"[INFO] Average time per record: {total_processing_time/total_rows:.4f} seconds")
        
        # Combine all results
        print(f"[INFO] Combining results from {len(all_labels)} batches...")
        combined_labels = np.vstack(all_labels)
        print(f"[INFO] Combined labels shape: {combined_labels.shape}")
        
        # Write results
        print(f"[OUTPUT] Writing results to: {args.output_path}")
        label_prefix = getattr(args, 'label_prefix', 'CheXpert_')
        print(f"[OUTPUT] Using label prefix: '{label_prefix}'")
        write_enhanced(all_reports, combined_labels, args.output_path, 
                      original_df, args.verbose, label_prefix)
        
        print(f"[SUCCESS] Processing completed! Results saved to: {args.output_path}")
        
    else:
        # Fallback to original single-threaded processing for text files
        print(f"[INFO] Text file input detected. Using single-threaded processing: {reports_path_str}")
        print("[WARNING] Parallel processing only supports CSV input. Use CSV format for better performance.")
        
        # Use original processing logic here
        raise NotImplementedError("Parallel processing currently only supports CSV input")


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
    
    label_parallel(parser.parse_args())