#!/usr/bin/env python3
"""
Entry-point script to label radiology reports with text segmentation support.

This enhanced version adds the ability to process long texts by segmenting them
into smaller chunks, processing each chunk separately, and then integrating
the results using CheXpert's original aggregation logic.
"""
import pandas as pd
import csv
import tempfile
import os
import numpy as np
from typing import List, Tuple

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *
import argparse


def intelligent_text_segmentation(text: str, segment_length: int = 300, overlap: int = 50) -> List[str]:
    """
    Intelligently segment text, prioritizing sentence boundaries for splitting.
    
    Args:
        text: Input text to segment
        segment_length: Target length of each segment in tokens
        overlap: Number of tokens to overlap between adjacent segments
    
    Returns:
        List of segmented text strings, each wrapped in quotes
    """
    if pd.isna(text) or not text.strip():
        return ['""']
    
    # Remove existing quotes and split into words
    text = str(text).replace('"', '')
    words = text.split()
    
    # If text is short enough, return as single segment
    if len(words) <= segment_length:
        return [f'"{text}"']
    
    segments = []
    start = 0
    
    # Common sentence ending punctuation
    sentence_endings = ['.', '!', '?', ';']
    
    while start < len(words):
        end = min(start + segment_length, len(words))
        
        # If not the last segment, try to split at sentence boundary
        if end < len(words):
            # Search backwards for sentence ending
            for i in range(end - 1, max(start + segment_length // 2, start), -1):
                if any(words[i].endswith(punct) for punct in sentence_endings):
                    end = i + 1
                    break
        
        # Extract current segment
        segment_words = words[start:end]
        segment_text = ' '.join(segment_words)
        segments.append(f'"{segment_text}"')
        
        # Calculate next segment start position (with overlap)
        if end < len(words):
            start = max(start + 1, end - overlap)
        else:
            break
    
    return segments


def preprocess_text_segmented(text: str, segment_length: int = 300, overlap: int = 50) -> List[str]:
    """
    Preprocess text for CheXpert: segment long texts for processing.
    
    Args:
        text: Input text
        segment_length: Target segment length in tokens
        overlap: Overlap between segments in tokens
    
    Returns:
        List of processed text segments
    """
    if pd.isna(text):
        return ['""']
    
    # Estimate token count
    estimated_tokens = len(str(text).split())
    
    if estimated_tokens <= segment_length:
        # Short text: process directly
        clean_text = str(text).replace('"', '')
        return [f'"{clean_text}"']
    else:
        # Long text: segment for processing
        print(f"[SEGMENTATION] Long text detected ({estimated_tokens} tokens), segmenting into chunks...")
        segments = intelligent_text_segmentation(text, segment_length, overlap)
        print(f"[SEGMENTATION] Created {len(segments)} segments")
        return segments


def integrate_segment_results(segment_labels: List[np.ndarray]) -> np.ndarray:
    """
    Integrate results from multiple text segments using CheXpert's aggregation logic.
    
    Based on the logic in stages/aggregate.py, this function resolves conflicts
    between segment results using the same rules as CheXpert's mention aggregation.
    
    Conflict resolution rules:
    1. NEGATIVE + UNCERTAIN → UNCERTAIN
    2. NEGATIVE + POSITIVE → POSITIVE  
    3. UNCERTAIN + POSITIVE → POSITIVE
    4. All same labels → take the first one
    
    Args:
        segment_labels: List of label arrays from each segment
    
    Returns:
        Integrated label array following CheXpert's aggregation logic
    """
    if not segment_labels:
        return np.array([''] * len(CATEGORIES), dtype=object).reshape(1, -1)
    
    # CheXpert constants (matching the original implementation)
    POSITIVE = 1
    NEGATIVE = 0
    UNCERTAIN = 'u'
    
    def clean_label(value):
        """Convert label values to CheXpert standard format."""
        if pd.isna(value) or value == '' or value == ' ':
            return None  # Blank/not mentioned
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
        # Collect all non-blank labels for this category across segments
        label_list = []
        for segment_result in segment_labels:
            if cat_idx < len(segment_result):
                cleaned = clean_label(segment_result[cat_idx])
                if cleaned is not None:  # Only add actual labels (skip blanks)
                    label_list.append(cleaned)
        
        # Apply CheXpert's aggregation logic
        if len(label_list) == 0:
            # No mentions found
            final_result = np.nan
        elif len(label_list) == 1:
            # Single label, no conflicts
            final_result = label_list[0]
        else:
            # Multiple labels: resolve conflicts using CheXpert rules
            if NEGATIVE in label_list and UNCERTAIN in label_list:
                final_result = UNCERTAIN     # NEGATIVE + UNCERTAIN → UNCERTAIN
            elif NEGATIVE in label_list and POSITIVE in label_list:
                final_result = POSITIVE      # NEGATIVE + POSITIVE → POSITIVE
            elif UNCERTAIN in label_list and POSITIVE in label_list:
                final_result = POSITIVE      # UNCERTAIN + POSITIVE → POSITIVE
            else:
                # All labels are the same, take the first one
                final_result = label_list[0]
        
        integrated_result.append(final_result)
    
    return np.array(integrated_result, dtype=object).reshape(1, -1)


def process_single_segment(segment_text: str, args, extractor, classifier, aggregator) -> Tuple[np.ndarray, str]:
    """
    Process a single text segment through the CheXpert pipeline.
    
    Args:
        segment_text: Text segment to process (already quoted)
        args: Command line arguments
        extractor, classifier, aggregator: CheXpert pipeline components
    
    Returns:
        Tuple of (label array, processed report text)
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        # Write segment to temporary file
        with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow([segment_text])
        
        # Process segment through CheXpert pipeline
        segment_loader = Loader(temp_path, 
                              args.sections_to_extract,
                              args.extract_strict)
        
        segment_loader.load()
        extractor.extract(segment_loader.collection)
        classifier.classify(segment_loader.collection)
        segment_labels = aggregator.aggregate(segment_loader.collection)
        
        return segment_labels[0], segment_loader.reports[0]
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def process_text_with_segmentation(text: str, args, extractor, classifier, aggregator) -> Tuple[np.ndarray, str]:
    """
    Process text with segmentation support for long texts.
    
    Args:
        text: Input text to process
        args: Command line arguments
        extractor, classifier, aggregator: CheXpert pipeline components
    
    Returns:
        Tuple of (integrated label array, combined report text)
    """
    segment_length = getattr(args, 'segment_length', 300)
    overlap = getattr(args, 'segment_overlap', 50)
    
    # Segment the text
    segments = preprocess_text_segmented(text, segment_length, overlap)
    
    if len(segments) == 1:
        # Short text: process directly
        return process_single_segment(segments[0], args, extractor, classifier, aggregator)
    
    # Long text: process segments separately
    print(f"[SEGMENTATION] Processing {len(segments)} segments...")
    all_segment_labels = []
    all_segment_reports = []
    
    for i, segment in enumerate(segments):
        print(f"[SEGMENT {i+1}/{len(segments)}] Processing...")
        segment_labels, segment_report = process_single_segment(segment, args, extractor, classifier, aggregator)
        all_segment_labels.append(segment_labels)
        all_segment_reports.append(segment_report.strip('"'))
    
    # Integrate results from all segments
    integrated_labels = integrate_segment_results(all_segment_labels)
    combined_report = " [SEG] ".join(all_segment_reports)
    
    return integrated_labels[0], f'"{combined_report}"'


def preprocess_text(text, max_length=350):
    """
    Original text preprocessing function (for backward compatibility).
    
    This function implements the traditional truncation approach:
    1. Remove existing quotes
    2. Truncate if too long (keeping the LAST max_length tokens)
    3. Wrap in quotes
    
    Args:
        text: Input text
        max_length: Maximum length in tokens
    
    Returns:
        Processed text string wrapped in quotes
    """
    if pd.isna(text):
        return '""'
    
    # Convert to string and remove existing quotes
    text = str(text).replace('"', '')
    
    # Check token length (rough estimation: 1 token ≈ 4-6 characters)
    # BLLIP parser limit is 399 tokens, we use 350 as safety margin
    estimated_tokens = len(text.split())
    
    if estimated_tokens > max_length:
        # Keep the LAST max_length words (medical impression usually at end)
        words = text.split()
        truncated_text = ' '.join(words[-max_length:])
        print(f"[WARNING] Truncated long text from {estimated_tokens} to {max_length} tokens (keeping end)")
        text = truncated_text
    
    # Wrap in quotes
    return f'"{text}"'


def prepare_csv_input(input_csv_path, text_column='text'):
    """
    Read CSV file and prepare it for CheXpert processing.
    
    Args:
        input_csv_path: Path to input CSV file
        text_column: Name of the column containing text data
        
    Returns:
        DataFrame: Original dataframe for later merging
    """
    print(f"[INFO] Reading CSV file: {input_csv_path}")
    
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] CSV loaded successfully. Shape: {df.shape}")
    print(f"[INFO] Available columns: {list(df.columns)}")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"[INFO] Using text column: '{text_column}'")
    
    return df


def process_batch_with_segmentation(batch_df, text_column, batch_num, total_batches, args, 
                                   extractor, classifier, aggregator):
    """
    Process a batch of records with optional segmentation support.
    
    Args:
        batch_df: DataFrame containing the batch of records to process
        text_column: Name of the text column
        batch_num: Current batch number
        total_batches: Total number of batches
        args: Command line arguments
        extractor, classifier, aggregator: CheXpert pipeline components
    
    Returns:
        Tuple of (label array, list of processed reports)
    """
    print(f"[BATCH {batch_num}/{total_batches}] Processing {len(batch_df)} reports...")
    
    use_segmentation = getattr(args, 'enable_segmentation', False)
    
    all_labels = []
    all_reports = []
    
    for idx, row in batch_df.iterrows():
        text = row[text_column]
        record_num = idx - batch_df.index[0] + 1
        print(f"[BATCH {batch_num}/{total_batches}] Record {record_num}/{len(batch_df)}")
        
        if use_segmentation:
            # Use segmentation processing
            record_labels, record_report = process_text_with_segmentation(
                text, args, extractor, classifier, aggregator
            )
            all_labels.append(record_labels)
            all_reports.append(record_report)
        else:
            # Use traditional truncation approach
            max_length = getattr(args, 'max_text_length', 350)
            processed_text = preprocess_text(text, max_length)
            
            # Process through CheXpert pipeline
            temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
            try:
                with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
                    writer = csv.writer(temp_file)
                    writer.writerow([processed_text])
                
                record_loader = Loader(temp_path,
                                    args.sections_to_extract,
                                    args.extract_strict)
                
                record_loader.load()
                extractor.extract(record_loader.collection)
                classifier.classify(record_loader.collection)
                record_labels = aggregator.aggregate(record_loader.collection)
                
                all_labels.append(record_labels[0])
                all_reports.extend(record_loader.reports)
                
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    print(f"[BATCH {batch_num}/{total_batches}] Completed! Generated {len(all_labels)} labels")
    
    return np.array(all_labels), all_reports


def write_enhanced(reports, labels, output_path, original_df=None, verbose=False, label_prefix="CheXpert_"):
    """
    Write labeled reports to specified path with original data preserved.
    
    Args:
        reports: List of processed report texts
        labels: Array of CheXpert labels
        output_path: Output file path
        original_df: Original DataFrame to merge with (optional)
        verbose: Whether to print detailed output information
        label_prefix: Prefix for CheXpert result columns
    """
    print(f"[OUTPUT] Creating output dataframe...")
    
    # Create base dataframe with reports
    labeled_reports = pd.DataFrame({REPORTS: reports})
    print(f"[OUTPUT] Added {len(reports)} reports to output")
    
    # Add CheXpert results with prefix to avoid column name conflicts
    chexpert_columns = []
    for index, category in enumerate(CATEGORIES):
        chexpert_col_name = f"{label_prefix}{category}"
        labeled_reports[chexpert_col_name] = labels[:, index]
        chexpert_columns.append(chexpert_col_name)
    
    print(f"[OUTPUT] Added {len(chexpert_columns)} CheXpert result columns")

    # If original dataframe provided, merge with original columns
    if original_df is not None:
        print(f"[OUTPUT] Merging with {len(original_df.columns)} original columns...")
        # Add original columns (except the text column which is now in REPORTS)
        for col in original_df.columns:
            if col not in labeled_reports.columns and col != 'text':
                labeled_reports[col] = original_df[col].values[:len(labeled_reports)]
        
        # Reorder columns: original columns first, then reports, then CheXpert results
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


def label(args):
    """
    Main labeling function that handles both CSV and text file inputs.
    
    Args:
        args: Parsed command line arguments
    """
    # Convert PosixPath to string if necessary
    reports_path_str = str(args.reports_path)
    
    # Check if input is CSV file
    if reports_path_str.endswith('.csv'):
        verbose = getattr(args, 'verbose', False)
        if verbose:
            print(f"Processing CSV input: {reports_path_str}")
            
        # Get processing parameters
        batch_size = args.batch_size
        print(f"[INFO] Using batch size: {batch_size}")
        
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
        
        # Initialize CheXpert components (shared across batches)
        extractor = Extractor(args.mention_phrases_dir,
                              args.unmention_phrases_dir,
                              verbose=args.verbose)
        classifier = Classifier(args.pre_negation_uncertainty_path,
                                args.negation_path,
                                args.post_negation_uncertainty_path,
                                verbose=args.verbose)
        aggregator = Aggregator(CATEGORIES, verbose=args.verbose)
        
        # Process in batches
        total_rows = len(original_df)
        total_batches = (total_rows + batch_size - 1) // batch_size
        print(f"[INFO] Processing {total_rows} rows in {total_batches} batches")
        
        all_labels = []
        all_reports = []
        
        for i in range(0, total_rows, batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, total_rows)
            batch_df = original_df.iloc[i:end_idx].copy()
            
            try:
                batch_labels, batch_reports = process_batch_with_segmentation(
                    batch_df, text_column, batch_num, total_batches, args,
                    extractor, classifier, aggregator
                )
                
                all_labels.append(batch_labels)
                all_reports.extend(batch_reports)
                
                print(f"[PROGRESS] Completed {batch_num}/{total_batches} batches ({end_idx}/{total_rows} rows)")
                
            except Exception as e:
                print(f"[ERROR] Failed to process batch {batch_num}: {e}")
                raise
        
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
        # Original behavior for direct text file input
        print(f"[INFO] Processing text file input: {reports_path_str}")
        
        loader = Loader(reports_path_str,
                        args.sections_to_extract,
                        args.extract_strict)

        extractor = Extractor(args.mention_phrases_dir,
                              args.unmention_phrases_dir,
                              verbose=args.verbose)
        classifier = Classifier(args.pre_negation_uncertainty_path,
                                args.negation_path,
                                args.post_negation_uncertainty_path,
                                verbose=args.verbose)
        aggregator = Aggregator(CATEGORIES,
                                verbose=args.verbose)

        try:
            # Process reports using original pipeline
            print(f"[STAGE 1/4] Loading reports from: {reports_path_str}")
            loader.load()
            print(f"[STAGE 1/4] Successfully loaded {len(loader.reports)} reports")
            
            print(f"[STAGE 2/4] Extracting mentions from reports...")
            extractor.extract(loader.collection)
            print(f"[STAGE 2/4] Mention extraction completed")
            
            print(f"[STAGE 3/4] Classifying mentions (negation, uncertainty detection)...")
            classifier.classify(loader.collection)
            print(f"[STAGE 3/4] Mention classification completed")
            
            print(f"[STAGE 4/4] Aggregating mentions to final labels...")
            labels = aggregator.aggregate(loader.collection)
            print(f"[STAGE 4/4] Aggregation completed. Generated labels shape: {labels.shape}")

            # Write results using original method
            print(f"[OUTPUT] Writing results to: {args.output_path}")
            labeled_reports = pd.DataFrame({REPORTS: loader.reports})
            for index, category in enumerate(CATEGORIES):
                labeled_reports[category] = labels[:, index]

            labeled_reports[[REPORTS] + CATEGORIES].to_csv(args.output_path, index=False)
            print(f"[SUCCESS] Processing completed! Results saved to: {args.output_path}")

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            raise


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
    
    # segmentation arguments
    parser.parser.add_argument('--enable_segmentation', action='store_true',
                                help='Enable text segmentation for long texts instead of truncation')
    parser.parser.add_argument('--segment_length', type=int, default=300,
                                help='Length of each text segment in tokens')
    parser.parser.add_argument('--segment_overlap', type=int, default=50,
                                help='Overlap between segments in tokens')
    
    label(parser.parse_args())