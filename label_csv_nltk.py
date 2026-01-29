#!/usr/bin/env python3
"""
Entry-point script to label radiology reports with sentence-level processing.

This enhanced version uses NLTK's punkt sentence tokenizer to split texts
into individual sentences, processes each sentence separately, and then
integrates the results using CheXpert's original aggregation logic.
"""
import pandas as pd
import csv
import tempfile
import os
import numpy as np
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import time

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *
import argparse


def ensure_nltk_data():
    """
    Ensure required NLTK data is downloaded.
    Downloads punkt tokenizer data if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[NLTK] Downloading punkt tokenizer data...")
        nltk.download('punkt', quiet=True)
        print("[NLTK] Punkt tokenizer data downloaded successfully")


def intelligent_sentence_segmentation(text: str, min_sentence_length: int = 5) -> List[str]:
    """
    Split text into sentences using NLTK's punkt tokenizer.
    
    Args:
        text: Input text to segment
        min_sentence_length: Minimum number of characters for a valid sentence
    
    Returns:
        List of sentence strings, each wrapped in quotes
    """
    if pd.isna(text) or not text.strip():
        return ['""']
    
    # Remove existing quotes and clean text
    text = str(text).replace('"', '').strip()
    
    # Use NLTK's sentence tokenizer
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        print(f"[WARNING] Sentence tokenization failed: {e}, using fallback method")
        # Fallback to simple period-based splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Filter out very short sentences and clean up
    valid_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= min_sentence_length:
            valid_sentences.append(f'"{sentence}"')
    
    # Return at least one sentence if no valid sentences found
    if not valid_sentences:
        return [f'"{text}"']
    
    return valid_sentences


def preprocess_text_sentences(text: str, min_sentence_length: int = 5) -> List[str]:
    """
    Preprocess text for CheXpert: split into sentences for processing.
    
    Args:
        text: Input text
        min_sentence_length: Minimum sentence length in characters
    
    Returns:
        List of processed sentence strings
    """
    if pd.isna(text):
        return ['""']
    
    # Get sentences
    sentences = intelligent_sentence_segmentation(text, min_sentence_length)
    
    if len(sentences) > 1:
        print(f"[SENTENCES] Text split into {len(sentences)} sentences")
    
    return sentences


def integrate_sentence_results(sentence_labels: List[np.ndarray]) -> np.ndarray:
    """
    Integrate results from multiple sentences using CheXpert's aggregation logic.
    
    Based on the logic in stages/aggregate.py, this function resolves conflicts
    between sentence results using the same rules as CheXpert's mention aggregation.
    
    Conflict resolution rules:
    1. NEGATIVE + UNCERTAIN → UNCERTAIN
    2. NEGATIVE + POSITIVE → POSITIVE  
    3. UNCERTAIN + POSITIVE → POSITIVE
    4. All same labels → take the first one
    
    Args:
        sentence_labels: List of label arrays from each sentence
    
    Returns:
        Integrated label array following CheXpert's aggregation logic
    """
    if not sentence_labels:
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
        # Collect all non-blank labels for this category across sentences
        label_list = []
        for sentence_result in sentence_labels:
            if cat_idx < len(sentence_result):
                cleaned = clean_label(sentence_result[cat_idx])
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


def process_single_sentence(sentence_text: str, args, extractor, classifier, aggregator, sentence_num: int = None) -> Tuple[np.ndarray, str]:
    """
    Process a single sentence through the CheXpert pipeline.
    
    Args:
        sentence_text: Sentence text to process (already quoted)
        args: Command line arguments
        extractor, classifier, aggregator: CheXpert pipeline components
        sentence_num: Sentence number for logging (optional)
    
    Returns:
        Tuple of (label array, processed sentence text)
    """
    start_time = time.time()
    
    # Print sentence if enabled
    if getattr(args, 'print_sentences', False):
        sentence_display = sentence_text.strip('"')
        if sentence_num is not None:
            print(f"[SENTENCE {sentence_num}] Processing: {sentence_display}")
        else:
            print(f"[SENTENCE] Processing: {sentence_display}")
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        # Write sentence to temporary file
        with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow([sentence_text])
        
        # Process sentence through CheXpert pipeline
        sentence_loader = Loader(temp_path, 
                              args.sections_to_extract,
                              args.extract_strict)
        
        sentence_loader.load()
        extractor.extract(sentence_loader.collection)
        classifier.classify(sentence_loader.collection)
        sentence_labels = aggregator.aggregate(sentence_loader.collection)
        
        # Calculate and print timing if enabled
        elapsed_time = time.time() - start_time
        if getattr(args, 'print_timing', False):
            if sentence_num is not None:
                print(f"[TIMING] Sentence {sentence_num} processed in {elapsed_time:.3f}s")
            else:
                print(f"[TIMING] Sentence processed in {elapsed_time:.3f}s")
        
        return sentence_labels[0], sentence_loader.reports[0]
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def process_text_with_sentences(text: str, args, extractor, classifier, aggregator) -> Tuple[np.ndarray, str]:
    """
    Process text with sentence-level processing for long texts.
    
    Args:
        text: Input text to process
        args: Command line arguments
        extractor, classifier, aggregator: CheXpert pipeline components
    
    Returns:
        Tuple of (integrated label array, combined sentence text)
    """
    start_time = time.time()
    min_sentence_length = getattr(args, 'min_sentence_length', 5)
    
    # Split the text into sentences
    sentences = preprocess_text_sentences(text, min_sentence_length)
    
    if len(sentences) == 1:
        # Single sentence: process directly
        result = process_single_sentence(sentences[0], args, extractor, classifier, aggregator)
        
        # Print timing for single sentence if enabled
        elapsed_time = time.time() - start_time
        if getattr(args, 'print_timing', False):
            print(f"[TIMING] Single sentence record processed in {elapsed_time:.3f}s")
        
        return result
    
    # Multiple sentences: process separately
    print(f"[SENTENCES] Processing {len(sentences)} sentences...")
    all_sentence_labels = []
    all_sentence_reports = []
    
    for i, sentence in enumerate(sentences):
        sentence_labels, sentence_report = process_single_sentence(
            sentence, args, extractor, classifier, aggregator, sentence_num=i+1
        )
        all_sentence_labels.append(sentence_labels)
        all_sentence_reports.append(sentence_report.strip('"'))
    
    # Integrate results from all sentences
    integrated_labels = integrate_sentence_results(all_sentence_labels)
    combined_report = " [SENT] ".join(all_sentence_reports)
    
    # Print timing for multi-sentence record if enabled
    elapsed_time = time.time() - start_time
    if getattr(args, 'print_timing', False):
        print(f"[TIMING] Multi-sentence record ({len(sentences)} sentences) processed in {elapsed_time:.3f}s")
    
    return integrated_labels[0], f'"{combined_report}"'


def preprocess_text(text):
    """
    Simple text preprocessing function for traditional mode.
    
    This function implements the traditional approach:
    1. Remove existing quotes
    2. Wrap in quotes (no truncation)
    
    Args:
        text: Input text
    
    Returns:
        Processed text string wrapped in quotes
    """
    if pd.isna(text):
        return '""'
    
    # Convert to string and remove existing quotes
    text = str(text).replace('"', '')
    
    # Wrap in quotes (no truncation)
    return f'"{text}"'


def prepare_csv_input(input_csv_path, text_column='text', max_num=None):
    """
    Read CSV file and prepare it for CheXpert processing.
    
    Args:
        input_csv_path: Path to input CSV file
        text_column: Name of the column containing text data
        max_num: Maximum number of records to process (None for all records)
        
    Returns:
        DataFrame: Original dataframe for later merging (limited to max_num if specified)
    """
    print(f"[INFO] Reading CSV file: {input_csv_path}")
    
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] CSV loaded successfully. Original shape: {df.shape}")
    print(f"[INFO] Available columns: {list(df.columns)}")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"[INFO] Using text column: '{text_column}'")
    
    # Apply max_num limit if specified
    if max_num is not None and max_num > 0:
        original_len = len(df)
        df = df.head(max_num)
        print(f"[INFO] Limited processing to first {len(df)} records (out of {original_len} total records)")
    else:
        print(f"[INFO] Processing all {len(df)} records")
    
    return df


def process_batch_with_sentences(batch_df, text_column, batch_num, total_batches, args, 
                                   extractor, classifier, aggregator):
    """
    Process a batch of records with sentence-level processing.
    
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
    batch_start_time = time.time()
    print(f"[BATCH {batch_num}/{total_batches}] Processing {len(batch_df)} reports...")
    
    use_sentences = getattr(args, 'enable_sentences', False)
    
    all_labels = []
    all_reports = []
    
    for idx, row in batch_df.iterrows():
        record_start_time = time.time()
        text = row[text_column]
        record_num = idx - batch_df.index[0] + 1
        print(f"[BATCH {batch_num}/{total_batches}] Record {record_num}/{len(batch_df)}")
        
        if use_sentences:
            # Use sentence-level processing
            record_labels, record_report = process_text_with_sentences(
                text, args, extractor, classifier, aggregator
            )
            all_labels.append(record_labels)
            all_reports.append(record_report)
        else:
            # Use traditional approach: process full text without truncation
            processed_text = preprocess_text(text)
            
            # Print text if enabled (for traditional mode)
            if getattr(args, 'print_sentences', False):
                text_display = processed_text.strip('"')
                print(f"[TRADITIONAL] Processing full text: {text_display}")
            
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
        
        # Print record timing if enabled
        if getattr(args, 'print_timing', False):
            record_elapsed = time.time() - record_start_time
            print(f"[TIMING] Record {record_num} completed in {record_elapsed:.3f}s")
    
    batch_elapsed = time.time() - batch_start_time
    print(f"[BATCH {batch_num}/{total_batches}] Completed! Generated {len(all_labels)} labels")
    
    if getattr(args, 'print_timing', False):
        print(f"[TIMING] Batch {batch_num} completed in {batch_elapsed:.3f}s")
    
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

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[OUTPUT] Created output directory: {output_dir}")
        
    # Write to CSV
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
    total_start_time = time.time()
    
    # Ensure NLTK punkt data is available
    ensure_nltk_data()
    
    # Convert PosixPath to string if necessary
    reports_path_str = str(args.reports_path)
    
    # Check if input is CSV file
    if reports_path_str.endswith('.csv'):
        verbose = getattr(args, 'verbose', False)
        if verbose:
            print(f"Processing CSV input: {reports_path_str}")
            
        # Get processing parameters
        batch_size = args.batch_size
        max_num = getattr(args, 'max_num', None)
        print(f"[INFO] Using batch size: {batch_size}")
        
        if max_num is not None:
            print(f"[INFO] Maximum records to process: {max_num}")
        else:
            print(f"[INFO] Processing all records in the file")
        
        # Check processing mode
        use_sentences = getattr(args, 'enable_sentences', False)
        if use_sentences:
            min_sentence_length = getattr(args, 'min_sentence_length', 5)
            print(f"[INFO] Sentence-level processing enabled - min_sentence_length: {min_sentence_length}")
        else:
            print(f"[INFO] Using traditional mode: processing full text without modification")
        
        # Print debug options status
        if getattr(args, 'print_sentences', False):
            print(f"[INFO] Sentence printing enabled")
        if getattr(args, 'print_timing', False):
            print(f"[INFO] Timing information enabled")
        
        # Read CSV file with max_num limit
        text_column = getattr(args, 'text_column', 'text')
        original_df = prepare_csv_input(reports_path_str, text_column, max_num)
        
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
                batch_labels, batch_reports = process_batch_with_sentences(
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
        
        # Print total processing time
        total_elapsed = time.time() - total_start_time
        print(f"[SUCCESS] Processing completed! Results saved to: {args.output_path}")
        if getattr(args, 'print_timing', False):
            print(f"[TIMING] Total processing time: {total_elapsed:.3f}s")
        
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
            
            # Print total processing time for text file input
            total_elapsed = time.time() - total_start_time
            print(f"[SUCCESS] Processing completed! Results saved to: {args.output_path}")
            if getattr(args, 'print_timing', False):
                print(f"[TIMING] Total processing time: {total_elapsed:.3f}s")

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            raise


if __name__ == "__main__":
    parser = ArgParser()
    parser.parser.add_argument('--text_column', default='text',
                                   help='Name of the column containing text data in CSV input')
    parser.parser.add_argument('--label_prefix', default='CheXpert_',
                                help='Prefix for CheXpert result columns to avoid naming conflicts')
    parser.parser.add_argument('--batch_size', type=int, default=100,
                                help='Number of rows to process in each batch (for CSV input)')

    parser.parser.add_argument('--max_num', type=int, default=None,
                                help='Maximum number of records to process from CSV file (None for all records)')
    
    # sentence processing arguments
    parser.parser.add_argument('--enable_sentences', action='store_true',
                                help='Enable sentence-level processing for long texts instead of truncation')
    parser.parser.add_argument('--min_sentence_length', type=int, default=5,
                                help='Minimum sentence length in characters')
    
    # debugging and monitoring arguments
    parser.parser.add_argument('--print_sentences', action='store_true',
                                help='Print each sentence/text being processed (useful for debugging)')
    parser.parser.add_argument('--print_timing', action='store_true',
                                help='Print timing information for sentences, records, and batches')
    
    label(parser.parse_args())