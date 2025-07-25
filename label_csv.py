"""Entry-point script to label radiology reports."""
import pandas as pd
import csv
import tempfile
import os

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *


def preprocess_text(text, max_length=350):
    """
    Preprocess text for CheXpert labeler:
    1. Remove existing quotes
    2. Truncate if too long (keeping the LAST max_length tokens for medical impression)
    3. Wrap the entire text in quotes
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
        truncated_text = ' '.join(words[-max_length:])  # Take last max_length words
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
        dataframe: Original dataframe for later merging
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


def process_batch(batch_df, text_column, batch_num, total_batches, args, 
                 extractor, classifier, aggregator):
    """
    Process a single batch of data.
    
    Returns:
        numpy.ndarray: Labels for this batch
    """
    print(f"[BATCH {batch_num}/{total_batches}] Processing {len(batch_df)} reports...")
    
    # Create temporary file for this batch
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as temp_file:
            writer = csv.writer(temp_file)
            
            # Process each text entry in this batch
            for idx, row in batch_df.iterrows():
                text = row[text_column]
                # Use max_length from args if available
                max_length = getattr(args, 'max_text_length', 350)
                processed_text = preprocess_text(text, max_length)
                writer.writerow([processed_text])
        
        print(f"[BATCH {batch_num}/{total_batches}] Created temporary file for batch")
        
        # Initialize loader for this batch
        batch_loader = Loader(temp_path,
                            args.sections_to_extract,
                            args.extract_strict)
        
        # Process this batch
        print(f"[BATCH {batch_num}/{total_batches}] Stage 1/4: Loading reports...")
        batch_loader.load()
        
        print(f"[BATCH {batch_num}/{total_batches}] Stage 2/4: Extracting mentions...")
        extractor.extract(batch_loader.collection)
        
        print(f"[BATCH {batch_num}/{total_batches}] Stage 3/4: Classifying mentions...")
        classifier.classify(batch_loader.collection)
        
        print(f"[BATCH {batch_num}/{total_batches}] Stage 4/4: Aggregating mentions...")
        batch_labels = aggregator.aggregate(batch_loader.collection)
        
        print(f"[BATCH {batch_num}/{total_batches}] Completed! Generated {batch_labels.shape[0]} labels")
        
        return batch_labels, batch_loader.reports
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def write_enhanced(reports, labels, output_path, original_df=None, verbose=False, label_prefix="CheXpert_"):
    """
    Write labeled reports to specified path with original data preserved.
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
    """Label the provided report(s)."""
    
    # Convert PosixPath to string if necessary
    reports_path_str = str(args.reports_path)
    
    # Check if input is CSV file
    if reports_path_str.endswith('.csv'):
        verbose = getattr(args, 'verbose', False)
        if verbose:
            print(f"Processing CSV input: {reports_path_str}")
            
        # Get batch size from args (argparse already handles default value)
        batch_size = args.batch_size
        print(f"[INFO] Using batch size: {batch_size}")
        
        # Read CSV file
        original_df = prepare_csv_input(
            reports_path_str, 
            getattr(args, 'text_column', 'text')
        )
        
        # Initialize components (shared across batches)
        extractor = Extractor(args.mention_phrases_dir,
                              args.unmention_phrases_dir,
                              verbose=args.verbose)
        classifier = Classifier(args.pre_negation_uncertainty_path,
                                args.negation_path,
                                args.post_negation_uncertainty_path,
                                verbose=args.verbose)
        aggregator = Aggregator(CATEGORIES,
                                verbose=args.verbose)
        
        # Process in batches
        total_rows = len(original_df)
        total_batches = (total_rows + batch_size - 1) // batch_size
        print(f"[INFO] Processing {total_rows} rows in {total_batches} batches")
        
        all_labels = []
        all_reports = []
        
        text_column = getattr(args, 'text_column', 'text')
        
        for i in range(0, total_rows, batch_size):
            batch_num = i // batch_size + 1
            end_idx = min(i + batch_size, total_rows)
            batch_df = original_df.iloc[i:end_idx].copy()
            
            try:
                batch_labels, batch_reports = process_batch(
                    batch_df, text_column, batch_num, total_batches, args,
                    extractor, classifier, aggregator
                )
                
                all_labels.append(batch_labels)
                all_reports.extend(batch_reports)
                
                print(f"[PROGRESS] Completed {batch_num}/{total_batches} batches "
                      f"({end_idx}/{total_rows} rows)")
                
            except ValueError as e:
                if "Sentence is too long" in str(e):
                    print(f"[ERROR] Batch {batch_num} contains sentences that are too long for parser")
                    print(f"[ERROR] Error details: {e}")
                    print(f"[WARNING] Skipping batch {batch_num} ({len(batch_df)} rows)")
                    
                    # Create dummy labels for skipped batch (all zeros/blanks)
                    import numpy as np
                    dummy_labels = np.full((len(batch_df), len(CATEGORIES)), '', dtype=object)
                    dummy_reports = [f"SKIPPED: Text too long" for _ in range(len(batch_df))]
                    
                    all_labels.append(dummy_labels)
                    all_reports.extend(dummy_reports)
                    
                    print(f"[PROGRESS] Skipped {batch_num}/{total_batches} batches "
                          f"({end_idx}/{total_rows} rows)")
                else:
                    print(f"[ERROR] Failed to process batch {batch_num}: {e}")
                    raise
            except Exception as e:
                print(f"[ERROR] Failed to process batch {batch_num}: {e}")
                raise
        
        # Combine all results
        print(f"[INFO] Combining results from {len(all_labels)} batches...")
        import numpy as np
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
            # Process reports
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
    
    # Add new arguments
    import argparse
    if hasattr(parser, 'parser'):
        parser.parser.add_argument('--text_column', default='text',
                                   help='Name of the column containing text data in CSV input')
        parser.parser.add_argument('--label_prefix', default='CheXpert_',
                                   help='Prefix for CheXpert result columns to avoid naming conflicts')
        parser.parser.add_argument('--batch_size', type=int, default=1000,
                                   help='Number of rows to process in each batch (for CSV input)')
        parser.parser.add_argument('--max_text_length', type=int, default=350,
                                   help='Maximum text length in tokens (BLLIP parser limit is 399)')
    
    label(parser.parse_args())