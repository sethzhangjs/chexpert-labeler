#!/usr/bin/env python3
"""Debug version of label.py with detailed timing."""
import pandas as pd
import time
from datetime import datetime

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *


def print_time_step(step_name, start_time):
    """Print timing information for each step."""
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {step_name} completed in {elapsed:.2f} seconds")
    return time.time()


def write(reports, labels, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[REPORTS] + CATEGORIES].to_csv(output_path, index=False)


def label(args):
    """Label the provided report(s) with detailed timing."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting labeling process...")
    overall_start = time.time()
    
    # Step 1: Initialize Loader
    step_start = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Loader...")
    loader = Loader(args.reports_path, args.sections_to_extract, args.extract_strict)
    step_start = print_time_step("Loader initialization", step_start)

    # Step 2: Initialize Extractor  
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Extractor...")
    extractor = Extractor(args.mention_phrases_dir, args.unmention_phrases_dir, verbose=args.verbose)
    step_start = print_time_step("Extractor initialization", step_start)
    
    # Step 3: Initialize Classifier
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Classifier...")
    classifier = Classifier(args.pre_negation_uncertainty_path, args.negation_path, 
                          args.post_negation_uncertainty_path, verbose=args.verbose)
    step_start = print_time_step("Classifier initialization", step_start)
    
    # Step 4: Initialize Aggregator
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Aggregator...")
    aggregator = Aggregator(CATEGORIES, verbose=args.verbose)
    step_start = print_time_step("Aggregator initialization", step_start)

    # Step 5: Load reports
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading reports...")
    loader.load()
    step_start = print_time_step("Report loading", step_start)
    print(f"Loaded {len(loader.reports)} reports")

    # Step 6: Extract mentions
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting mentions...")
    extractor.extract(loader.collection)
    step_start = print_time_step("Mention extraction", step_start)

    # Step 7: Classify mentions
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Classifying mentions...")
    classifier.classify(loader.collection)
    step_start = print_time_step("Mention classification", step_start)

    # Step 8: Aggregate mentions
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregating mentions...")
    labels = aggregator.aggregate(loader.collection)
    step_start = print_time_step("Mention aggregation", step_start)

    # Step 9: Write output
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Writing output...")
    write(loader.reports, labels, args.output_path, args.verbose)
    step_start = print_time_step("Output writing", step_start)
    
    total_time = time.time() - overall_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] TOTAL PROCESSING TIME: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args())