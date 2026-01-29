"""Entry-point script to count words and BLLIP tokens in radiology reports."""
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
from args import ArgParser
from loader import Loader
from constants import *


class TokenCounter:
    """Count words and BLLIP tokens in text."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            if verbose:
                print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
    
    def count_words(self, text):
        """Count words in text using simple whitespace splitting."""
        if pd.isna(text) or text is None:
            return 0
        # Remove extra whitespace and split by whitespace
        words = text.strip().split()
        return len(words)
    
    def count_bllip_tokens(self, text):
        """Count BLLIP-style tokens using NLTK word tokenizer."""
        if pd.isna(text) or text is None:
            return 0
        try:
            # NLTK word_tokenize provides BLLIP-style tokenization
            tokens = word_tokenize(text)
            return len(tokens)
        except Exception as e:
            if self.verbose:
                print(f"Error tokenizing text: {e}")
            return 0
    
    def process_reports(self, reports):
        """Process all reports and return word and token counts."""
        word_counts = []
        bllip_token_counts = []
        
        total_reports = len(reports)
        
        for i, report in enumerate(reports):
            if self.verbose and i % 1000 == 0:
                print(f"Processing report {i+1}/{total_reports}")
            
            word_count = self.count_words(report)
            bllip_count = self.count_bllip_tokens(report)
            
            word_counts.append(word_count)
            bllip_token_counts.append(bllip_count)
        
        return word_counts, bllip_token_counts


def write_counts(reports, word_counts, bllip_counts, output_path, verbose=False):
    """Write reports with counts to specified path."""
    result_df = pd.DataFrame({
        REPORTS: reports,
        'word_count': word_counts,
        'bllip_token_count': bllip_counts
    })
    
    if verbose:
        print(f"Writing reports with counts to {output_path}")
        print(f"Total reports processed: {len(reports)}")
        print(f"Average words per report: {sum(word_counts)/len(word_counts):.2f}")
        print(f"Average BLLIP tokens per report: {sum(bllip_counts)/len(bllip_counts):.2f}")
    
    result_df.to_csv(output_path, index=False)


def count_tokens(args):
    """Count words and BLLIP tokens in the provided reports."""
    
    # Load reports
    loader = Loader(args.reports_path,
                    args.sections_to_extract,
                    args.extract_strict)
    
    # Initialize token counter
    counter = TokenCounter(verbose=args.verbose)
    
    if args.verbose:
        print("Loading reports...")
    
    # Load reports in place
    loader.load()
    
    if args.verbose:
        print(f"Loaded {len(loader.reports)} reports")
        print("Starting token counting...")
    
    # Count words and BLLIP tokens
    word_counts, bllip_counts = counter.process_reports(loader.reports)
    
    # Write results
    write_counts(loader.reports, word_counts, bllip_counts, 
                args.output_path, args.verbose)
    
    if args.verbose:
        print("Token counting completed successfully!")


if __name__ == "__main__":
    parser = ArgParser()
    count_tokens(parser.parse_args())