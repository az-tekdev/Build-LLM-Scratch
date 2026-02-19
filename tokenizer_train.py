#!/usr/bin/env python3
"""
Script to train and save a BPE tokenizer.
"""
import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import BPETokenizer
from data import load_tiny_shakespeare, load_wikitext_small


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "wikitext"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tokenizer.json",
        help="Output path for tokenizer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    if args.dataset == "tiny_shakespeare":
        texts = load_tiny_shakespeare()
    elif args.dataset == "wikitext":
        texts = load_wikitext_small()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Loaded {len(texts)} text samples")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    
    # Train tokenizer
    tokenizer.train(texts, verbose=args.verbose)
    
    # Save tokenizer
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")


if __name__ == "__main__":
    main()
