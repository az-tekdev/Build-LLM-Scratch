#!/usr/bin/env python3
"""
Script for generating text from a trained model.
"""
import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from tokenizer import BPETokenizer
from model import TransformerLM
from inference import generate_text
from config import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="top_p",
        choices=["greedy", "top_k", "top_p"],
        help="Sampling method",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k parameter (for top_k method)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p parameter (for top_p method)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = load_config(args.config)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at {args.tokenizer}")
        return
    
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Create model (need to infer config from checkpoint or use defaults)
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        pad_idx=tokenizer.vocab.get(tokenizer.pad_token, 0),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded. Parameters: {model.get_num_params():,}")
    
    # Interactive or single generation
    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive mode. Type 'quit' to exit.")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                generated = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=args.max_length,
                    method=args.method,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                )
                
                print(f"\nGenerated:\n{generated}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        print("Generating...")
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            method=args.method,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
        
        print(f"\nGenerated text:\n{generated}")


if __name__ == "__main__":
    main()
