#!/usr/bin/env python3
"""
Main training script for the transformer language model.
"""
import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from tokenizer import BPETokenizer
from model import TransformerLM
from data import create_data_loader, load_tiny_shakespeare, load_wikitext_small
from trainer import Trainer
from config import load_config, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./tokenizer.json",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "wikitext"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.checkpoint_dir is not None:
        config.paths.checkpoint_dir = args.checkpoint_dir
    
    print("=" * 60)
    print("Transformer Language Model Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model config: {config.model.__dict__}")
    print(f"Training config: {config.training.__dict__}")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    if not os.path.exists(args.tokenizer):
        print(f"Tokenizer not found at {args.tokenizer}")
        print("Please train the tokenizer first using tokenizer_train.py")
        return
    
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    if args.dataset == "tiny_shakespeare":
        texts = load_tiny_shakespeare()
    elif args.dataset == "wikitext":
        texts = load_wikitext_small()
    
    # Split into train/val
    split_idx = int(len(texts) * config.data.train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_data_loader(
        train_texts,
        tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length,
        pad_idx=tokenizer.vocab.get(tokenizer.pad_token, 0),
        shuffle=True,
    )
    
    val_loader = create_data_loader(
        val_texts,
        tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length,
        pad_idx=tokenizer.vocab.get(tokenizer.pad_token, 0),
        shuffle=False,
    )
    
    # Create model
    print("\nCreating model...")
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
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        warmup_steps=config.training.warmup_steps,
        save_dir=config.paths.checkpoint_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {epoch}")
    
    # Train
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_every=config.training.save_every,
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
