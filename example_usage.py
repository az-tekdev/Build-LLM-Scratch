#!/usr/bin/env python3
"""
Example usage script demonstrating the complete workflow.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import BPETokenizer
from model import TransformerLM
from data import load_tiny_shakespeare, create_data_loader
from inference import generate_text


def main():
    print("=" * 60)
    print("Build LLM from Scratch - Example Usage")
    print("=" * 60)
    
    # Step 1: Load sample data
    print("\n1. Loading sample data...")
    texts = load_tiny_shakespeare()[:100]  # Use first 100 samples
    print(f"   Loaded {len(texts)} text samples")
    
    # Step 2: Train tokenizer
    print("\n2. Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(texts, verbose=True)
    print(f"   Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")
    
    # Step 3: Create model
    print("\n3. Creating transformer model...")
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=256,
    )
    print(f"   Model created. Parameters: {model.get_num_params():,}")
    
    # Step 4: Test encoding/decoding
    print("\n4. Testing tokenizer...")
    test_text = "Hello world, this is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"   Original: {test_text}")
    print(f"   Encoded: {encoded[:10]}... (showing first 10 tokens)")
    print(f"   Decoded: {decoded}")
    
    # Step 5: Test model forward pass
    print("\n5. Testing model forward pass...")
    import torch
    test_tokens = tokenizer.encode("The quick brown fox")
    input_ids = torch.tensor([test_tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   ✓ Model forward pass successful!")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train tokenizer: python tokenizer_train.py")
    print("2. Train model: python train.py")
    print("3. Generate text: python inference.py")


if __name__ == "__main__":
    main()
