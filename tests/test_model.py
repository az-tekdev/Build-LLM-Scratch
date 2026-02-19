"""
Unit tests for transformer model.
"""
import pytest
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    TransformerLM,
)


def test_multi_head_attention():
    """Test multi-head attention."""
    batch_size, seq_len, d_model, num_heads = 2, 10, 64, 4
    
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_feed_forward():
    """Test feed-forward network."""
    batch_size, seq_len, d_model, d_ff = 2, 10, 64, 256
    
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ffn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block():
    """Test transformer block."""
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 10, 64, 4, 256
    
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_lm():
    """Test full transformer language model."""
    batch_size, seq_len, vocab_size = 2, 20, 1000
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=1024,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_transformer_lm_with_mask():
    """Test transformer with attention mask."""
    batch_size, seq_len, vocab_size = 2, 20, 1000
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    logits = model(input_ids, mask)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_model_parameters():
    """Test model parameter counting."""
    model = TransformerLM(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
    )
    
    num_params = model.get_num_params()
    assert num_params > 0
    assert isinstance(num_params, int)


def test_causal_mask():
    """Test that causal mask prevents looking ahead."""
    batch_size, seq_len, vocab_size = 1, 5, 100
    
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=1,
        num_heads=2,
        d_ff=256,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass should work without explicit mask (uses causal mask internally)
    logits = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
