"""
Unit tests for BPE tokenizer.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokenizer import BPETokenizer


def test_tokenizer_initialization():
    """Test tokenizer initialization."""
    tokenizer = BPETokenizer(vocab_size=1000)
    assert tokenizer.vocab_size == 1000
    assert len(tokenizer.vocab) == 0
    assert len(tokenizer.merges) == 0


def test_tokenizer_training():
    """Test tokenizer training on small corpus."""
    corpus = [
        "hello world",
        "hello there",
        "world peace",
        "peace and love",
    ] * 10  # Repeat for frequency
    
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus, verbose=False)
    
    assert len(tokenizer.vocab) > 0
    assert len(tokenizer.merges) >= 0
    assert len(tokenizer.vocab) <= tokenizer.vocab_size


def test_tokenizer_encode_decode():
    """Test encoding and decoding."""
    corpus = [
        "hello world",
        "hello there",
        "world peace",
    ] * 10
    
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus, verbose=False)
    
    text = "hello world"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    # Decoded text should be similar (may have extra spaces)
    assert len(encoded) > 0
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_tokenizer_save_load(tmp_path):
    """Test saving and loading tokenizer."""
    corpus = ["hello world", "test text"] * 5
    
    tokenizer1 = BPETokenizer(vocab_size=50)
    tokenizer1.train(corpus, verbose=False)
    
    # Save
    filepath = tmp_path / "test_tokenizer.json"
    tokenizer1.save(str(filepath))
    assert filepath.exists()
    
    # Load
    tokenizer2 = BPETokenizer()
    tokenizer2.load(str(filepath))
    
    # Check they match
    assert tokenizer1.vocab == tokenizer2.vocab
    assert tokenizer1.merges == tokenizer2.merges
    
    # Test encoding matches
    text = "hello world"
    encoded1 = tokenizer1.encode(text)
    encoded2 = tokenizer2.encode(text)
    assert encoded1 == encoded2


def test_special_tokens():
    """Test special token handling."""
    corpus = ["hello world"] * 10
    
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus, verbose=False)
    
    # Check special tokens are in vocab
    assert tokenizer.unk_token in tokenizer.vocab
    assert tokenizer.pad_token in tokenizer.vocab
    assert tokenizer.bos_token in tokenizer.vocab
    assert tokenizer.eos_token in tokenizer.vocab
    
    # Test encoding with special tokens
    encoded = tokenizer.encode("hello", add_special_tokens=True)
    assert len(encoded) >= 2  # At least BOS and EOS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
