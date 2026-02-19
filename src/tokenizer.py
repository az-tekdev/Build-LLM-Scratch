"""
Byte Pair Encoding (BPE) Tokenizer implementation from scratch.
"""
import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """
    A simple BPE tokenizer implementation.
    
    This tokenizer learns subword units by iteratively merging the most frequent
    pairs of consecutive bytes/characters.
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.word_freqs: Dict[str, int] = {}
        self.splits: Dict[str, List[str]] = {}
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Special tokens
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
    def _get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """Count word frequencies in the corpus."""
        word_freqs = defaultdict(int)
        for text in corpus:
            # Simple whitespace tokenization
            words = text.split()
            for word in words:
                word_freqs[word] += 1
        return dict(word_freqs)
    
    def _get_splits(self, word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
        """Split words into characters with end-of-word marker."""
        splits = {}
        for word, freq in word_freqs.items():
            # Split into characters and add end-of-word marker
            splits[word] = list(word) + ["</w>"]
        return splits
    
    def _get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word, word_split in splits.items():
            for i in range(len(word_split) - 1):
                pair = (word_split[i], word_split[i + 1])
                pairs[pair] += self.word_freqs.get(word, 1)
        return dict(pairs)
    
    def _merge_pair(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair in all words."""
        bigram = "".join(pair)
        new_splits = {}
        for word in splits:
            new_split = []
            i = 0
            word_split = splits[word]
            while i < len(word_split):
                if (i < len(word_split) - 1 and 
                    word_split[i] == pair[0] and 
                    word_split[i + 1] == pair[1]):
                    new_split.append(bigram)
                    i += 2
                else:
                    new_split.append(word_split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits
    
    def train(self, corpus: List[str], verbose: bool = False):
        """
        Train the BPE tokenizer on a corpus.
        
        Args:
            corpus: List of text strings to train on
            verbose: Whether to print progress
        """
        if verbose:
            print("Training BPE tokenizer...")
        
        # Get word frequencies
        self.word_freqs = self._get_word_freqs(corpus)
        if verbose:
            print(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize splits (characters)
        self.splits = self._get_splits(self.word_freqs)
        
        # Build initial vocabulary from characters
        vocab = set()
        for word_split in self.splits.values():
            vocab.update(word_split)
        
        # Add special tokens
        special_tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            vocab.add(token)
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(vocab)
        if num_merges <= 0:
            if verbose:
                print("Vocab size too small, using character-level vocabulary")
            vocab = list(vocab)[:self.vocab_size]
        else:
            if verbose:
                print(f"Performing {num_merges} merges...")
            
            for i in range(num_merges):
                pairs = self._get_stats(self.splits)
                if not pairs:
                    break
                
                # Get most frequent pair
                best_pair = max(pairs, key=pairs.get)
                self.merges.append(best_pair)
                
                # Merge the pair
                self.splits = self._merge_pair(best_pair, self.splits)
                
                # Add merged token to vocab
                merged_token = "".join(best_pair)
                vocab.add(merged_token)
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"Merge {i + 1}/{num_merges}: {best_pair}")
        
        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        if verbose:
            print(f"Vocabulary size: {len(self.vocab)}")
            print(f"Number of merges: {len(self.merges)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE splits to a word."""
        if word in self.splits:
            return self.splits[word]
        
        # If word not seen during training, split into characters
        word_split = list(word) + ["</w>"]
        
        # Apply learned merges
        for pair in self.merges:
            new_split = []
            i = 0
            bigram = "".join(pair)
            while i < len(word_split):
                if (i < len(word_split) - 1 and 
                    word_split[i] == pair[0] and 
                    word_split[i + 1] == pair[1]):
                    new_split.append(bigram)
                    i += 2
                else:
                    new_split.append(word_split[i])
                    i += 1
            word_split = new_split
        
        return word_split
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Simple whitespace tokenization
        words = text.split()
        
        token_ids = []
        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, self.vocab.get(self.unk_token))
            token_ids.append(bos_id)
        
        for word in words:
            subwords = self._apply_bpe(word)
            for subword in subwords:
                token_id = self.vocab.get(subword, self.vocab.get(self.unk_token, 0))
                token_ids.append(token_id)
        
        if add_special_tokens:
            eos_id = self.vocab.get(self.eos_token, self.vocab.get(self.unk_token))
            token_ids.append(eos_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            if skip_special_tokens:
                if token in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                    continue
            tokens.append(token)
        
        # Join and clean up
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, filepath: str):
        """Save tokenizer to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "word_freqs": self.word_freqs,
            "special_tokens": {
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load tokenizer from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.word_freqs = data.get("word_freqs", {})
        
        special_tokens = data.get("special_tokens", {})
        self.unk_token = special_tokens.get("unk_token", "<unk>")
        self.pad_token = special_tokens.get("pad_token", "<pad>")
        self.bos_token = special_tokens.get("bos_token", "<bos>")
        self.eos_token = special_tokens.get("eos_token", "<eos>")
        
        # Reconstruct splits from merges (approximate)
        self.splits = {}
        for word in self.word_freqs:
            self.splits[word] = self._apply_bpe(word)
