"""
Data loading and preprocessing utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from tqdm import tqdm


class TextDataset(Dataset):
    """
    Dataset for text data with tokenization.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance with encode method
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.tokenized = []
        for text in tqdm(texts, desc="Tokenizing dataset"):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.tokenized.append(tokens)
    
    def __len__(self) -> int:
        return len(self.tokenized)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = self.tokenized[idx]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor], pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching sequences of different lengths.
    
    Args:
        batch: List of token sequences
        pad_idx: Padding token index
        
    Returns:
        Tuple of (padded_sequences, attention_mask)
    """
    # Find max length in batch
    max_len = max(len(seq) for seq in batch)
    
    # Pad sequences
    padded_batch = []
    masks = []
    
    for seq in batch:
        pad_length = max_len - len(seq)
        padded_seq = torch.cat([seq, torch.full((pad_length,), pad_idx, dtype=torch.long)])
        padded_batch.append(padded_seq)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.ones(len(seq), dtype=torch.long)
        mask = torch.cat([mask, torch.zeros(pad_length, dtype=torch.long)])
        masks.append(mask)
    
    return torch.stack(padded_batch), torch.stack(masks)


def create_data_loader(
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    pad_idx: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        pad_idx: Padding token index
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = TextDataset(texts, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_tiny_shakespeare() -> List[str]:
    """
    Load Tiny Shakespeare dataset.
    Returns list of text chunks.
    """
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("tiny_shakespeare", split="train")
        texts = [item["text"] for item in dataset]
        
        # Split into smaller chunks if needed
        chunk_size = 500  # characters per chunk
        chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if len(chunk) > 50:  # Only keep substantial chunks
                    chunks.append(chunk)
        
        return chunks[:1000]  # Limit for demo purposes
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback sample text...")
        # Fallback sample text
        return [
            "To be or not to be, that is the question.",
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
        ] * 100


def load_wikitext_small() -> List[str]:
    """
    Load WikiText-2 dataset (small subset).
    Returns list of text chunks.
    """
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]
        
        return texts[:2000]  # Limit for demo purposes
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return load_tiny_shakespeare()
