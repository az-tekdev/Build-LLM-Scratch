"""
Inference utilities for text generation.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


def greedy_decode(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    device: str = "auto",
) -> str:
    """
    Generate text using greedy decoding.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_length: Maximum generation length
        device: Device to use
        
    Returns:
        Generated text
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS token
            eos_token_id = tokenizer.vocab.get(tokenizer.eos_token, -1)
            if next_token_id.item() == eos_token_id:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token_id], dim=1)
    
    # Decode
    generated_ids = generated[0].cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text


def top_k_sampling(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    k: int = 50,
    temperature: float = 1.0,
    device: str = "auto",
) -> str:
    """
    Generate text using top-k sampling.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_length: Maximum generation length
        k: Number of top tokens to sample from
        temperature: Sampling temperature
        device: Device to use
        
    Returns:
        Generated text
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from top-k
            next_token_idx = torch.multinomial(probs, 1)
            next_token_id = top_k_indices[next_token_idx].unsqueeze(0).unsqueeze(0)
            
            # Check for EOS token
            eos_token_id = tokenizer.vocab.get(tokenizer.eos_token, -1)
            if next_token_id.item() == eos_token_id:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token_id], dim=1)
    
    # Decode
    generated_ids = generated[0].cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text


def top_p_sampling(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    p: float = 0.9,
    temperature: float = 1.0,
    device: str = "auto",
) -> str:
    """
    Generate text using top-p (nucleus) sampling.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_length: Maximum generation length
        p: Nucleus sampling parameter (0.0 to 1.0)
        temperature: Sampling temperature
        device: Device to use
        
    Returns:
        Generated text
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create mask
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            # Renormalize
            probs = probs / probs.sum()
            
            # Sample
            next_token_id = torch.multinomial(probs, 1).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS token
            eos_token_id = tokenizer.vocab.get(tokenizer.eos_token, -1)
            if next_token_id.item() == eos_token_id:
                break
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token_id], dim=1)
    
    # Decode
    generated_ids = generated[0].cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return text


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    method: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "auto",
) -> str:
    """
    Generate text with specified sampling method.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_length: Maximum generation length
        method: Sampling method ('greedy', 'top_k', or 'top_p')
        temperature: Sampling temperature
        top_k: Top-k parameter (for top_k method)
        top_p: Top-p parameter (for top_p method)
        device: Device to use
        
    Returns:
        Generated text
    """
    if method == "greedy":
        return greedy_decode(model, tokenizer, prompt, max_length, device)
    elif method == "top_k":
        return top_k_sampling(model, tokenizer, prompt, max_length, top_k, temperature, device)
    elif method == "top_p":
        return top_p_sampling(model, tokenizer, prompt, max_length, top_p, temperature, device)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
