"""
Training utilities and trainer class.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict
from tqdm import tqdm
import json


class Trainer:
    """
    Trainer class for language model training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "auto",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        save_dir: str = "./checkpoints",
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('auto', 'cuda', or 'cpu')
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            max_grad_norm: Gradient clipping norm
            warmup_steps: Number of warmup steps
            save_dir: Directory to save checkpoints
        """
        self.model = model
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * 10  # Assume 10 epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1,
        )
        
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
        }
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss, ignoring padding tokens.
        
        Args:
            logits: Model output of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
            mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Scalar loss value
        """
        # Shift targets for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_mask = shift_mask.view(-1)
        
        # Compute loss only on non-padding tokens
        loss = nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='none',
        )
        loss = (loss * shift_mask).sum() / shift_mask.sum()
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (input_ids, mask) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, mask)
            loss = self.compute_loss(logits, input_ids, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Learning rate scheduling (with warmup)
            if batch_idx < self.warmup_steps:
                lr_scale = min(1.0, float(batch_idx + 1) / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * lr_scale
            else:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating")
        for input_ids, mask in pbar:
            input_ids = input_ids.to(self.device)
            mask = mask.to(self.device)
            
            logits = self.model(input_ids, mask)
            loss = self.compute_loss(logits, input_ids, mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "perplexity": perplexity}
    
    def train(self, num_epochs: int, save_every: int = 1):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_ppl"].append(train_metrics["perplexity"])
            
            # Validate
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_ppl"].append(val_metrics["perplexity"])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train PPL: {train_metrics['perplexity']:.2f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val PPL: {val_metrics['perplexity']:.2f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path = os.path.join(self.save_dir, "best_model.pt")
                self.save_checkpoint(best_path, epoch)
                print(f"New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to {history_path}")
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']
