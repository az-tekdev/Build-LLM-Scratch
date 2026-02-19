"""
Configuration management.
"""
import yaml
from typing import Dict, Any
import os


class Config:
    """Configuration class for model and training parameters."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load config from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def save_yaml(self, filepath: str):
        """Save config to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "vocab_size": 10000,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    "tokenizer": {
        "vocab_size": 10000,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "num_epochs": 10,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "save_every": 1,
    },
    "data": {
        "dataset": "tiny_shakespeare",
        "max_length": 512,
        "train_split": 0.9,
    },
    "inference": {
        "max_length": 100,
        "method": "top_p",
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
    },
    "paths": {
        "checkpoint_dir": "./checkpoints",
        "tokenizer_path": "./tokenizer.json",
        "data_dir": "./data",
    },
}


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    else:
        return Config.from_dict(DEFAULT_CONFIG)
