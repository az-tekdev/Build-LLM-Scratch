# Build LLM from Scratch

A complete, production-ready implementation of a GPT-style Large Language Model (LLM) built entirely from scratch using PyTorch. This project implements a full transformer architecture with tokenization, training pipeline, and inference capabilities.

## 🎯 Features

- **Full Transformer Architecture**: Decoder-only transformer with multi-head self-attention, feed-forward layers, and positional encoding
- **BPE Tokenizer**: Byte Pair Encoding tokenizer implemented from scratch
- **Training Pipeline**: Complete training loop with AdamW optimizer, cosine annealing scheduler, gradient clipping, and checkpointing
- **Inference**: Multiple sampling strategies (greedy, top-k, top-p) with temperature control
- **Modular Design**: Clean, well-organized codebase with type hints and docstrings
- **Production Ready**: Error handling, configuration management, and comprehensive testing

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Hugging Face Datasets (for sample data)
- tqdm (for progress bars)
- PyYAML (for configuration)
- pytest (for testing)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/build-llm-from-scratch.git
cd build-llm-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Train the Tokenizer

First, train a BPE tokenizer on your dataset:

```bash
python tokenizer_train.py --dataset tiny_shakespeare --vocab-size 10000 --output ./tokenizer.json
```

Options:
- `--dataset`: Choose between `tiny_shakespeare` or `wikitext`
- `--vocab-size`: Vocabulary size (default: 10000)
- `--output`: Output path for the tokenizer (default: `./tokenizer.json`)
- `--verbose`: Print progress information

### 2. Train the Model

Train the transformer language model:

```bash
python train.py --tokenizer ./tokenizer.json --dataset tiny_shakespeare --num-epochs 10
```

Options:
- `--config`: Path to YAML config file (optional)
- `--tokenizer`: Path to trained tokenizer
- `--dataset`: Dataset to use (`tiny_shakespeare` or `wikitext`)
- `--batch-size`: Batch size (overrides config)
- `--learning-rate`: Learning rate (overrides config)
- `--num-epochs`: Number of training epochs
- `--checkpoint-dir`: Directory to save checkpoints
- `--device`: Device to use (`auto`, `cuda`, or `cpu`)
- `--resume`: Resume training from checkpoint

### 3. Generate Text

Generate text from a trained model:

```bash
python inference.py --checkpoint ./checkpoints/best_model.pt --tokenizer ./tokenizer.json --prompt "The quick brown fox"
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--tokenizer`: Path to tokenizer
- `--prompt`: Input prompt text
- `--max-length`: Maximum generation length
- `--method`: Sampling method (`greedy`, `top_k`, or `top_p`)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top-k`: Top-k parameter for top_k sampling
- `--top-p`: Top-p parameter for top_p sampling
- `--interactive`: Enable interactive mode

### Interactive Mode

Run inference in interactive mode for a chat-like experience:

```bash
python inference.py --checkpoint ./checkpoints/best_model.pt --tokenizer ./tokenizer.json --interactive
```

## 🏗️ Architecture

### Model Architecture

The model implements a decoder-only transformer with the following components:

1. **Token Embeddings**: Learnable token embeddings
2. **Positional Encoding**: Sinusoidal positional encodings
3. **Transformer Blocks**: Stack of decoder blocks, each containing:
   - Multi-head self-attention with causal masking
   - Feed-forward network (GELU activation)
   - Layer normalization and residual connections
4. **Output Head**: Linear projection to vocabulary size

### Key Components

- **MultiHeadAttention**: Scaled dot-product attention with multiple heads
- **FeedForward**: Position-wise feed-forward network
- **TransformerBlock**: Complete transformer block with attention and FFN
- **PositionalEncoding**: Sinusoidal positional encodings
- **TransformerLM**: Full language model combining all components

### Tokenizer

The BPE tokenizer learns subword units by iteratively merging the most frequent pairs of consecutive characters/bytes. It supports:
- Training on custom corpora
- Encoding/decoding text
- Special tokens (BOS, EOS, PAD, UNK)
- Saving/loading trained tokenizers

## ⚙️ Configuration

The project supports configuration via YAML files. See `config.yaml` for an example:

```yaml
model:
  vocab_size: 10000
  d_model: 512
  num_layers: 6
  num_heads: 8
  d_ff: 2048
  max_seq_len: 1024
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 3e-4
  weight_decay: 0.1
  num_epochs: 10
  max_grad_norm: 1.0
  warmup_steps: 100
```

## 📊 Hyperparameters

Default hyperparameters for a small model:

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 10,000 |
| Model Dimension | 512 |
| Number of Layers | 6 |
| Number of Heads | 8 |
| Feed-forward Dimension | 2,048 |
| Max Sequence Length | 1,024 |
| Dropout | 0.1 |
| Batch Size | 32 |
| Learning Rate | 3e-4 |
| Weight Decay | 0.1 |

For larger models, increase `d_model`, `num_layers`, and `d_ff` accordingly.

## 📧 Contact

- Telegram: https://t.me/az_tekDev
- Twitter: https://x.com/az_tekDev

