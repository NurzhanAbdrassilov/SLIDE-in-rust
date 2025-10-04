# SLIDE-in-rust

A Rust implementation of SLIDE (Sub-Linear Deep Learning Engine) for efficient training of large sparse neural networks.

## Overview

SLIDE is a novel training algorithm that makes neural network training sub-linear in the number of network parameters by using adaptive sampling techniques. This implementation focuses on multilabel text classification tasks.

## Features

- **Sparse Neural Networks**: Efficient handling of high-dimensional sparse inputs
- **Adaptive Sampling**: LSH-based node sampling for sub-linear training complexity  
- **Distributed Weights**: PSL array-based distributed weight storage
- **Multi-worker Support**: Configurable parallel training workers

## Dataset

Currently configured for the **Eurlex-4.3K** multilabel text classification dataset:
- Input dimension: 200,000 features
- Output classes: 4,271 labels
- Training samples: ~45,000
- Architecture: Input → ReLU (64 nodes) → Softmax (4,271 nodes)

## Usage

### Build and Run

```bash
# Build in release mode for optimal performance
cargo build --release

# Run training
cargo run --release
```

### Configuration

Key training parameters can be modified in `main.rs`:
- Learning rate: `cfg.Lr`
- Batch size: `cfg.Batchsize`  
- Number of epochs: `cfg.Epoch`
- Network architecture: `cfg.sizesOfLayers`

## Performance

The implementation uses several optimizations:
- Release mode compilation for performance
- Minimal debug output during training
- Efficient sparse matrix operations
- ADAM optimizer with gradient accumulation

## Project Structure

- `src/main.rs` - Main training loop and configuration
- `src/network.rs` - Neural network implementation
- `src/node.rs` - Individual neuron/node implementation
- `src/layer.rs` - Layer management
- `src/psl_array.rs` - Distributed weight storage
- `dataset/` - Training and test data files

## Notes

This is a research implementation focused on exploring SLIDE algorithms for sparse neural network training. The codebase prioritizes clarity and correctness over maximum optimization.