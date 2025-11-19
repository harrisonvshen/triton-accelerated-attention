# Triton Accelerated Attention
This project implements the core computations behind multi-head self-attention using custom Triton GPU kernels. The goal is to reproduce the main stages of attention ($QK^T$, softmax, and value aggregation) without relying on PyTorch’s built-in CUDA kernels, and to understand how these operations behave when written at a lower level.

The codebase is organized so that each part of attention can be examined, benchmarked, and tested in isolation. The final module assembles the kernels into a complete multi-head attention layer.

------------------------------------------------------------

## Project Overview

The repository provides custom Triton implementations of the following attention components:

- The Query–Key dot-product ($QK^T$)
- A row-wise softmax operation
- Weighted value aggregation (probs @ V)
- A self-contained multi-head attention layer
- Benchmarking and correctness tests

All kernels operate on block-tiled data layouts. This makes it easier to experiment with `BLOCK_M`, `BLOCK_N`, and head-dimension settings, and to observe how these choices influence performance.

------------------------------------------------------------

## Attention Computation Flow

    Q --------------\
                       --->  [ QK^T ] ---> [ Softmax ] ---> [ Probs @ V ] ---> Output
    K --------------/

    V ---------------------------------------------------------------/

    **Note:** V does not interact with Q or K; it is applied only at the final stage, where the softmax probabilities weight the value vectors.

------------------------------------------------------------

## Repository Structure

kernels/  
    All Triton kernels ($QK^T$, softmax, value aggregation, and supporting matmul kernels).

benchmarks/  
    Benchmark scripts and plots for block-size sweeps, sequence-length scaling, and
    Triton vs PyTorch comparisons.

tests/  
    Debug tools and correctness tests validating numerical equivalence with PyTorch.

README.md  
    Project documentation.

LICENSE  
    MIT license.

.gitignore  
    Environment and cache exclusions.

------------------------------------------------------------

## Installation

1. Create and activate a virtual environment:

    python3 -m venv venv  
    source venv/bin/activate

2. Install the required packages:

    pip install torch triton matplotlib numpy

------------------------------------------------------------

## How to Use This Repository

Run the numerical correctness test:

    python3 tests/test_attention.py

Generate the block-size heatmap:

    python3 benchmarks/benchmark_heatmap.py

Run the Triton vs PyTorch runtime benchmarks:

    python3 benchmarks/benchmark_attention.py

Measure runtime scaling for different sequence lengths:

    python3 benchmarks/benchmark_seq_lengths.py

------------------------------------------------------------

## Results

The repository includes several performance evaluations that highlight how the Triton kernels behave under different workloads.

### Block-Size Sweep
The `BLOCK_M` × `BLOCK_N` heatmap reveals how latency shifts across 16 different tile configurations. The benchmark illustrates where particular tile shapes align well with memory-access patterns and reduction sizes.

### Sequence Length Scaling
Runtime increases quadratically with sequence length, consistent with the theoretical behavior of standard attention. The Triton kernels maintain stable tile-level efficiency across the tested sequence lengths.

### Comparison with PyTorch CUDA Kernels
On the same RTX 3060 Laptop GPU, the Triton implementation processes a 4096-token, 4-head attention layer in about 0.41 ms, while PyTorch’s optimized CUDA MultiheadAttention takes around 12.6 ms under the same conditions. This reflects a roughly 30× speed difference for this specific benchmark and configuration.

### Reproducing the Benchmark

You can reproduce this measurement with:

    python3 benchmarks/benchmark_attention.py

All plots and raw outputs are available in the `benchmarks/` directory.

------------------------------------------------------------

## Notes on Kernel Design

Key design principles:

- Kernels follow a block-tiling structure controlled by `BLOCK_M` and `BLOCK_N`.
- Reductions over the head dimension must satisfy Triton’s requirement that the reduction size be at least 16.
- Vectorized loads and pointer arithmetic manage memory access explicitly.
- Boundary handling relies on mask-based loads and stores.
- Each Triton program instance computes a tile of the output matrix, mirroring the structure of optimized attention kernels.

The project separates $QK^T$, softmax, and value aggregation for clarity. These steps can be fused for further optimization, similar to approaches used in FlashAttention.

------------------------------------------------------------

## Motivation

The purpose of this project is to examine attention at a low level and to understand how its components interact with GPU execution and memory hierarchy. Triton provides a practical way to explore these kernels while retaining control over tiling and memory behavior.

------------------------------------------------------------

## Possible Extensions

Potential future directions:

- Combining $QK^T$, softmax, and value aggregation into a single fused kernel
- Adding support for FP16 or BF16
- Using Triton’s autotuning tools to analyze larger configuration spaces
- Implementing multi-query or grouped-query attention variants

------------------------------------------------------------

## License


This project is provided under the MIT license.
