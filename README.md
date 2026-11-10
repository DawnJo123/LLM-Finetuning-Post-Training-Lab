# LLM Fine-Tuning & Post-Training Lab

An experimental engineering project exploring the fine-tuning, post-training, evaluation, and optimization of small language models (SLMs).

The goal is to understand and implement the complete lifecycle of adapting pretrained language models, from baseline inference to fine-tuning and optimized deployment.

## Project Objectives

- Understand pretrained language model behaviour.
- Build supervised fine-tuning (SFT) pipelines.
- Experiment with LoRA and QLoRA.
- Evaluate model behaviour before and after fine-tuning.
- Explore quantization and inference optimization.
- Profile GPU performance and memory consumption.
- Build reproducible experiments with documented results.

## Initial Model

**Qwen3-0.6B**

- Model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Parameters: Approximately 596 million
- Framework: PyTorch
- Model loading: Hugging Face Transformers
- Initial precision: FP16

## Hardware and Software

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM | 4 GB |
| Operating System | Windows 11 |

### Software

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0+cu126 |
| Transformers | 5.22.2 |
| Accelerate | 1.15.0 |
| uv | 0.12.9 |
| CUDA runtime (PyTorch) | 12.6 |

Dependencies are managed using `uv`. See `pyproject.toml` and `uv.lock` for the project environment.

## Project Structure

```text
LLM_finetuning/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   └── training/
├── experiments/
├── tests/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Completed Work

### Phase 1 — Environment Setup and Baseline Inference

**Status: Completed**

- Created the Python project environment using `uv`.
- Installed PyTorch with CUDA support.
- Verified CUDA availability and GPU detection.
- Installed Hugging Face Transformers and Accelerate.
- Downloaded and loaded Qwen3-0.6B.
- Successfully executed inference on the NVIDIA RTX 3050 Laptop GPU.
- Explored greedy decoding and probabilistic sampling.
- Examined tokenization, chat templates, and autoregressive generation.
- Observed GPU memory consumption during inference.

### Initial Inference Observations

| Measurement | Observed result |
|---|---:|
| Model parameters | 596,049,920 |
| GPU allocated memory | Approximately 1.13 GB |
| GPU reserved memory | Approximately 1.56 GB |
| Model execution device | CUDA |

These are observations from the initial inference experiment, not training benchmarks.

## Planned Work

1. Baseline evaluation dataset and metrics.
2. Dataset collection and preprocessing.
3. Supervised fine-tuning pipeline.
4. LoRA and QLoRA experiments.
5. Post-training evaluation and comparison.
6. Quantization and inference optimization.
7. GPU profiling using PyTorch Profiler and NVIDIA Nsight tools.

## Experimental Philosophy

- Preserve the original pretrained model as a baseline.
- Keep evaluation data separate from training data.
- Record actual experimental results rather than assumed performance.
- Track configurations and reproducibility.
- Document engineering decisions, limitations, and observations.

## Current Status

**Phase 1 completed — Baseline inference successfully executed.**

The next milestone is to establish a baseline evaluation framework before modifying the model through fine-tuning.

---

*This repository is an evolving engineering portfolio project. Documentation and results will be updated as experiments are completed.*