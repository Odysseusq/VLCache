# VLCache

[![arXiv](https://img.shields.io/badge/arXiv-2512.12977-b31b1b.svg)](https://arxiv.org/abs/2512.12977)

**Official repository for "VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference".**

This repository provides a **nanovllm-based implementation** of VLCache, demonstrating efficient KV cache reuse and acceleration for Vision-Language Models. This implementation is specifically optimized for **Qwen2.5-VL**.

## 🚀 Introduction

Processing high-resolution images in Vision-Language Models (VLMs) incurs significant computational costs. While text prompts often benefit from prefix caching, image inputs—despite being identical—cannot be easily cached and reused because their positions and contexts change across requests.

**VLCache** addresses this by enabling **position-agnostic KV cache reuse** for multimodal inputs. By computing only a fraction of vision tokens and reusing the rest, it achieves significant speedups in Time To First Token (TTFT) while maintaining model accuracy.

## 🌟 Key Features

- **Built for Qwen2.5-VL**: Tailored optimizations for the Qwen2.5-VL architecture.
- **Position-Agnostic Reuse**: Overcomes the limitation where varying image positions and contexts prevent direct KV cache reuse.
- **Dynamic Recomputation**: Combines cache reuse with a strategic recomputation policy to eliminate cumulative reuse errors.
- **Theoretical Foundation**: Formally identifies "Reuse Error" propagation and determines the optimal layers for recomputation, outperforming heuristic methods like CacheBlend and EPIC.
- **High Performance**:
  - **1.2x - 16x** speedup in TTFT.
  - Computes only **2% - 5%** of vision tokens.
  - Integrates **Encoder Cache**, **Attention Skip**, and **MLP Skip** optimizations.
- **Lossless Accuracy**: Achieves accuracy on par with full recomputation.

## 📄 Abstract

> This paper presents VLCache, a cache reuse framework that exploits both Key-Value (KV) cache and encoder cache from prior multimodal inputs to eliminate costly recomputation when the same multimodal inputs recur. Unlike previous heuristic approaches, we formally identify the cumulative reuse error effect and demonstrate how to minimize the non-prefix cache reuse error effectively. We further analyze the varying importance of model layers and propose a dynamic, layer-aware recomputation strategy to balance accuracy and efficiency. Experimental results show that VLCache achieves an accuracy on par with full recomputation, while requiring only 2-5% of the tokens to compute, yielding 1.2x-16x TTFT speedups. We develop an experimental implementation of the proposed VLCache pipeline based on SGLang, enabling significantly faster inference in practical deployments.

## 🛠️ Implementation

This codebase implements the VLCache mechanism on top of **nanovllm**, a lightweight LLM inference engine. It demonstrates:
- KV Cache reuse logic for vision tokens.
- Dynamic recomputation strategies.
- Integration with Qwen2.5-VL models.

## 📦 Installation

```bash
git clone https://github.com/VLCache/VLCache.git
cd VLCache
pip install -e .
```

## ⚡ Quick Start

See `example.py` for usage.

## 📊 Benchmark

See `bench.py` for benchmark scripts.

```bash
python bench.py
```

## 🔗 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{qin2025vlcache,
  title={VLCache: Computing 2\% Vision Tokens and Reusing 98\% for Vision-Language Inference},
  author={Qin, Shengling and Yu, Hao and Wu, Chenxin and Li, Zheng and Cao, Yizhong and Zhuge, Zhengyang and Zhou, Yuxin and Yao, Wentao and Zhang, Yi and Wang, Zhengheng and Bai, Shuai and Zhang, Jianwei and Lin, Junyang},
  journal={arXiv preprint arXiv:2512.12977},
  year={2025}
}
```
