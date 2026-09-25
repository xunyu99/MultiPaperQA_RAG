# Efficient Transformers for Long Documents

Wei Zhang, Ming Li, and Hao Chen

## Abstract

Self-attention scales quadratically with sequence length, which limits its use on long documents.
We propose a sparse attention mechanism that keeps the dominant contribution of the full attention
matrix at linear cost.

## 1 Introduction

Transformer models have become the default architecture for natural language processing.

## 2 Related Work

Sparse attention has been explored through fixed patterns, learnable patterns, and low-rank
approximations.

## 3 Method

### 3.1 Sparse Attention

Each query attends to a local window of size w and to a set of global tokens.

### 3.2 Complexity Analysis

With window size w and n global tokens, the total cost becomes linear in the sequence length.

## 4 Experiments

We evaluate on three long-document benchmarks.

## 5 Conclusion

We presented a sparse attention mechanism that makes document-level transformers practical.
