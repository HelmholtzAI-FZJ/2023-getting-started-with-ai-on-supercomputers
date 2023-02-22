---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Parallelize Training
date: March 01, 2023
---

## Parallel ML

![](images/paralellism-types.jpg)
Shamelessly stolen from [twitter](https://twitter.com/rasbt/status/1625494398778892292)

---

## Parallel Training

- [PyTorch's DDP (Distributed Data Parallel)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Multi-node, multi-GPU
- Multiple processes: One process per model replica
- Each process uses multiple GPUs: model parallel)
- All processes collectively use data parallel

---

## DEMO