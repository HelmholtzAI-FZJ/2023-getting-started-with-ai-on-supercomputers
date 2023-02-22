---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Speedup Data loading
date: March 01, 2023
---

### Schedule for day 2

| Time          | Title                |
| ------------- | -----------          |
| 09:00 - 09:15 | Welcome, questions   |
| 09:15 - 10:30 | Speedup data loading |
| 10:30 - 13:00 | Parallelize Training |


---


### Jülich Supercomputers

![JSC Supercomputer Stragegy](images/machines.png)


---


## I/O is separate and shared

#### All compute nodes of all supercomputers see the same files

- Performance tradeoff between shared acessibility and speed

---

## Data loading

![Fat GPUs need to be fed FAST](images/nomnom.jpg)

---

## Data Staging

- [LARGEDATA filesystem](https://apps.fz-juelich.de/jsc/hps/juwels/filesystems.html) is not accessible by compute nodes
- Copy files to an accessible filesystem
- Imagenet-21K copy alone takes 21+ minutes to $SCRATCH


--- 

## Strategies

- We have CPUs and lots of memory - let's use them
    - `/dev/shm` is a filesystem on ram - ultra fast ⚡️
- Use big files made for parallel computing
    - HDF5, Zarr, mmap(), LMDB
- Use specialized data loading libraries
    - FFCV, DALI
- Compression
    - data transfer can be slower than decompression (must be checked case by case)

---

## Libraries

- FFCV [https://github.com/libffcv/ffcv](https://github.com/libffcv/ffcv) and [FFCV for PyTorch-Lightning](https://github.com/SerezD/ffcv_pytorch_lightning)
- Nvidia's DALI [https://developer.nvidia.com/dali](https://developer.nvidia.com/dali)

---

## DEMO
