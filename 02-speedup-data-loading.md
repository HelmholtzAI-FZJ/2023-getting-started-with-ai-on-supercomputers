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

## Strategies

- We have CPUs and lots of memory - let's use them
- Use big files made for parallel computing - HDF5, Zarr, mmap()
- Use specialized data loading libraries
- Compression - data throughput can be slower than time to decompression (must be checked case by case)

---

## Libraries

- FFCV [https://github.com/libffcv/ffcv](https://github.com/libffcv/ffcv)
- Nvidia's DALI [https://developer.nvidia.com/dali](https://developer.nvidia.com/dali)


---