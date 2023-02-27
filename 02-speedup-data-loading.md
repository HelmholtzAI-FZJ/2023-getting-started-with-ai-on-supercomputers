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
| 10:30 - 11:00 | Coffee Break (flexible) |
| 10:30 - 13:00 | Parallelize Training |

---

## I/O is separate and shared

#### All compute nodes of all supercomputers see the same files

- Performance tradeoff between shared acessibility and speed
- It's simple to load data fast to 1 or 2 gpus. But to 100? 1000? 10000?

---

### Jülich Supercomputers

- Our I/O server is almost a supercomputer by itself
- ![JSC Supercomputer Stragegy](images/machines.png)

---

## Data services

- JSC provides different data services
- Data projects give massive amounts of storage
- We use it for ML datasets. Join the project at [Judoor](https://judoor.fz-juelich.de/projects/join/datasets)
- After being approved, connect to the supercomputer and try it:
- ```bash
cd $DATA_datasets
```

---

## Data Staging

- [LARGEDATA filesystem](https://apps.fz-juelich.de/jsc/hps/juwels/filesystems.html) is not accessible by compute nodes
- Copy files to an accessible filesystem BEFORE working
- Imagenet-21K copy alone takes 21+ minutes to $SCRATCH

---

## Data loading

![Fat GPUs need to be fed FAST](images/nomnom.jpg)

--- 

## Strategies

- We have CPUs and lots of memory - let's use them
    - multitask training and data loading for the next batch
    - `/dev/shm` is a filesystem on ram - ultra fast ⚡️
- Use big files made for parallel computing
    - HDF5, Zarr, mmap() in a parallel fs, LMDB
- Use specialized data loading libraries
    - FFCV, DALI
- Compression
    - data transfer can be slower than decompression (must be checked case by case)

---

## Libraries

- FFCV [https://github.com/libffcv/ffcv](https://github.com/libffcv/ffcv) and [FFCV for PyTorch-Lightning](https://github.com/SerezD/ffcv_pytorch_lightning)
- Nvidia's DALI [https://developer.nvidia.com/dali](https://developer.nvidia.com/dali)

---

## Access file VS Access H5 file

```python
images_data = data_loader.ImageNetKaggle(images_root, "train", transforms["train"]) 
dataloaders = DataLoader(images_data, batch_size=batch_size, num_workers=workers)

start_t = time.time()
for x in dataloaders:
    x

end_t = time.time()
print("Time without h5 file: ", str(datetime.timedelta(seconds=int(end_t-start_t))))       
```
```bash 
Time without h5 file:  0:00:29
```     

```python
images_data = data_loader.ImagenetH5(h5_file, "train", transforms["train"]) 
dataloadersh5= DataLoader(images_data, batch_size=batch_size, num_workers=workers)
 
start_t = time.time()
for x in dataloadersh5:
    x
end_t = time.time()

print("Time with h5 file: ", str(datetime.timedelta(seconds=int(end_t-start_t))))

```
```bash 
Time with h5 file:  0:00:26
```
---
## Demo
