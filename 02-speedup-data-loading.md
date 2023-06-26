---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Speedup Data loading
date: June 28, 2023
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
- Compression sush as squashfs 
    - data transfer can be slower than decompression (must be checked case by case)
    - Beneficial in cases where numerous small files are at hand.

---

## Libraries

- FFCV [https://github.com/libffcv/ffcv](https://github.com/libffcv/ffcv) and [FFCV for PyTorch-Lightning](https://github.com/SerezD/ffcv_pytorch_lightning)
- Nvidia's DALI [https://developer.nvidia.com/dali](https://developer.nvidia.com/dali)

---

## H5 file
![](images/hdf5.svg)

---

## Access ImageNet images

```python
def __getitem__(self, idx):
    x = Image.open(self.samples[idx]).convert("RGB")
    if self.transform:
        x = self.transform(x)
    return x, self.targets[idx]
```

---

## Access ImageNet images

```python
image_datasets = ImageNet(args.data_root, "train",  transformation()) 
dataloaders = DataLoader(image_datasets, batch_size=args.batch_size, \
    num_workers=workers,  pin_memory=True)

print("Start loading ImageNet images")
for data in dataloaders:
    pass
```

```bash 
elapsed: 00 hours 07 min 29 sec
```     

---

## Use H5 file

```python
def __getitem__(self, index: int):
    img_string = self.imgs["images"][index]

    with BytesIO(img_string) as byte_stream:
        img = Image.open(byte_stream)
        img = img.convert("RGB")

    if self.transform:
        img = self.transform(img)

    return img, self.targets[index]
```

---

## Use H5 file

```python
image_datasets = ImageNetH5(args.data_root, "train", transformation()) 
dataloadersh5= DataLoader(image_datasets, batch_size=args.batch_size, \
    num_workers=workers, pin_memory=True)

print("Start loading with H5 file")
for x in dataloadersh5:
    pass
```

```bash 
elapsed: 00 hours 03 min 48 sec
```    

---

## Demo
