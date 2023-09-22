---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Speedup Data loading
date: September 27, 2023
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

## The ImageNet dataset
#### Large Scale Visual Recognition Challenge (ILSVRC)
- An image dataset organized according to the WordNet hierarchy. 
- Evaluates algorithms for object detection and image classification at large scale. 
- It has 1000 classes, that comprises 1.2 million images for training, and 5,000 images for the validation set.

![](images/imagenet_banner.jpeg)
---

## Some imports 

```python
import os 
from io import BytesIO

import click
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
```

---

## ImageNet class

```python
class ImageNet(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        with open(os.path.join(self.root, "train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)
        self.samples = list(train_data.keys())
        self.targets = list(train_data.values())
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.root, self.samples[idx])).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
```

---

## Access ImageNet images

```python
@click.command()
@click.option("--data_root", "-r")
def main(data_root):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    image_datasets = ImageNet(data_root, transform) 
    dataloaders = DataLoader(image_datasets, batch_size=1024, \
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')))

    for _ in dataloaders:
        pass
```

```python
if __name__ == "__main__":
    main()
```

```bash
real	9m54.053s
```     

---

## ImageNet class

```python
def main():
    root = "/p/scratch/training2324/data/"
    splits = ["train", "val"]

    with open(os.path.join(root, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)

    train_samples = list(train_data.keys())
    train_targets = list(train_data.values())

    with open(os.path.join(root, "val_data.pkl"), "rb") as f:
        val_data = pickle.load(f)

    val_samples = list(val_data.keys())
    val_targets = list(val_data.values())
```

```python
train_samples = ['ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_8050.JPEG',
 'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_12728.JPEG',
 'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_9736.JPEG',
 'ILSVRC/Data/CLS-LOC/train/n03146219/n03146219_22069.JPEG',
 ...]

train_targets = [524,
 524,
 524,
 524,
 ...]
```
---

## Demo

- Please clone the repository provided by the link bellow

- `git clone git@github.com:HelmholtzAI-FZJ/2023-getting-started-with-ai-on-supercomputers.git`
