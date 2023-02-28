---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Parallelize Training
date: March 01, 2023
---
## One GPU training 

```python
device = torch.device(args.device)

images_data = data_loader.ImagenetH5(args.h5_file, "train", transforms["train"]) 
dataloadersh5= DataLoader(images_data, batch_size=args.batch_size, 
num_workers=args.workers)

model = resnet50(True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
weight_decay=args.weight_decay)

for epoch in range(args.epochs):
    train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], 
    device, epoch)
    evaluate(model, criterion, dataloaders["val"], device=device)

checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
    'epoch': epoch,'args': args}
torch.save(checkpoint, os.path.join(args.log, 'checkpoint.pth'))
```

---

## One GPU training 

``` bash 
#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2303
#SBATCH --reservation=training-20230301

mkdir -p output
source sc_venv_template/activate.sh

srun python -u main.py --data_dir --log "logs/" 
```

---

## TensorBoard

```python 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(tb_dir)
writer.add_scalar('loss/train', train_loss, epoch)
```

```bash
ssh  -L 16000:localhost:16000 booster
```

```bash
ml TensorFlow
tensorboard --logdir=[PATH_TO_TENSOR_BOARD] --port=16000
```

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

## DDP steps

1. Set up the environement variable for the distributed mode (WORLD_SIZE, RANK, LOCAL_RANK ...)
2. Initialize the torch.distributed package
3. Enable cuDNN to select the fastest convolution algorithms. 
4. Initialize a sampler to specify the sequence of indices/keys used in data loading.
5. Synchronization of batchnorm statistics
6. Implements data parallelism of the model. 
7. Allow only process to save checkpoints.

---

## Terminologies

---
## DDP code

---
## DEMO

---