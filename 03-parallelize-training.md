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
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2303
#SBATCH --reservation=training-20230301

mkdir -p outputs
source sc_venv_template/activate.sh

srun python -u main.py  --log "logs/" 
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

## DEMO

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
3. Initialize a sampler to specify the sequence of indices/keys used in data loading.
4. Implements data parallelism of the model. 
5. Allow only process to save checkpoints.

---

## Terminologies

- WORLD_SIZE: number of processes participating in the job.
- RANK: the rank of the process in the network.
- LOCAL_RANK: the rank of the process on the local machine.
- MASTER_PORT: free port on machine with rank 0.
- MASTER_ADDR: address of rank 0 node.

---

## DDP code

```python
utils.init_distributed_mode(args.master_port)
torch.distributed.init_process_group(backend='nccl')
device = torch.device(args.device)
torch.backends.cudnn.benchmark = True

image_datasets = load_h5data(args)
datasets_sampler = {x: torch.utils.data.distributed.DistributedSampler(image_datasets[x])
                for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, sampler=datasets_sampler[x], num_workers=args.workers, pin_memory=True)
                for x in ['train', 'val']}
model = resnet50()
model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
model_without_ddp = model.module

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
for epoch in range(args.epochs):
    datasets_sampler["train"].set_epoch(epoch)
    train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], datasets_sampler["train"], device, epoch)
    evaluate(model, criterion, dataloaders["val"], device=device)       
    if utils.is_main_process():
        if args.log:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.log, 'checkpoint.pth'))
```

---

## DEMO

--- 

## DAY 2 RECAP 

- Difference between reading from folders and reading from H5 file.
- Use TensorBoard on the supercomputer.
- Write parallel code.
- Can submit multi-node multi-gpu training.

---

## ANY QUESTIONS??

#### Feedback is more than welcome!

---