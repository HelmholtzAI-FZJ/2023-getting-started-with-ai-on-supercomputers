---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Parallelize Training
date: March 01, 2023
---
## One GPU training 

```python
device = torch.device(args.device)

images_data = data_loader.ImagenetH5(h5_file, "train", transforms["train"]) 
dataloadersh5= DataLoader(images_data, batch_size=batch_size, num_workers=workers)

model = resnet50(True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
weight_decay=args.weight_decay)

print("Start training")

for epoch in range(args.epochs):
    train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], 
    device, epoch)
    evaluate(model, criterion, dataloaders["val"], device=device)


```

---
## TensorBoard 

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

---
## DDP code

---
## DEMO