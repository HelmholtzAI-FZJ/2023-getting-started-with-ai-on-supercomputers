---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Parallelize Training
date: June 28, 2023
---

## One GPU training 

```python
# set the random seeds.
seed_everything(42)

# 1. Download and organize the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
    batch_size=1,
)

# 2. Build the model using desired Task
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, pretrained=False)

# 3. Create the trainer 
trainer = flash.Trainer(max_epochs=50,  accelerator="gpu", devices=1)

# 4. Train the model
trainer.fit(model, datamodule=datamodule)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
```

---

## One GPU training 

``` bash 
#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# activate env
source ../sc_venv_template/activate.sh

# run script 
srun python3 one_gpu.py
```

```bash
elapsed: 00 hours 06 min 04 sec
```

---

## Parallel ML

![](images/paralellism-types.jpg)
Shamelessly stolen from [twitter](https://twitter.com/rasbt/status/1625494398778892292)

---

## Parallel Training

- [PyTorch's DDP (Distributed Data Parallel)](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html) works as follows:
    - Each GPU across each node gets its own process.
    - Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.
    - Each process inits the model.
    - Each process performs a full forward and backward pass in parallel.
    - The gradients are synced and averaged across all processes.
    - Each process updates its optimizer.

---

## Terminologies

- WORLD_SIZE: number of processes participating in the job.
- RANK: the rank of the process in the network.
- LOCAL_RANK: the rank of the process on the local machine.
- MASTER_PORT: free port on machine with rank 0.
- MASTER_ADDR: address of rank 0 node.

---

## DDP steps

1. Set up the environement variable for the distributed mode (WORLD_SIZE, RANK, LOCAL_RANK ...)

```python
# The number of total processes started by Slurm.
ntasks = os.getenv('SLURM_NTASKS')

# Index of the current process.
rank = os.getenv('SLURM_PROCID')

# Index of the current process on this node only.
local_rank = os.getenv('SLURM_LOCALID')

# The number of nodes
nnodes = os.getenv("SLURM_NNODES")
```

---

## DDP steps

2. Initialize the torch.distributed package

```python
utils.init_distributed_mode(12354)
```

---

## DDP steps

3. Initialize a sampler to specify the sequence of indices/keys used in data loading.
4. Implements data parallelism of the model. 
5. Allow only process to save checkpoints.


```python
# 3. Create the trainer 
trainer = flash.Trainer(max_epochs=50,  accelerator="gpu", devices=ntasks,\
    num_nodes=nnodes)
```

---

## DDP training

```bash
#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err

CUDA_VISIBLE_DEVICES=0,1,2,3
# activate env
source ../sc_venv_template/activate.sh

# run script 
srun python3 ddp.py
```

```bash
elapsed: 00 hours 01 min 16 sec
```

---

## DEMO

--- 

## TensorBoard

```python 
# 3. Create the logger 
logger = TensorBoardLogger("tb_logs", name="my_model")

# 4. Create the trainer and pass the logger 
trainer = flash.Trainer(max_epochs=50,  accelerator="gpu", devices=ntasks, \
    num_nodes=nnodes, logger=logger)
```

--- 

## TensorBoard

```bash
ssh  -L 16000:localhost:16000 booster
```

```bash
ml load Stages/2023 
ml load GCC 
ml TensorFlow
tensorboard --logdir=[PATH_TO_TENSOR_BOARD] --port=16000
```
![](images/tb.png)

---

## DEMO

---

## DAY 2 RECAP 

- Difference between reading from folders and reading from H5 file.
- Write parallel code.
- Can submit multi-node multi-gpu training.
- Use TensorBoard on the supercomputer.

---

## ANY QUESTIONS??

#### Feedback is more than welcome!

---