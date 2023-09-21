---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
subtitle: Parallelize Training
date: September 28, 2023
---

## PyTroch Lightning Data Module 

```python
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        dataset_transforms: dict(),
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_transforms = dataset_transforms
        
    def setup(self, stage: Optional[str] = None):
        self.train = ImageNet(self.data_root, self.dataset_transforms) 
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, \
            num_workers=self.num_workers, drop_last=True)
```

---

## PyTorch Lightning Module

``` python
class resnet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self,batch):
        x, labels = batch
        pred=self.forward(x)
        train_loss = F.cross_entropy(pred, labels)
        self.log("training_loss", train_loss)
    
        return train_loss

    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

```

--- 

## One GPU training 

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 1. Organize the data
datamodule = ImageNetDataModule("/p/scratch/training2324/data/", 256, \
    int(os.getenv('SLURM_CPUS_PER_TASK')), transform)
# 2. Build the model using desired Task
model = resnet50Model()
# 3. Create the trainer
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu")
# 4. Train the model
trainer.fit(model, datamodule=datamodule)
# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
```

---

## One GPU training 

``` bash 
#!/bin/bash -x
#SBATCH --nodes=1            
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --reservation=ai_sc_day2

# To get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# activate env
source $HOME/course/$USER/sc_venv_template/activate.sh
# run script from above
time srun python3 gpu_training.py
```

```bash
real	342m11.864s
```

---

## DEMO

---

## But what about many GPUs?

- It's when things get interesting

---

## Data Parallel

![](images/data-parallel.svg)

---

## Data Parallel

![](images/data-parallel-multiple-data.svg)

---

## Data Parallel - Averaging

![](images/data-parallel-averaging.svg)

---

## Data Parallel

### There are other approaches too!

- For the sake of completeness:
    - Asynchronous Stochastic Gradient Descent
        - Don't average the parameters, but send the updates (gradients post learning rate and momentum) asynchronously
        - Advantageous for slow networks
        - Problem: stale gradient (things might change while calculating gradients)
        - The more nodes, the worse it gets
        - Won't talk about it anymore

---

## Data Parallel

### There are other approaches too!

- Decentralized Asychronous Stochastic Gradient Descent
    - Updates are peer-to-peer
    - The updates are heavily compressed and quantized
    - Disadvantage: extra computation per minibatch, more memory needed

- WE DON'T NEED THOSE


--- 

## Multi-GPU training

1 node and 4 GPU

```bash
#!/bin/bash -x
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:4                  # Use the 4 GPUs available
#SBATCH --ntasks-per-node=4           # When using pl it should always be set to 4
#SBATCH --mem=0
#SBATCH --cpus-per-task=24            # Divide the number of cpus (96) by the number of GPUs (4)
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --reservation=ai_sc_day2

export CUDA_VISIBLE_DEVICES=0,1,2,3    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/$USER/sc_venv_template/activate.sh
time srun python3 gpu_training.py
```

```bash
real	89m15.923s
```

---

## DEMO

---

## That's it for data parallel!

- Copy of the model on each GPU
- Use different data for each GPU
- Everything else is the same
- Average after each epoch

---

## There are more levels!

![](images/lets-go-deeper.jpg)

--- 

## Data Parallel - Multi Node

![](images/data-parallel-multi-node.svg)

---

## Data Parallel - Multi Node

![](images/data-parallel-multi-node-averaging.svg)

---

## Before we go further...

- Data parallel is usually good enough 👌
- If you need more than this, you should be giving this course, not me 🤷‍♂️

---

## Model Parallel

- Model *itself* is too big to fit in one single GPU 🐋
- Each GPU holds a slice of the model 🍕
- Data moves from one GPU to the next

---

## Model Parallel

![](images/model-parallel.svg)

---


## Model Parallel

![](images/model-parallel-pipeline-1.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-2.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-3.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-4.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-5.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-6.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-7.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-8.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-9.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-10.svg)

---

## What's the problem here? 🧐

---

## Model Parallel

- Waste of resources
- While one GPU is working, others are waiting the whole process to end
- ![](images/no_pipe.png)
    - [Source: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)


---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-1.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-2-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-3-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-4-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-5-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-6-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-7-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-8-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-9-multibatch.svg)

---

## This is an oversimplification!

- Actually, you split the input minibatch into multiple microbatches.
- There's still idle time - an unavoidable "bubble" 🫧
- ![](images/pipe.png)

---

## Model Parallel - Multi Node

- In this case, each node does the same as the others. 
- At each step, they all synchronize their weights.

---

## Model Parallel - Multi Node

![](images/model-parallel-multi-node.svg)

---

## Model Parallel - going bigger

- You can also have layers spreaded over multiple gpus
- One can even pipeline among nodes....

---

## Recap

- Data parallelism:
    - Split the data over multiple GPUs
    - Each GPU runs the whole model
    - The gradients are averaged at each step
- Data parallelism, multi-node:
    - Same, but gradients are averaged across nodes
- Model parallelism:
    - Split the model over multiple GPUs
    - Each GPU does the forward/backward pass
    - The gradients are averaged at the end
- Model parallelism, multi-node:
    - Same, but gradients are averaged across nodes

---


## Parallel Training with PyTorch DDP

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
<!-- - MASTER_ADDR: address of rank 0 node. -->

---

## DDP steps

1. Set up the environement variable for the distributed mode (WORLD_SIZE, RANK, LOCAL_RANK ...)

- ```python
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

2. Initialize a sampler to specify the sequence of indices/keys used in data loading.
3. Implements data parallelism of the model. 
4. Allow only one process to save checkpoints.

- ```python
datamodule = ImageNetDataModule("/p/scratch/training2324/data/", 256, \
    int(os.getenv('SLURM_CPUS_PER_TASK')), transform)
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", num_nodes=nnodes)
trainer.fit(model, datamodule=datamodule)
trainer.save_checkpoint("image_classification_model.pt")
```

---

## DDP steps

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 1. The number of nodes
nnodes = os.getenv("SLURM_NNODES")
# 2. Organize the data
datamodule = ImageNetDataModule("/p/scratch/training2324/data/", 128, \
    int(os.getenv('SLURM_CPUS_PER_TASK')), transform)
# 3. Build the model using desired Task
model = resnet50Model()
# 4. Create the trainer
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", num_nodes=nnodes)
# 5. Train the model
trainer.fit(model, datamodule=datamodule)
# 6. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
```

---

## DDP training

16 nodes and 4 GPU each 

```bash
#!/bin/bash -x
#SBATCH --nodes=16                     # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4                   # Use the 4 GPUs available
#SBATCH --ntasks-per-node=4            # When using pl it should always be set to 4
#SBATCH --mem=0
#SBATCH --cpus-per-task=24             # Divide the number of cpus (96) by the number of GPUs (4)
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --reservation=ai_sc_day2

export CUDA_VISIBLE_DEVICES=0,1,2,3    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/$USER/sc_venv_template/activate.sh
time srun python3 ddp_training.py
```

```bash
real	6m56.457s
```

---

## DDP training

With 4 nodes: 

```bash
real	24m48.169s
```

With 8 nodes: 

```bash
real	13m10.722s
```

With 16 nodes: 

```bash
real	6m56.457s
```

With 32 nodes: 

```bash
real	4m48.313s
```
---

## Data Parallel

<!-- What changed? -->

- It was 
- ```python
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu")
``` 
- Became 
- ```python
nnodes = os.getenv("SLURM_NNODES")
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", num_nodes=nnodes)
```

---

## Data Parallel

<!-- What changed? -->

- It was
- ```bash
#SBATCH --nodes=1                
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
```
- Became
- ```bash
#SBATCH --nodes=16                   # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4                 # Use the 4 GPUs available
#SBATCH --ntasks-per-node=4          # When using pl it should always be set to 4
#SBATCH --cpus-per-task=24           # Divide the number of cpus (96) by the number of GPUs (4)
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Very important to make the GPUs visible
```

---

## DEMO

--- 

## TensorBoard

```python 
# 3. Create the logger 
logger = TensorBoardLogger("tb_logs", name="resnet50")

# 4. Create the trainer and pass the logger 
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", \
    num_nodes=nnodes, logger=logger)
```

--- 

## TensorBoard

```bash
ssh  -L 16000:localhost:16000 booster
```

```bash
source $HOME/course/$USER/sc_venv_template/activate.sh
tensorboard --logdir=[PATH_TO_TENSOR_BOARD] --port=16000
```
![](images/tb.png){ width=750px }

---

## DEMO

---

## Llview

- ![](images/llview.png)

---

## DAY 2 RECAP 

- Write parallel code.
- Can submit single node, multi-gpu and multi-node training.
- Use TensorBoard on the supercomputer.

---

## ANY QUESTIONS??

#### Feedback is more than welcome!

---