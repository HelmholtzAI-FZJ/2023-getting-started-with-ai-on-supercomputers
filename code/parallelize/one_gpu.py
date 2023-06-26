import os

import pytorch_lightning as pl 
from pytorch_lightning import seed_everything
from torchvision import transforms

from dataModule import ImageNetDataModule
from resnet50 import resnet50Model
import utils

def transformation():
    _IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
    _IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
    
    return dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),

            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([        
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))

# set the random seeds.
seed_everything(42)

# The number of total processes started by Slurm.
ntasks = os.getenv('SLURM_NTASKS')

# Index of the current process.
rank = os.getenv('SLURM_PROCID')

# Index of the current process on this node only.
local_rank = os.getenv('SLURM_LOCALID')

# The number of nodes
nnodes = os.getenv("SLURM_NNODES")

utils.init_distributed_mode(12354)

# 1. Organize the data
datamodule = ImageNetDataModule("/p/scratch/training2303/data/", 128, int(os.getenv('SLURM_CPUS_PER_TASK')), transformation())

# 2. Build the model using desired Task
model = resnet50Model()

# 3. Create the trainer
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu")

# 4. Train the model
trainer.fit(model, datamodule=datamodule)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")






