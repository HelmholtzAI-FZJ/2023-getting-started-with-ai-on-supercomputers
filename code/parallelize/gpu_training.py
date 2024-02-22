import os

import pytorch_lightning as pl 
from torchvision import transforms

from dataModule import ImageNetDataModule
from resnet50 import resnet50Model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 1. Organize the data
datamodule = ImageNetDataModule("/p/scratch/training2402/data/", 256, \
    int(os.getenv('SLURM_CPUS_PER_TASK')), transform)

# 2. Build the model using desired Task
model = resnet50Model()

# 3. Create the trainer
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu")

# 4. Train the model
trainer.fit(model, datamodule=datamodule)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")






