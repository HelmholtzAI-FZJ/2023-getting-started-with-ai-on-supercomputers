import os

import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from dataModule import ImageNetDataModule
from resnet50 import resnet50Model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# The number of nodes
nnodes = os.getenv("SLURM_NNODES")

# 1. Organize the data
datamodule = ImageNetDataModule("/p/scratch/training2324/data/", 128, int(os.getenv('SLURM_CPUS_PER_TASK')), transform)

# 2. Build the model using desired Task
model = resnet50Model()

# 3. Create the logger 
logger = TensorBoardLogger("tb_logs", name="resnet50")

# 4. Create the trainer
trainer = pl.Trainer(max_epochs=10,  accelerator="gpu", num_nodes=nnodes, logger=logger)

# 5. Train the model
trainer.fit(model, datamodule=datamodule)

# 6. Save the model!
trainer.save_checkpoint("image_classification_model.pt")




