import os
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

import utils

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
trainer = flash.Trainer(max_epochs=50,  accelerator="gpu", devices=ntasks, num_nodes=nnodes)

# 4. Train the model
trainer.fit(model, datamodule=datamodule)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")




