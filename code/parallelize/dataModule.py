import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

sys.path.append("..")
sys.path.append("2023-getting-started-with-ai-on-supercomputers/code/")

from dataLoading.imageNet import ImageNet

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
        self.dataset_transforms = dataset_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 1000

    # no need for prepare_data
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = ImageNet(self.data_root, "train",  self.dataset_transforms) 
            
        if stage == "test" or stage is None:
            self.val = ImageNet(self.data_root, "val",  self.dataset_transforms) 


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

