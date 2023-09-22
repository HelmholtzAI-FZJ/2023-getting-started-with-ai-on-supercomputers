import os 
import pickle 
import time

import click
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNet(Dataset):

    def __init__(self, root, transform=None):
        self.root = root

        with open(os.path.join(self.root, "train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)

        self.samples = list(train_data.keys())
        self.targets = list(train_data.values())
        
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.root, self.samples[idx])).convert("RGB")
        if self.transform:
            x = self.transform(x)

        return x, self.targets[idx]

