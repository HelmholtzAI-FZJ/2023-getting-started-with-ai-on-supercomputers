import os 
import pickle 

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



@click.command()
@click.option("--data_root", "-r")
def main(data_root):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    image_datasets = ImageNet(data_root, transform) 
    dataloaders = DataLoader(image_datasets, batch_size=1024, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')))

    for _ in dataloaders:
        pass


if __name__ == "__main__":
    main()