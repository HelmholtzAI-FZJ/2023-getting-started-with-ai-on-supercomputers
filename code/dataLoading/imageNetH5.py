import os 
from io import BytesIO

import click
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

class ImageNetH5(Dataset):

    def __init__(self, train_data_path, split, transform=None):
        self.h5file = h5py.File(train_data_path, 'r')[split]  
        self.imgs = self.h5file["images"]
        self.targets = self.h5file["targets"]
        self.transform = transform

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img_string = self.imgs[idx]
        target = self.targets[idx]

        with BytesIO(img_string) as byte_stream:
            img = Image.open(byte_stream)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
    
        return img, target

@click.command()
@click.option("--h5_path", "-r")
def main(h5_path):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    image_datasets = ImageNetH5(h5_path, "train", transform) 
    dataloadersh5= DataLoader(image_datasets, batch_size=1024, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')), pin_memory=True)
    
    for _ in dataloadersh5:
        pass

if __name__ == "__main__":
    main()