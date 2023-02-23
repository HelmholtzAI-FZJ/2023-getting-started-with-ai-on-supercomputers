import os
import pickle
import h5py 
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset

class ImagenetH5(Dataset):

    def __init__(self, h5_file, subset, transform=None):

        self.imgs = h5py.File(h5_file, 'r')[subset] 
    
        self.img_ids = [img_id for img_id in self.imgs.keys()]
            
        self._transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        if not 0 <= (index) < len(self.img_ids):
            raise IndexError(index)

        idx = self.imgs[self.img_ids[index]]
        img = idx["image"][:]
        label = idx["label"][()]

        if self._transform:
            img = self._transform(img)
    
        return img, label



class Imagenet(Dataset):

    def __init__(self, root, img_files, labels, subset, transform=None):

        self.root = root
        self.subset = subset
       
        with open(img_files, 'rb') as f:
            self.imgs = pickle.load(f)
       
        with open(labels, 'rb') as f:
            self.labels = pickle.load(f)
            
        self._transform = transform

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        if not 0 <= (index) < len(self.imgs):
            raise IndexError(index)

        idx = self.imgs[index]
        label = self.labels[idx.split("/")[0]]
        if self.subset == "val": 
            idx = idx.split("/")[-1]+".JPEG"
        img = Image.open(os.path.join(self.root,idx))
        img = img.convert('RGB')

        

        if self._transform:
            img = self._transform(img)
    
        return img, label

