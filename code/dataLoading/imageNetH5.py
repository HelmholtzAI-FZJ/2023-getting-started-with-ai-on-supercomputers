import os 
from io import BytesIO
import argparse
import json
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

class ImageNetH5(Dataset):

    def __init__(self, data_root, split, transform=None):

        self.imgs = h5py.File(os.path.join(data_root, "ImageNetFinal.h5"), 'r')[split] 
    
        self.targets = []
        self.transform = transform[split]
        self.syn_to_class = {}

        with open(os.path.join(data_root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        with open(os.path.join(data_root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        samples_dir = os.path.join(data_root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                self.targets.append(target)

    def __len__(self) -> int:
        return self.imgs["images"].shape[0]

    def __getitem__(self, index: int):
        img_string = self.imgs["images"][index]

        with BytesIO(img_string) as byte_stream:
            img = Image.open(byte_stream)
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
    
        return img, self.targets[index]


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/p/scratch/training2303/data/")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    image_datasets = ImageNetH5(args.data_root, "train", transformation()) 
    dataloadersh5= DataLoader(image_datasets, batch_size=args.batch_size, num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')), pin_memory=True)
    
    print("Start loading with H5 file")
    for x in dataloadersh5:
        pass
        