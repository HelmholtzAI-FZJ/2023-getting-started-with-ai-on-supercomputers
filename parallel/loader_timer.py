import time 
import datetime

from tqdm import tqdm 

from torch.utils.data import DataLoader
from torchvision import transforms

from resnet import resnet50 
import data_loader

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




imagenet_root = "/p/scratch/training2303/data/"
h5_file = "/p/scratch/training2303/data/ImageNetFinal.h5"
batch_size = 2048
workers = 40

dataset_transforms = transformation()

image_datasets = data_loader.ImageNetKaggle(imagenet_root, "train", dataset_transforms["train"]) 
samplet_data torch.utils.data.distributed.DistributedSampler(image_datasets)
dataloaders = DataLoader(image_datasets, batch_size=batch_size, num_workers=workers, sampler=samplet_data, pin_memory=True)

print("Start loading without H5 file")
start_time = time.time()
for x in tqdm(dataloaders):
    x

end_time = time.time()
print("Time without h5 file: ", str(datetime.timedelta(seconds=int(end_time-start_time))))



image_datasets = data_loader.ImagenetH5(h5_file, "train", dataset_transforms["train"]) 
samplet_data torch.utils.data.distributed.DistributedSampler(image_datasets)
dataloadersh5= DataLoader(image_datasets, batch_size=batch_size, num_workers=workers, sampler=samplet_data, pin_memory=True)
 
print("Start loading with H5 file")
start_time = time.time()
for x in tqdm(dataloadersh5):
    x
    
end_time = time.time()

print("Time with h5 file: ",str(datetime.timedelta(seconds=int(end_time-start_time))))