import os
import argparse
import time
import datetime

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from resnet import resnet50 
import data_loader
import utils

def train_one_epoch(model, criterion, optimizer, data_loader, sampler, device, epoch):
    model.train()
    total_loss = 0.0
    
    for image, target in data_loader:  

        image, target = image.to(device), target.type(torch.LongTensor).to(device)
        output = model(image)
        
        loss = criterion(output, target)
        total_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()


    # Total loss is devided by the number of 
    total_loss /= len(sampler)
    
    torch.distributed.all_reduce(total_loss)

    if utils.is_main_process():
        print("Epoch {}: avg_loss {}".format(epoch, total_loss))

    return total_loss
     

def evaluate(model, criterion, data_loader, device):
    model.eval()

    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.type(torch.LongTensor).to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

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


# Data loading code
def load_h5data(args):

    dataset_transforms = transformation()

    image_datasets = {x: data_loader.ImagenetH5(args.imagenet_root, args.h5_file, x, dataset_transforms[x]) 
                    for x in ['train', 'val']}


    return image_datasets

def load_data(args):
    dataset_transforms = transformation()

    image_datasets = {x: data_loader.ImageNetKaggle(args.imagenet_root, x, dataset_transforms[x]) 
                    for x in ['train', 'val']}

    return image_datasets


def main(args):
    
    # set up the environement variable for the distributed mode
    utils.init_distributed_mode(args.master_port)

    # Initialize the torch.distributed package
    torch.distributed.init_process_group(backend='nccl')

    # return an object representing the device on which tensors will be allocated.
    device = torch.device(args.device)

    # enable benchmark mode in cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True
    
    # only process rank 0 initialize the SummaryWriter.
    if utils.is_main_process():
        print(args)
        writer = SummaryWriter(args.tb_dir)
    
   
    if args.kaggle:
        image_datasets = load_data(args)
    else:
        image_datasets = load_h5data(args)
    

    # sampler is used to specify the sequence of indices/keys used in data loading
    datasets_sampler = {x: torch.utils.data.distributed.DistributedSampler(image_datasets[x])
                    for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, sampler=datasets_sampler[x], num_workers=args.workers, pin_memory=True)
                    for x in ['train', 'val']}
    

    model = resnet50()
    model.to(device)

    # Synchronization of batchnorm statistics
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # implements data parallelism of the model which can run across multiple machines. 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_time = time.time()
    for epoch in range(args.epochs):
        # it’s necessary to use set_epoch to guarantee a different shuffling order
        datasets_sampler["train"].set_epoch(epoch)

        train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], datasets_sampler["train"], device, epoch)

        evaluate(model, criterion, dataloaders["val"], device=device)
           
        # Only process rank 0 can add scalar data to summary
        if utils.is_main_process():
            writer.add_scalar('loss/train', train_loss, epoch)
        
        # Only process rank 0 can save checkpoints
        if args.log:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.log, 'checkpoint.pth'))
        
           
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if utils.is_main_process():
            print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--master_port', type=int, default=12354)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gpu', type=list, default=[0,1,2,3])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--kaggle', type=bool, default=False)
    parser.add_argument('--h5_file', type=str, default="/p/scratch/training2303/data/new_imagenet2.h5")
    parser.add_argument('--imagenet_root', type=str, default="/p/scratch/training2303/data/")
    parser.add_argument('--log', type=str)
    parser.add_argument('--tb_dir', type=str)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log = os.path.join(args.log, now)
    utils.mkdir(args.log)
    args.tb_dir = os.path.join(args.log, "tensor_board")
    
    main(args)