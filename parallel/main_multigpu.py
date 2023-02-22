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
import utils

def train_one_epoch(model, criterion, optimizer, data_loader, sampler, device, epoch):
    model.train()
    total_loss = 0.0

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        total_loss += loss
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

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
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

# Data loading code
def load_data(args):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) 
                    for x in ['train', 'val','test']}

    datasets_sampler = {x: torch.utils.data.distributed.DistributedSampler(image_datasets[x])
                    for x in ['train', 'val','test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, sampler=datasets_sampler[x], num_workers=args.workers, pin_memory=True)
                    for x in ['train', 'val', 'test']}

    return datasets_sampler["train"], dataloaders


def main(args):


    
    utils.init_distributed_mode(args.master_port)
    torch.distributed.init_process_group(backend='nccl')

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    if utils.is_main_process():
        writer = SummaryWriter(args.tb_dir)
        
    train_sampler, dataloaders = load_data(args)

    model = resnet50()
    model.to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    start_time = time.time()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], train_sampler, device, epoch)
        evaluate(model, criterion, dataloaders["val"], device=device)
  
        writer.add_scalar('loss/train', train_loss, epoch)

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
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--master_port', type=int, default=12354)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gpu', type=list, default=[0,1,2,3])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--tb_dir', type=str)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log = os.path.join(args.log, now)
    utils.mkdir(args.log)
    args.tb_dir = os.path.join(args.log, "tensor_board")
    
    main(args)