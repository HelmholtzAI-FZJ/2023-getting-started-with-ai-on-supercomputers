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

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
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

    total_loss /= len(data_loader)
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
    
    print("Loading training data")
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


    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                    for x in ['train', 'val', 'test']}

    return dataloaders


def main(args):

    writer = SummaryWriter(args.tb_dir)
    device = torch.device(args.device)

    dataloaders = load_data(args)

    print("Creating model")
    model = resnet50(True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, criterion, optimizer, dataloaders["train"], device, epoch)
        evaluate(model, criterion, dataloaders["val"], device=device)

        writer.add_scalar('loss/train', train_loss, epoch)

        if args.log:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(args.log, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    utils.mkdir(args.tb_dir)
    
    main(args)