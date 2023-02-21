import argparse
import os
import datetime
import time 

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

import horovod.torch as hvd

from resnet import generate_model
    
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()
    
def main(args):
        
    hvd.init()

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
    
    # load data

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

    datasets_sampler = {x: distributed.DistributedSampler(image_datasets[x], num_replicas=hvd.size(), rank=hvd.rank())
                    for x in ['train', 'val','test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, num_workers=args.num_workers, sampler=datasets_sampler[x])
                    for x in ['train', 'val', 'test']}

    model = generate_model(args.arch)
    model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    writer = SummaryWriter(args.tb_dir)
  
    start = time.time()
    train_loop(args, dataloaders["train"], dataloaders["val"], datasets_sampler["train"], datasets_sampler["val"], model, loss_fn, optimizer, writer, args.log)
    duration = time.time() - start

    test_loop(dataloaders["test"], datasets_sampler["test"], model, loss_fn)


def train_loop(args, train_dataloader, val_loader, train_sampler, val_sampler, model, loss_fn, optimizer, log, PATH):

    for t in tqdm(range(args.epochs)):
        
        model.train()
        total_loss, correct = 0.0, 0.0

        for X, y in train_dataloader:
            X = X.cuda()
            y = y.cuda()

            pred = model(X)

            loss = loss_fn(pred, y) 
            total_loss += loss
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            
            optimizer.step()
            correct += (pred.argmax(1) == y).float().sum()

        total_loss /= len(train_sampler)
        correct /= len(train_sampler)

        total_loss = metric_average(total_loss, 'avg_loss')
        correct = metric_average(correct, 'avg_accuracy')

        if hvd.rank() == 0:
            print('\nEpoch {}: Train set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                t, total_loss, 100. * correct))

            torch.save(model.state_dict(), PATH+"/epoch"+str(t)+".pth")

        test_loss, acc = test_loop(val_loader, val_sampler, model, loss_fn)

        if hvd.rank() == 0:
            log.add_scalar('loss/train', total_loss, t)
            log.add_scalar('loss/valid', test_loss, t)
            log.add_scalar('acc/train_acc', correct, t)
            log.add_scalar('acc/val_acc', acc, t)
        
        
def test_loop(dataloader, test_sampler, model, loss_fn):
    test_loss, test_accuracy = 0.0, 0.0
    
    model.eval()

    for X, y in dataloader:
        X = X.cuda()
        y = y.cuda()

        pred = model(X)

        test_loss += loss_fn(pred, y)
        test_accuracy += (pred.argmax(1) == y).float().sum()
           
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    test_loss = metric_average(test_loss, 'test_avg_loss')
    test_accuracy = metric_average(test_accuracy, 'test_avg_accuracy')

    if hvd.rank() == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))
            
    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--tb_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--arch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log = os.path.join(args.log, now)
    args.tb_dir = os.path.join(args.log, "tensor_board")

    main(args)