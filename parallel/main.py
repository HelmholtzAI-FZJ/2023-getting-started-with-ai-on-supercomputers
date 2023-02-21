import argparse
import os
import datetime
import time 

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from torchvision import transforms

from resnet import generate_model
    
def main(args):

    print(args)
    
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
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                    for x in ['train', 'val', 'test']}

    
    model = generate_model(args.arch)
    model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(args.tb_dir)
  
    print("Start training ...")

    start = time.time()
    train_loop(args, dataloaders["train"], dataloaders["val"], model, loss_fn, optimizer, writer, args.log)
    duration = time.time() - start

    print("The training took: ", duration)

    print("Start testing ...")
    test_loop(dataloaders["test"],model,loss_fn)


def train_loop(args, train_dataloader, val_loader, model, loss_fn, optimizer, log, PATH):

    size = len(train_dataloader.dataset)

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

        total_loss /= size 
        correct = 100 * correct / size
        
        print(f"Epoch : {t}, loss: {total_loss:>7f}, acc: {correct} ")
        
        torch.save(model.state_dict(), PATH+"/epoch"+str(t)+".pth")

        test_loss, acc = test_loop(val_loader, model, loss_fn)

        log.add_scalar('loss/train', total_loss, t)
        log.add_scalar('loss/valid', test_loss, t)
        log.add_scalar('acc/train_acc', correct, t)
        log.add_scalar('acc/val_acc', acc, t)
        
        
def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    test_loss, correct = 0.0, 0.0
    
    model.eval()

    for X, y in dataloader:
        X = X.cuda()
        y = y.cuda()

        pred = model(X)

        test_loss += loss_fn(pred, y)
        correct += (pred.argmax(1) == y).float().sum()
           
    test_loss /= size
    correct = 100*correct / size
    print(f"Test Error: Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            
    return test_loss, correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--tb_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--arch', type=int, default=18)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.log = os.path.join(args.log, now)
    args.tb_dir = os.path.join(args.log, "tensor_board")
    
    main(args)