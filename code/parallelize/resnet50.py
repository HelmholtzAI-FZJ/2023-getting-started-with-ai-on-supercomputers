import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

class resnet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self,batch):
        # REQUIRED- run at every batch of training data
        # extracting input and output from the batch
        x,labels=batch
         
        # forward pass on a batch
        pred=self.forward(x)
 
        # calculating the loss
        train_loss = F.cross_entropy(pred, labels)
         
        # logs for tensorboard
        self.log("training_loss", train_loss)
    
        return train_loss



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)