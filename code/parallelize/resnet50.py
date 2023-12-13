import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

class resnet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self,batch):
        x, labels = batch
        pred=self.forward(x)
        train_loss = F.cross_entropy(pred, labels)
        self.log("training_loss", train_loss)
    
        return train_loss

    def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

