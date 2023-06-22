import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet50

class resnet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)