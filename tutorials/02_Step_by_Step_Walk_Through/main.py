import os
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
from typing import Optional


class LitMNIST(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # (bs, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sched]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = os.getcwd()
        self.batch_size = 32
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(self.mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == '__main__':
    model = LitMNIST()
    data = MNISTDataModule()
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=data)
