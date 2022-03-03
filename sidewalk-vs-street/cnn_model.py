
import torch.nn as nn
import torch 
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers, Trainer
import torchmetrics
from dataloader import SidewalkDataSet



class SidewalkClassifier(pl.LightningModule):

    def __init__(self, train_path, val_path, path_to_constants, use_dropout=False):
        #architecture form A Deep Learning Model for Transportation Mode Detection Based on Smartphone Sensing Data"
        super().__init__()
        self.layerin = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=32, kernel_size=15, stride=1), nn.LeakyReLU())
        self.maxpool_in = nn.MaxPool1d(kernel_size=4, stride=1)
        self.layer1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU())
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.layer2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1), nn.LeakyReLU())
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.layer3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=10, stride=1), nn.LeakyReLU())
        self.maxpool3 = nn.MaxPool1d(4,2)
        self.layer4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1), nn.LeakyReLU())
        self.maxpool4 = nn.MaxPool1d(4,2)
        self.layer5 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1), nn.LeakyReLU())
        self.maxpool5 = nn.MaxPool1d(4,2)
        self.layer6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1), nn.LeakyReLU())
        self.maxpool6 = nn.MaxPool1d(4,2)

        self.fn = nn.Linear(in_features=448, out_features=200)
        self.dropoutlayer = nn.Dropout(p=0.2)
        self.out = nn.Linear(in_features=200, out_features=2)
        self.sigmoid = nn.Sigmoid()

        #misc
        self.dropout = use_dropout
        self.accuracy = torchmetrics.Accuracy()
        self.loss = nn.BCELoss()
        self.train_path = train_path
        self.val_path = val_path
        self.constants = path_to_constants

    def forward(self,x):
        x = torch.squeeze(x).float()
        x1 = self.layerin(x)
        x1 = self.maxpool_in(x1)
        x2 = self.layer1(x1)
        x2 = self.maxpool1(x2)
        x3 = self.layer2(x2)
        x3 = self.maxpool2(x3)
        x4 = self.layer3(x3)
        x4 = self.maxpool3(x4)
        x5 = self.layer4(x4)
        x5 = self.maxpool4(x5)
        x6 = self.layer5(x5)
        x6 = self.maxpool5(x6)
        x7 = self.layer6(x6)
        x7 = self.maxpool6(x7)
        flat = x7.view((x7.shape[0], -1))
        out1 = self.fn(flat)
        if self.dropout:
            out1 = self.dropoutlayer(out1)
        output = self.out(out1)
        return self.sigmoid(output)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = torch.squeeze(y)
        x_hat = self.forward(x)
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = torch.squeeze(y).to(torch.float32)
        x_hat = self.forward(x).to(torch.float32)

        loss = self.loss(x_hat, y)
        y = y.to(torch.int32)
        val_acc = self.accuracy(x_hat, y)
        self.log('validation_loss', loss)
        self.log('val_accuracy', val_acc)

        return loss, val_acc

    def train_dataloader(self):
        train_set = SidewalkDataSet(self.train_path, self.constants)
        return DataLoader(train_set, batch_size=256, num_workers=2)
    
    def val_dataloader(self):
        val_set = SidewalkDataSet(self.val_path, self.constants)
        return DataLoader(val_set, batch_size=64, num_workers=2)


def run_trainer():
    window_size = 256
    model = SidewalkClassifier(window_size)

    early_stop_call_back = callbacks.EarlyStopping(
		monitor='val_loss',
		min_delta=0.00,
		patience=5,
		verbose=False,
		mode='max'
	)
    
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="val_loss")
    lr_callback = callbacks.LearningRateMonitor(logging_interval='epoch')
    logger = loggers.TensorBoardLogger(save_dir = 'logs/')
    print("using GPU", torch.cuda.is_available())
    trainer = Trainer(max_epochs=300,
					  #gpus=1,
					  #logger=logger, #use default tensorboard
					  log_every_n_steps=1, #log every update step for debugging
					  limit_train_batches=1.0,
					  limit_val_batches=1.0,
					  check_val_every_n_epoch=1,
					  callbacks=[early_stop_call_back, lr_callback, checkpoint_callback])
	
    trainer.fit(model)

if __name__ == "__main__":
    run_trainer()
