import os
import argparse
from sklearn.metrics import mean_absolute_percentage_error
import pytorch_lightning as pl
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class dataset(pl.LightningDataModule):

    def __init__(self, data=None, scaler=None):
        super(dataset,self).__init__()
        self.lookback_size = 5
        self.batch_size = 32

        if data is not None:
            self.data=data
            self.series = pd.read_csv(self.data, index_col='Date')
            # Train:valid:test = 80:10:10
            self.train_df, self.valid_df = train_test_split(self.series, test_size=0.2)
            self.valid_df, self.test_df = train_test_split(self.valid_df, test_size=0.5)

        if scaler is None and data is not None:
            self.scaler = MinMaxScaler().fit(self.train_df)
        else:
            self.scaler=scaler

    def train_tensors(self,df):

        X, y = [], []

        for i in np.arange(self.lookback_size, len(df)-1):
            X.append(df[i-self.lookback_size:i])
            y.append(df[i+1])

        X = np.array(X).reshape(-1,self.lookback_size,1)
        y = np.array(y).reshape(-1,1)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    def predict_tensors(self,df):
    
        X = []

        for i in np.arange(self.lookback_size, len(df)+1):
            X.append(df[i-self.lookback_size:i])
        
        X = np.array(X).reshape(-1,self.lookback_size,1)
        return torch.from_numpy(X).float()

    def setup(self, stage=None):
        
        self.train_df = self.scaler.transform(self.train_df) 
        self.valid_df = self.scaler.transform(self.valid_df) 
        self.test_df = self.scaler.transform(self.test_df) 

        self.train_data = self.train_tensors(self.train_df)
        self.valid_data = self.train_tensors(self.valid_df)
        self.test_data = self.train_tensors(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
       return DataLoader(self.valid_data, batch_size=self.batch_size)

    def test_dataloader(self):
       return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self, data):
        self.pred_df= self.scaler.transform(data)
        self.pred_data = self.predict_tensors(self.pred_df)
        return self.pred_data

class model(pl.LightningModule):

    def __init__(self,lookback_size=5):

        super(model,self).__init__()

        self.lookback_size = lookback_size
        self.lstm=torch.nn.LSTM(batch_first=True, input_size=1, hidden_size=self.lookback_size)
        self.out=torch.nn.Linear(5,1)
        self.loss=torch.nn.functional.mse_loss

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x)
        x = x[:,-1]
        x = self.out(x)
        return x, hidden

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-3)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [early_stop, checkpoint]
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        logits,_ = self.forward(x.type(torch.float32)) 
        loss = self.loss(logits.float(), y.float()) 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        logits,_ = self.forward(x.type(torch.float32)) 
        loss = self.loss(logits.float(), y.float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx): 
        x, y = test_batch 
        logits,_ = self.forward(x.type(torch.float32)) 
        loss = self.loss(logits.float(), y.float())
        self.log("test_loss", loss)

    def predict_step(self,batch, batch_idx, dataloader_idx=0):
        X,_=batch
        return self(X.type(torch.float32))

def train(args):

    trainer=pl.Trainer(max_epochs=5)
    datamod=dataset(args.data)
    mod=model()
    trainer.fit(model=mod, datamodule=datamod)

    trainer.test(model=mod, datamodule=datamod)

    return mod, datamod.scaler

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")

    args = parser.parse_args()
    
    # Load Scaler object later and send it for scaling data

    trainedModel, scalerObj = train(args)

    pickle.dump(scalerObj, open('./outputs/scaler.pkl','wb'))
    model_file = f"./outputs/model.pth"
    torch.save(trainedModel.state_dict(), model_file)

if __name__ == "__main__":
    main()
