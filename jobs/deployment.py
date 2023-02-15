import pandas as pd
import numpy as np
import pickle
import torch
import os
import json
import pytorch_lightning as pl

def init():
    
    global mod
    global datamod
    global scaler

    class dataset(pl.LightningDataModule):

        def __init__(self, scaler):
            super(dataset,self).__init__()
            self.lookback_size = 5
            self.scaler=scaler
        
        def predict_tensors(self,df):
        
            X = []

            for i in np.arange(self.lookback_size, len(df)+1):
                X.append(df[i-self.lookback_size:i])
            
            X = np.array(X).reshape(-1,self.lookback_size,1)
            return torch.from_numpy(X).float()

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

        def predict_step(self,batch, batch_idx, dataloader_idx=0):
            X,_=batch
            return self(X.type(torch.float32))

    scalerpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/scaler.pkl")
    # deserialize the model file back into a sklearn model
    with open(scalerpath, 'rb') as f:
        scaler = pickle.load(f)

    modelpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/model.pth")    
    mod = model()
    mod.load_state_dict(torch.load(modelpath))
    mod.eval()

    datamod = dataset(scaler)

def run(raw_data):

    data = json.loads(raw_data)
    data = np.array(list(data.values())).astype(float)
    pred_data=datamod.predict_dataloader(data=pd.DataFrame(data, columns=['Close']))

    result, _ = mod(pred_data)
    
    result = scaler.inverse_transform(result.detach().numpy())

    return result.tolist()
