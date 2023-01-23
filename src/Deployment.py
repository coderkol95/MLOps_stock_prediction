import pandas as pd
import numpy as np
import pickle
import torch
import os
import logging
import json

class lstm_model(torch.nn.Module):

    def __init__(self):
        super(lstm_model, self).__init__()
        self.lstm1=torch.nn.LSTM(batch_first=True, input_size=5, hidden_size=1)
        self.out=torch.nn.Linear(1,1)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm1(x)
        x = self.out(x)
        return x.flatten()

def init():
    
    global model
    global scaler

    scalerpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/scaler.pkl")
    print(scalerpath)
    # deserialize the model file back into a sklearn model
    with open(scalerpath, 'rb') as f:
        scaler = pickle.load(f)

    modelpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/modelstock_pred_2023-01-23.pth")    
    print(modelpath)
    model = lstm_model()
    model = torch.load(modelpath)

def run(raw_data):

    data = json.loads(raw_data)["data"]
    data = np.array(data)
    scaled_data = scaler.transform(data)

    result = model.predict(scaled_data)

    return result.tolist()
