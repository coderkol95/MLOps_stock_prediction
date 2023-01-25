import pandas as pd
import numpy as np
import pickle
import torch
import os
import logging
import json

def init():
    
    global model
    global scaler

    class lstm_model(torch.nn.Module):

        def __init__(self):
            super(lstm_model, self).__init__()
            self.lstm1=torch.nn.LSTM(batch_first=True, input_size=5, hidden_size=1)
            self.out=torch.nn.Linear(1,1)

        def forward(self, x, hidden=None):
            x, hidden = self.lstm1(x)
            x = x[:,-1]
            x = self.out(x)
            return x, hidden

    scalerpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/scaler.pkl")
    # deserialize the model file back into a sklearn model
    with open(scalerpath, 'rb') as f:
        scaler = pickle.load(f)

    modelpath = os.path.join(
    os.getenv("AZUREML_MODEL_DIR"), "outputs/modelstock_pred_2023-01-25.pth")    
    model = lstm_model()
    model.load_state_dict(torch.load(modelpath))
    model.eval()

def run(raw_data):

    data = json.loads(raw_data)
    data = np.array(list(data.values())).astype(float)
    scaled_data = scaler.transform(data.reshape(-1,1))

    tensor_data = torch.from_numpy(scaled_data.reshape(-1,1,5))

    result, _ = model(tensor_data.float())
    
    result = scaler.inverse_transform(result.detach().numpy())

    return result.tolist()
