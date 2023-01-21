import pandas as pd
import numpy as np
from azureml.core import Run, Model
import argparse
from TickerData import download, query
from Models import lstm_model
import torch
from datetime import datetime
import os

def query_data(args):

    try:
        data = query(ticker=args.ticker, start=args.start, end=args.end)
    except:
        download(ticker=args.ticker, start=args.start, end=args.end, period=args.period)
        data = query(ticker=args.ticker, start=args.start, end=args.end)
    return data

def training_data(series, loookaheadSize=5):
    X,y = [],[]
    for i in np.arange(5,len(series)-1):
        X.append(series[i-loookaheadSize:i])
        y.append(series[i+1])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(len(series)-loookaheadSize-1,1,5)
    y=y.reshape(-1,1)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return train_dataset

def train(args):

    if args.filePath is None:
        data=query_data(args)
        data = list(data[f"{args.ticker}_Close"].values())
    else:
        data = pd.read_csv(args.filePath).values.tolist()

    trainset = training_data(data)
    seq_model = lstm_model()
    optim = torch.optim.Adam(lr = 0.0001, params=seq_model.parameters())

    epochs = 10

    for epoch in np.arange(epochs):

        Loss=0

        for data in trainset:

            feats, target = data
            optim.zero_grad()

            y_p = seq_model(feats.float())
            loss = torch.nn.functional.mse_loss(y_p.float(), target.float())

            loss.backward()
            optim.step()
            Loss += loss.item()

        print(f"Epoch: {epoch}, loss: {Loss}")

    model_path = f"./models/{str(datetime.now().date())}.pth"
    torch.save(seq_model, path = model_path)
    Model.register(workspace=ws, model_name = "Test_model",model_path = model_path)

    run.complete()

if __name__ == "__main__":

    run = Run.get_context()
    ws = run.experiment.workspace
    parser = argparse.ArgumentParser()

    parser.add_argument("--ticker", help="Ticker to train")
    parser.add_argument("--start", help="Start of training period")
    parser.add_argument("--end", help="End of training period")
    parser.add_argument("--period", default='1d', help="Frequency of training data")    
    parser.add_argument("--filePath", help="Filepath of the file")    
    
    args = parser.parse_args()
    
    print("OS Path", os.getcwd())

    try:    
        if args.filePath is None:
            if args.ticker is None:
                raise ValueError("Please enter ticker which you want to train the model on. Available options: HDFCLIFE.NS, NESTLEIND.NS, KOTAKBANK.NS, INDUSINDBK.NS, TATASTEEL.NS, ITC.NS, ONGC.NS, TITAN.NS, ULTRACEMCO.NS, BAJAJFINSV.NS, BAJFINANCE.NS, BRITANNIA.NS, BAJAJ-AUTO.NS, COALINDIA.NS, BHARTIARTL.NS, TATACONSUM.NS, LTI.NS, CIPLA.NS, MARUTI.NS, ICICIBANK.NS, APOLLOHOSP.NS, NTPC.NS, HEROMOTOCO.NS, HINDALCO.NS, WIPRO.NS, TCS.NS, ADANIENT.NS, MM.NS, TECHM.NS, RELIANCE.NS")
            if args.start is None or args.end is None:
                raise ValueError("Please define both start and end for downloading data.")
    except:
        raise ValueError("Error with input args")

    train(args)    
    

    