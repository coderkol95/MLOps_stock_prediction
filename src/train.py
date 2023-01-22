import os
import argparse
import pandas as pd
import torch
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import mlflow
import pickle

def series_to_tensors(series, lookaheadSize=5):

    X,y = [],[]
    for i in np.arange(5,len(series)-1):
        X.append(series[i-lookaheadSize:i])
        y.append(series[i+1])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(len(series)-lookaheadSize-1,1,5)
    y=y.reshape(-1,1)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return dataset

def dataprep(args):

    stockData = pd.read_csv(args.data)
    stock_train_df, stock_test_df = train_test_split(stockData, test_size=args.test_train_ratio)

    scaler = MinMaxScaler.fit(stock_train_df)
    stock_train_df = scaler.transform(stock_train_df)
    stock_test_df = scaler.transform(stock_test_df)

    train_tensors = series_to_tensors(stock_train_df)
    test_tensors = series_to_tensors(stock_test_df)

    return scaler, train_tensors, test_tensors

class lstm_model(torch.nn.Module):

    def __init__(self):
        super(lstm_model, self).__init__()
        self.lstm1=torch.nn.LSTM(batch_first=True, input_size=5, hidden_size=1)
        self.out=torch.nn.Linear(1,1)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm1(x)
        x = self.out(x)
        return x.flatten()

def train(trainset):

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
    return seq_model

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)

    args = parser.parse_args()
    
    # Load Scaler object later and send it for scaling data

    scaler, trainset, _ = dataprep(args)

    trainedModel = train(trainset)

    pickle.dump(scaler, open('scaler.pkl','wb'))
    model_file = f"modelstock_pred_{str(datetime.now().date())}.pth"
    torch.save(trainedModel, path = model_file)

    # Registering the model to the workspace

    # job_name = "<JOB_NAME>"

    # run_model = Model(
    #     path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/scaler.pkl",
    #     name="MinMaxScaler,
    #     description="Scaler object",
    #     type=AssetTypes.MLFLOW_MODEL,
    # )
    #     run_model = Model(
    #     path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/{model_file}",
    #     name="run-model-example",
    #     description="Model created from run.",
    #     type=AssetTypes.MLFLOW_MODEL,
    # )

if __name__ == "__main__":
    main()
