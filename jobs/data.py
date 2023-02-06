import json
from datetime import datetime, timedelta
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
import logging
from azure.identity import DefaultAzureCredential
import yfinance as yf
import re
import pandas as pd

# logging.basicConfig(filename='../logs.log', encoding='utf-8', level=logging.INFO)

def get_ml_client():

    with open('../config/config.json','r') as f:
        configs=json.loads(f.read())

    subscription_id, resource_group, workspace = configs['subscription_id'], configs['resource_group'], configs['workspace_name']

    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")

        ml_client = MLClient(
        credential, subscription_id, resource_group, workspace
        )

    except Exception as ex:

        logging.error("Could not instantiate ML Client")

    return ml_client

def get_ticker_data(TICKER:str):

    try:
        start = str(datetime.today().date()-timedelta(days=366))
        end = str(datetime.today().date()-timedelta(days=1))

        tickerData=yf.download(TICKER,start=start, end=end, period='1d')
        tickerData['Date']=[str(x)[:10] for x in tickerData.index]

        if tickerData.shape[0]==0:
            raise ValueError("No data found via YFinance.")

        tickerData['Ticker']=TICKER
        tickerData = tickerData['Adj Close']
        logging.info(f"Length of ticker data: {len(tickerData.index)}")
        path = f'../data/{TICKER}.csv'
        tickerData.to_csv(path)
    except:
        logging.error("Problem with downloading data from YFinance.")
        path=None
    finally:
        return path

def upload_data(ml_client, path, TICKER):

    try:
        ticker = TICKER[:TICKER.index('.')]
        ticker_data = pd.read_csv(path)
        ticker_data.index=ticker_data['Date']
        ticker_data.drop(["Date"],axis=1,inplace=True)
        data_to_upload=Data(
            name=ticker,
            version=re.sub('-','',str(datetime.today().date())),
            description=f"Stock data for {TICKER} during {str(ticker_data.index[0][:10])}:{str(ticker_data.index[-1][:10])} in 1d interval.",
            type='uri_file',
            path=path,
            tags={
                'Length':len(ticker_data),
                'Start':str(ticker_data.index[0][:10]),
                'End':str(ticker_data.index[-1][:10]),
                'Median':ticker_data.median(),
                'SD':ticker_data.std()}
        )
        ml_client.data.create_or_update(data_to_upload)

        logging.info(f"Uploaded stock data for {TICKER} during {str(ticker_data.index[0][:10])}:{str(ticker_data.index[-1][:10])} in 1d interval.")
    except:
        logging.error("Problem with persisting data to Azure ML datastore.")

if __name__=="__main__":

    ml_client = get_ml_client()
    TICKER="LT.NS"
    path = get_ticker_data(TICKER=TICKER)
    print(f"path: {path}")
    if path is not None:
        upload_data(ml_client,path, TICKER=TICKER)
