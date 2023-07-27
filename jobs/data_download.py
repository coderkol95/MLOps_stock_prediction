from datetime import datetime, timedelta
import logging
import yfinance as yf
import pandas as pd
import os
import pandas as pd
import re
import argparse
import numpy as np

def get_ticker_data(
    ticker:str,
    start:int,
    end:int):

    """
    1. Download ticker data using YFinance
    2. Generate data tags
    3. Write out details to data_upload.yml

    """

    # try:
    start = str(datetime.today().date()-timedelta(days=start))
    end = str(datetime.today().date()-timedelta(days=end))

    tickerData=yf.download(ticker,start=start, end=end, period='1d')
    tickerData['Date']=[str(x)[:10] for x in tickerData.index]

    if tickerData.shape[0]==0:
        raise ValueError("No data found via YFinance.")

    logging.info(f"{os.getcwd()}")
    logging.info(f"Length of ticker data: {tickerData.shape[0]}")
    if tickerData.shape[0]!=0:

        tickerData = tickerData['Close']

        logging.info(f"Length of ticker data: {len(tickerData.index)}")

            # Only persisting the latest in the repository
        path = f'../data/{ticker}.csv'
        tickerData.to_csv(path,index=True)
        tags = get_dataset_tags(tickerData)
        save_to_data_upload(path, ticker, tags)
    # except:
    #     logging.error("Problem with downloading data from YFinance.")
    
def get_dataset_tags(df):

    """
    Return dataset tags for storing against Azure dataset.

    """

    tags={
            'Length':   len(df),
            'Start':    str(df.index[0].date()),
            'End':      str(df.index[-1].date()),
            'Median':   np.round(df.median(),2),
            'SD':       np.round(df.std(),2)}
    return tags

def save_to_data_upload(path, ticker, tags):

    """
    Write out the specifications to the Azure YAML file for uploading to Azure datastore.
    
    """

    try:
        name=ticker[:ticker.index('.')]
        version=re.sub('-','',str(datetime.today().date()))
        description=f"Stock data for {ticker} during {tags['Start']}:{tags['End']} in 1d interval."
        path=path
        tags=tags

        # Write to yaml file

        with open("../jobs/data_upload.yml","w") as f:

            f.write(
            f"""$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
            \ntype: uri_file\nname: '{name}'\ndescription: {description}\npath: '{path}'\ntags: {tags}\nversion: {version}""")

        logging.info(f"Uploaded stock data for {ticker} during {tags['Start']}:{tags['End']} in 1d interval.")
    except:
        logging.error("Problem with writing data to upload YAML file.")

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ticker',type=str, required=True, help="Ticker from YFinance that you want to download.")
    parser.add_argument('--start',type=str, help="Lookback period start in days, eg.366.", default=366)
    parser.add_argument('--end',type=str, help="Lookback period end in days, eg.1.", default=1)

    args = parser.parse_args()

    get_ticker_data(args.ticker, int(args.start), int(args.end))