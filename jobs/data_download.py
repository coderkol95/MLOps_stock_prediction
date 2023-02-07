from datetime import datetime, timedelta
import logging
import yfinance as yf
import pandas as pd
import os
import pandas as pd
import re

def get_ticker_data(TICKER:str):

    """
    1. Download ticker data using YFinance
    2. Generate data tags
    3. Write out details to data_upload.yml

    """

    # try:
    start = str(datetime.today().date()-timedelta(days=366))
    end = str(datetime.today().date()-timedelta(days=1))

    tickerData=yf.download(TICKER,start=start, end=end, period='1d')
    tickerData['Date']=[str(x)[:10] for x in tickerData.index]

    if tickerData.shape[0]==0:
        raise ValueError("No data found via YFinance.")

    logging.info(f"{os.getcwd()}")
    logging.info(f"Length of ticker data: {tickerData.shape[0]}")
    if tickerData.shape[0]!=0:

        tickerData = tickerData['Close']

        logging.info(f"Length of ticker data: {len(tickerData.index)}")

        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            print(f"filename={TICKER[:TICKER.index('.')]}", f)

        # Only persisting the latest in the repository
        path = f'../data/{TICKER}.csv'
        tickerData.to_csv(path,index=True)
        
        tags = get_dataset_tags(tickerData)
        save_to_data_upload(path, TICKER, tags)
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            print(f'downloaded={True}', f)
    # except:
    #     with open(os.environ['GITHUB_OUTPUT'],'a') as f:
    #         print(f'downloaded={False}', f)
    #     logging.error("Problem with downloading data from YFinance.")

def get_dataset_tags(df):

    tags={
            'Length':len(df),
            'Start':str(df.index[0].date()),
            'End':str(df.index[-1].date()),
            'Median':df.median(),
            'SD':df.std()}
    return tags

def save_to_data_upload(path, ticker, tags):

    # try:
    name=ticker[:ticker.index('.')]
    version=re.sub('-','',str(datetime.today().date()))
    description=f"Stock data for {TICKER} during {tags['Start']}:{tags['End']} in 1d interval."
    path=path
    tags=tags

    # Write to yaml file

    with open("../jobs/data_upload.yml","w") as f:

        f.write(
        f"""$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
        \ntype: uri_file\nname: '{name}'\ndescription: {description}\npath: '{path}'\ntags: {tags}\nversion: {version}""")

    logging.info(f"Uploaded stock data for {TICKER} during {tags['Start']}:{tags['End']} in 1d interval.")
    # except:
    #     logging.error("Problem with persisting data to Azure ML datastore.")

if __name__=="__main__":

    TICKER="LT.NS"
    get_ticker_data(TICKER=TICKER)