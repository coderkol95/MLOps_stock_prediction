import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from ...jobs.data_download import get_ticker_data, get_dataset_tags

@pytest.fixture(scope="module", autouse=True)
def ticker():
    return "WIPRO.NS"

@pytest.fixture
def start():
    return 100

@pytest.fixture
def end():
    return 50

@pytest.fixture
def tickerDefault():
    tickerDefault=pd.DataFrame(index=pd.DatetimeIndex(['2023-04-18 00:00:00+05:30', '2023-04-19 00:00:00+05:30',
        '2023-04-20 00:00:00+05:30', '2023-04-21 00:00:00+05:30',
        '2023-04-24 00:00:00+05:30', '2023-04-25 00:00:00+05:30',
        '2023-04-26 00:00:00+05:30', '2023-04-27 00:00:00+05:30',
        '2023-04-28 00:00:00+05:30', '2023-05-02 00:00:00+05:30'],
        dtype='datetime64[ns, Asia/Kolkata]', name='Date', freq=None), data=np.random.random_integers(low=1,high=34,size=[10,4]))
    return tickerDefault

def test_get_ticker_data(ticker, start, end):

    tickerData=get_ticker_data(ticker=ticker,start=start, end=end)

    assert isinstance(tickerData, pd.DataFrame)    
    assert tickerData.shape[0]>0
    assert "Open" in tickerData.columns
    assert "Close" in tickerData.columns

def test_get_dataset_tags(tickerDefault):
        
    tags=get_dataset_tags(df=tickerDefault)

    assert len(list(tags.keys()))==5
    assert list(tags.keys())==['Length','Start','End','Median','SD']
    assert len(list(tags.values()))==5
