import yfinance as yf
import pandas as pd
import argparse
import pymongo

mongoServer = pymongo.MongoClient("mongodb://localhost:27017/")

def download(
    ticker:str,
    start:str,
    end:str,
    period:str
    ):
    """Download single stock data using YFinance and persist it in mongoDB """

    try:
        collection = mongoServer["TickerData"]["DayFreq"] 
    except:
        raise ValueError("Could not instantiate MongoDB collection.")

    try:
        tickerData= yf.download(tickers=ticker, start=start, end=end, period=period)
        tickerData['Date'] = [str(x)[:10] for x in tickerData.index]
        tickerData['Ticker'] = ticker
        collection.insert_many(list(tickerData.transpose().to_dict().values()))
    except:
        raise ValueError("Could not download ticker data using yfinance.")

def query(
    ticker:str,
    start:str,
    end:str,
    col=mongoServer["TickerData"]["DayFreq"],
    column="Close"
    ):
    """Query stock data previously persisted."""
    result = {}
    for i in col.find({"Ticker":ticker,"Date":{"$gte":start,"$lte":end}}):
        result.update({i['Date']:i[column]})
    return {f"{ticker}_{column}":result}

if __name__=="__main__":

    parser = argparse.parArgumentParser()

    parser.add_argument("--tickers", help="Ticker(s) you want to download", type=str)
    parser.add_argument("--action", help="Action-download, query", type=str)
    parser.add_argument("--start", help="Start date in format 'YYYY-MM-DD'", type=str)
    parser.add_argument("--end", help="End date in format 'YYYY-MM-DD'", type=str)
    parser.add_argument("--freq", help="Frequency of data reqd., '1d','5d','1mo','3mo' etc.", default='1d', type=str)

    args = parser.parse_args()

    if args.action == "download":

        if args.tickers is None:
            raise ValueError("Please enter tickers which you want to download. Available options: HDFCLIFE.NS, NESTLEIND.NS, KOTAKBANK.NS, INDUSINDBK.NS, TATASTEEL.NS, ITC.NS, ONGC.NS, TITAN.NS, ULTRACEMCO.NS, BAJAJFINSV.NS, BAJFINANCE.NS, BRITANNIA.NS, BAJAJ-AUTO.NS, COALINDIA.NS, BHARTIARTL.NS, TATACONSUM.NS, LTI.NS, CIPLA.NS, MARUTI.NS, ICICIBANK.NS, APOLLOHOSP.NS, NTPC.NS, HEROMOTOCO.NS, HINDALCO.NS, WIPRO.NS, TCS.NS, ADANIENT.NS, MM.NS, TECHM.NS, RELIANCE.NS")
        if args.start is None or args.end is None:
            raise ValueError("Please define both start and end for downloading data.")

        tickers = args.tickers.split() if len(args.tickers)>1 else [args.tickers]

        for tick in tickers:
            download(ticker=tick, start=args.start, end=args.end, period=args.freq)
    
    if args.action == "query":
        if args.tickers is None:
            raise ValueError("Please enter ticker(s) which you want to query.")
        if args.start is None or args.end is None:
            raise ValueError("Please define both start and end for querying data.")

        tickers = args.tickers.split() if len(args.tickers)>1 else [args.tickers]
        for tick in tickers:
            query(ticker=tick, start=args.start, end=args.end)
